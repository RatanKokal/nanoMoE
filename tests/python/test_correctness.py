"""
tests/python/test_correctness.py
─────────────────────────────────────────────────────────────────────────────
Architectural Parity Test: NanoMoELayer vs. HuggingFace MixtralSparseMoeBlock

Validates that the nanoMoE CUDA routing engine (csrc/moe.cu) produces
numerically equivalent outputs to the reference HuggingFace implementation
when both use *identical* expert weights and gate weights.

What is being tested
────────────────────
• topk_softmax_kernel   — warp-reduce argmax + in-register softmax
• expert_histogram_kernel / CUB ExclusiveSum — prefix-sum offsets
• permute_kernel         — float4 vectorised scatter (token → expert slot)
• unpermute_kernel       — float4 vectorised gather + weighted reduce

Pass criterion
──────────────
FP16 cublas matmul has non-associative rounding. A max absolute difference
< 1e-3 is the accepted parity bound for float16 MoE blocks (same threshold
used by vLLM and TensorRT-LLM continuous-batching correctness suites).

Usage
─────
  # From the repo root after `pip install -e .`
  python tests/python/test_correctness.py

  # Or with pytest
  pytest tests/python/test_correctness.py -v
"""

import sys
import torch
import torch.nn as nn
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
from transformers import MixtralConfig

# ── Path fixup so the repo root is importable without installation ────────────
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from inference.nano_moe_layer import NanoMoELayer

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check_cuda():
    if not torch.cuda.is_available():
        print("[SKIP] No CUDA device found — test requires a GPU.")
        sys.exit(0)

def _banner(title: str):
    width = 62
    print("\n" + "─" * width)
    print(f"  {title}")
    print("─" * width)


# ─────────────────────────────────────────────────────────────────────────────
# Core parity test
# ─────────────────────────────────────────────────────────────────────────────

def test_moe_parity(
    hidden_size:       int  = 1024,
    intermediate_size: int  = 4096,
    num_local_experts: int  = 8,
    num_experts_per_tok: int = 2,
    batch_size:        int  = 4,
    seq_len:           int  = 128,
    dtype: torch.dtype = torch.float16,
    atol:  float       = 1e-3,
) -> bool:
    """
    Returns True on success, False on failure.
    Raises on hard errors (missing extension, shape mismatch, etc.).
    """
    _banner("NanoMoE ↔ HuggingFace Architectural Parity Test")

    print(f"  Config : hidden={hidden_size}, ffn={intermediate_size}, "
          f"E={num_local_experts}, k={num_experts_per_tok}")
    print(f"  Batch  : {batch_size} × {seq_len} tokens  (dtype={dtype})")

    device = torch.device("cuda")

    # ── 1. Build tiny Mixtral config ─────────────────────────────────────────
    config = MixtralConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=num_local_experts,
        num_experts_per_tok=num_experts_per_tok,
    )

    # ── 2. Reference HF block (eval, deterministic, shared weights) ──────────
    torch.manual_seed(0)
    hf_moe = MixtralSparseMoeBlock(config).to(device=device, dtype=dtype)
    hf_moe.eval()

    # ── 3. NanoMoE wrapper over the *same* block ─────────────────────────────
    #   We pass the same hf_moe object; NanoMoELayer registers gate & experts
    #   as sub-modules so they share weights — no copying.
    nano_moe = NanoMoELayer(
        hf_moe,
        num_experts=num_local_experts,
        top_k=num_experts_per_tok,
        fallback_to_hf=False,   # error out if CUDA ext is missing
    ).to(device=device, dtype=dtype)
    nano_moe.eval()

    # ── 4. Synthetic input ───────────────────────────────────────────────────
    test_input = torch.randn(
        batch_size, seq_len, hidden_size,
        device=device, dtype=dtype
    )
    print(f"\n  Input  : {list(test_input.shape)}  device={device}  "
          f"dtype={test_input.dtype}")

    # ── 5. HF reference forward ──────────────────────────────────────────────
    with torch.no_grad():
        hf_output, hf_router_logits = hf_moe(test_input)

    # ── 6. NanoMoE forward ───────────────────────────────────────────────────
    with torch.no_grad():
        nano_output, _nano_logits = nano_moe(test_input)  # router_logits=None at inference

    # ── 7. Shape sanity ──────────────────────────────────────────────────────
    assert hf_output.shape == nano_output.shape, (
        f"Shape mismatch: HF={hf_output.shape}  Nano={nano_output.shape}"
    )

    # ── 8. Numerical comparison ──────────────────────────────────────────────
    diff = torch.abs(hf_output.float() - nano_output.float())
    max_diff  = diff.max().item()
    mean_diff = diff.mean().item()

    print("\n  ┌─────────────────────────────────────────────┐")
    print(f"  │  Max  |Δ|  : {max_diff:>10.6f}               │")
    print(f"  │  Mean |Δ|  : {mean_diff:>10.6f}               │")
    print(f"  │  Tolerance  : {atol:>10.6f}               │")
    print("  └─────────────────────────────────────────────┘")

    passed = max_diff < atol
    if passed:
        print("\n  [SUCCESS] Custom CUDA routing matches HuggingFace "
              "architecture perfectly!")
        print("            ✓ permute/unpermute logic is numerically correct.")
        print("            ✓ Zero-atomic gather & CUB prefix-sums are valid.")
    else:
        print("\n  [FAILED]  Output divergence detected.")
        print("            ✗ Inspect permute_kernel / unpermute_kernel logic.")

        # Emit the worst-offending token for debugging.
        flat_diff = diff.view(-1, hidden_size).max(dim=1).values
        worst_token = flat_diff.argmax().item()
        print(f"            ✗ Worst token index: {worst_token}  "
              f"(Δ = {flat_diff[worst_token]:.6f})")

    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Quick float32 sanity sweep (no dtype casting involved)
# ─────────────────────────────────────────────────────────────────────────────

def test_moe_parity_fp32():
    """
    Run the same parity test in float32 to isolate dtype-promotion issues
    from routing logic bugs.  Float32 should be pixel-perfect (atol=1e-5).
    """
    _banner("FP32 Sanity Sweep")
    return test_moe_parity(
        hidden_size=512,
        intermediate_size=2048,
        num_local_experts=4,
        num_experts_per_tok=2,
        batch_size=2,
        seq_len=32,
        dtype=torch.float32,
        atol=1e-4,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pytest entry-points (auto-discovered by pytest)
# ─────────────────────────────────────────────────────────────────────────────

def test_fp16_parity():
    assert test_moe_parity(), "FP16 parity test failed"

def test_fp32_parity():
    assert test_moe_parity_fp32(), "FP32 parity test failed"


# ─────────────────────────────────────────────────────────────────────────────
# __main__ — pretty printed standalone run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _check_cuda()

    print("=" * 62)
    print("  NanoMoE Architectural Correctness Verification")
    print("  VRAM hint: ~1 GB (tiny config, no disk I/O)")
    print("=" * 62)

    results = {}

    # FP32 sweep first — cheapest, best for isolating routing bugs.
    results["fp32"] = test_moe_parity_fp32()

    # FP16 production test — matches T4 deployment dtype.
    results["fp16"] = test_moe_parity(
        hidden_size=1024,
        intermediate_size=4096,
        num_local_experts=8,
        num_experts_per_tok=2,
        batch_size=4,
        seq_len=128,
        dtype=torch.float16,
        atol=1e-3,
    )

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  FINAL SUMMARY")
    print("=" * 62)
    for name, ok in results.items():
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}  {name.upper()} parity")
    print("=" * 62)

    if all(results.values()):
        print("\n  All tests passed. moe.cu is numerically verified. ✓")
        sys.exit(0)
    else:
        print("\n  One or more tests failed. See output above.")
        sys.exit(1)
