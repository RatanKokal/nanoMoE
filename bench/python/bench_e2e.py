"""
bench/python/bench_e2e.py
─────────────────────────────────────────────────────────────────────────────
Three-Way End-to-End Generation Throughput Benchmark

Compares:
  1. HF Baseline   — stock MixtralSparseMoeBlock (no custom CUDA)
  2. Legacy MoE    — custom_moe_legacy: atomic unpermute, serial prefix-sum,
                     matmul fused inside routing kernel
  3. NanoMoE       — custom_moe_cuda:   CUB ExclusiveSum, zero-atomic float4
                     unpermute, persistent buffer cache, cuBLAS mm_out gate

Usage (from repo root after `pip install -e .`):
    python bench/python/bench_e2e.py
    python bench/python/bench_e2e.py --warmup 5 --runs 10 --tokens 64
"""

import sys, os, time, argparse, importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, MixtralConfig, MixtralForCausalLM

# ── repo root on path ─────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from inference.engine import generate_text

# ─────────────────────────────────────────────────────────────────────────────
# NanoMoELayer shim — re-implements the layer swapping logic so we can
# explicitly choose which CUDA extension to load (legacy vs latest).
# ─────────────────────────────────────────────────────────────────────────────

class NanoMoELayerForBench(nn.Module):
    """
    Thin MoE layer that delegates routing to whichever `cuda_ext` module is
    passed in.  Shares weights with an existing MixtralSparseMoeBlock.
    """
    def __init__(self, hf_moe_block, num_experts: int, top_k: int, cuda_ext):
        super().__init__()
        self.gate       = hf_moe_block.gate        # MixtralTopKRouter
        self.experts    = hf_moe_block.experts      # MixtralExperts
        self.num_experts = num_experts
        self.top_k      = top_k
        self._ext       = cuda_ext
        self._act       = hf_moe_block.experts.act_fn
        self._expert_fns = self._build_fns()

    def _build_fns(self):
        gu = self.experts.gate_up_proj  # [E, 2*ffn, D]
        d  = self.experts.down_proj     # [E, D, ffn]
        act = self._act
        half = gu.shape[1] // 2
        fns = []
        for e in range(self.num_experts):
            fns.append(
                lambda x, _w=gu[e], _d=d[e], _h=half:
                F.linear(act(F.linear(x, _w[:_h])) * F.linear(x, _w[_h:]), _d)
            )
        return fns

    def forward(self, hidden_states):
        batch, seq_len, d_model = hidden_states.shape
        orig_dtype = hidden_states.dtype

        x_2d  = hidden_states.view(-1, d_model)
        N      = x_2d.shape[0]
        x_f32  = x_2d.float()
        W_g    = self.gate.weight.T.float()   # [D, E]

        permuted_x, reverse_map, topk_weights, offsets = \
            self._ext.route_and_permute(x_f32, W_g, self.top_k)

        total_routed = N * self.top_k
        expert_out   = torch.empty_like(permuted_x)

        for e_id in range(self.num_experts):
            start = offsets[e_id].item()
            end   = offsets[e_id + 1].item() if e_id + 1 < self.num_experts else total_routed
            if end - start == 0:
                continue
            chunk = permuted_x[start:end].to(orig_dtype)
            with torch.no_grad():
                expert_out[start:end] = self._expert_fns[e_id](chunk).float()

        final = self._ext.unpermute(expert_out, reverse_map, topk_weights, N)
        return final.to(orig_dtype).view(batch, seq_len, d_model)


def patch_with_ext(hf_model, cuda_ext, num_experts: int, top_k: int):
    """Walk the model graph and replace every MixtralSparseMoeBlock."""
    def _replace(module):
        for name, child in module.named_children():
            if child.__class__.__name__ == "MixtralSparseMoeBlock":
                setattr(module, name,
                        NanoMoELayerForBench(child, num_experts, top_k, cuda_ext))
            else:
                _replace(child)
    _replace(hf_model)
    return hf_model


# ─────────────────────────────────────────────────────────────────────────────
# Timing helper
# ─────────────────────────────────────────────────────────────────────────────

def measure_tps(model, tokenizer, prompt: str,
                warmup: int, runs: int, tokens_per_run: int) -> tuple[float, int]:
    """
    Returns (tokens_per_second, total_tokens_generated).
    Synchronizes CUDA before/after the timed region.
    """
    # ── Warmup (not measured) ─────────────────────────────────────────────────
    for _ in range(warmup):
        generate_text(prompt, model, None, tokenizer, max_tokens=tokens_per_run)

    # ── Timed region ──────────────────────────────────────────────────────────
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    total = 0
    for _ in range(runs):
        _, _, count = generate_text(
            prompt, model, None, tokenizer, max_tokens=tokens_per_run
        )
        total += count

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return total / elapsed if elapsed > 0 else 0.0, total


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NanoMoE three-way E2E benchmark")
    parser.add_argument("--warmup",  type=int, default=3,
                        help="Warmup passes before measuring (default: 3)")
    parser.add_argument("--runs",    type=int, default=5,
                        help="Measured generation passes (default: 5)")
    parser.add_argument("--tokens",  type=int, default=64,
                        help="max_tokens per generation pass (default: 64)")
    parser.add_argument("--hidden",  type=int, default=512)
    parser.add_argument("--ffn",     type=int, default=2048)
    parser.add_argument("--layers",  type=int, default=4)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k",   type=int, default=2)
    args = parser.parse_args()

    DEVICE = "cuda"
    DTYPE  = torch.float16

    print("=" * 62)
    print("  NanoMoE — Three-Way E2E Generation Benchmark")
    print("=" * 62)
    print(f"  hidden={args.hidden}  ffn={args.ffn}  layers={args.layers}  "
          f"E={args.experts}  k={args.top_k}")
    print(f"  warmup={args.warmup}  runs={args.runs}  tokens/run={args.tokens}")
    print("=" * 62)

    # ── Check extensions ──────────────────────────────────────────────────────
    try:
        import custom_moe_cuda   as ext_new
    except ImportError:
        print("[ERROR] custom_moe_cuda not found — run `pip install -e .` first.")
        sys.exit(1)

    try:
        import custom_moe_legacy as ext_legacy
    except ImportError:
        print("[ERROR] custom_moe_legacy not found — run `pip install -e .` first.")
        sys.exit(1)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.eos_token_id = tokenizer.eos_token_id or 50256

    # ── Tiny Mixtral config ───────────────────────────────────────────────────
    config = MixtralConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden,
        intermediate_size=args.ffn,
        num_hidden_layers=args.layers,
        num_attention_heads=8,
        num_key_value_heads=2,
        num_local_experts=args.experts,
        num_experts_per_tok=args.top_k,
    )

    PROMPT = "Explain the architectural differences between MoE routing strategies."

    results = {}

    # ── Phase 1: HF Baseline ──────────────────────────────────────────────────
    print("\n[1/3] Building HF baseline model (no custom CUDA)...")
    model_hf = MixtralForCausalLM(config).to(device=DEVICE, dtype=DTYPE)
    model_hf.eval()

    print(f"      Warmup x{args.warmup}...")
    tps, tok = measure_tps(model_hf, tokenizer, PROMPT,
                           args.warmup, args.runs, args.tokens)
    results["HF Baseline (PyTorch)"] = tps
    print(f"  → {tps:.1f} TPS  ({tok} tokens in {args.runs} runs)")

    # ── Phase 2: Legacy CUDA engine ───────────────────────────────────────────
    print("\n[2/3] Patching with Legacy CUDA engine (atomic unpermute, serial prefix-sum)...")
    # Re-create model to get fresh, unpatched weights
    model_legacy = MixtralForCausalLM(config).to(device=DEVICE, dtype=DTYPE)
    model_legacy.eval()
    model_legacy = patch_with_ext(model_legacy, ext_legacy, args.experts, args.top_k)

    print(f"      Warmup x{args.warmup}...")
    tps, tok = measure_tps(model_legacy, tokenizer, PROMPT,
                           args.warmup, args.runs, args.tokens)
    results["Legacy CUDA (atomic + serial prefix-sum)"] = tps
    print(f"  → {tps:.1f} TPS  ({tok} tokens in {args.runs} runs)")
    del model_legacy
    torch.cuda.empty_cache()

    # ── Phase 3: NanoMoE latest ───────────────────────────────────────────────
    print("\n[3/3] Patching with NanoMoE latest (CUB ExclusiveSum, zero-atomic float4)...")
    model_nano = MixtralForCausalLM(config).to(device=DEVICE, dtype=DTYPE)
    model_nano.eval()
    model_nano = patch_with_ext(model_nano, ext_new, args.experts, args.top_k)

    print(f"      Warmup x{args.warmup}...")
    tps, tok = measure_tps(model_nano, tokenizer, PROMPT,
                           args.warmup, args.runs, args.tokens)
    results["NanoMoE (CUB + zero-atomic float4)"] = tps
    print(f"  → {tps:.1f} TPS  ({tok} tokens in {args.runs} runs)")

    # ── Summary ───────────────────────────────────────────────────────────────
    baseline = results["HF Baseline (PyTorch)"]
    print("\n" + "=" * 62)
    print("  RESULTS (tokens/sec)      │  Speedup vs HF baseline")
    print("─" * 62)
    for label, tps in results.items():
        speedup = tps / baseline if baseline > 0 else float("nan")
        print(f"  {label:<38}│  {tps:>8.1f} TPS  ({speedup:.2f}x)")
    print("=" * 62)

    nano_tps    = results["NanoMoE (CUB + zero-atomic float4)"]
    legacy_tps  = results["Legacy CUDA (atomic + serial prefix-sum)"]
    if legacy_tps > 0:
        print(f"\n  NanoMoE vs Legacy improvement : "
              f"{(nano_tps - legacy_tps) / legacy_tps * 100:+.1f}%")
    print()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("[SKIP] No CUDA device — benchmark requires a GPU.")
        sys.exit(0)
    main()
