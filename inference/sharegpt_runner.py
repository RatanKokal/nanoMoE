import sys, os
import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer

# Make sure the repo root is on sys.path so `inference` is importable
# regardless of the CWD from which this script is invoked.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from inference.engine import patch_model, generate_text

class DummyMixtralSparseMoeBlock(nn.Module):
    def __init__(self, d_model, num_experts):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model * 4), nn.SiLU(), nn.Linear(d_model * 4, d_model))
            for _ in range(num_experts)
        ])
    def forward(self, x):
        return x, None

class DummyModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_experts):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.moe_block = DummyMixtralSparseMoeBlock(d_model, num_experts)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, input_ids):
        return self.lm_head(self.moe_block(self.embed(input_ids))[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("[Init] Constructing synthetic model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.eos_token_id = tokenizer.eos_token_id or 50256
        
    model = DummyModel(vocab_size=tokenizer.vocab_size, d_model=256, num_experts=8).cuda()
    model = patch_model(model, num_experts=8, top_k=2)

    prompts = [f"Explain CUDA memory optimization {i}" for i in range(args.n)]
    
    for p in prompts:
        print(f"\nUSER: {p}")
        resp, tps, count = generate_text(p, model, None, tokenizer, max_tokens=16)
        print(f"NANOMOE: {resp}\n-> Speed: {tps:.2f} tokens/sec ({count} tokens)")
