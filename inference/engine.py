import time
import torch
from .nano_moe_layer import NanoMoELayer

def patch_model(hf_model, num_experts, top_k):
    patch_count = 0
    def _replace(model):
        nonlocal patch_count
        for child_name, child in model.named_children():
            if child.__class__.__name__ == "MixtralSparseMoeBlock":
                setattr(model, child_name, NanoMoELayer(child, num_experts, top_k))
                patch_count += 1
            else:
                _replace(child)
    _replace(hf_model)
    return hf_model

def generate_text(prompt, model, allocator, tokenizer, max_tokens=128):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    generated_tokens = []

    # Flush any pending CUDA work (e.g. weight-copy from .to(device)) so the
    # timer reflects true generation latency, not initialization overhead.
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for step in range(max_tokens):
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        generated_tokens.append(next_token.item())  # .item() blocks until GPU is done

        if next_token.item() == tokenizer.eos_token_id:
            break

        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

    # Block until all GPU work is complete before sampling the clock.
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    return tokenizer.decode(generated_tokens), len(generated_tokens) / elapsed if elapsed > 0 else 0.0, len(generated_tokens)
