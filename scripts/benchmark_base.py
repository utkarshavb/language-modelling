import argparse, timeit, os
import torch
from einops import einsum
from jaxtyping import Float, Bool
import base_model.layers as layers
from base_model.training_utils import *
import torch.cuda.nvtx as nvtx

parser = argparse.ArgumentParser()
parser.add_argument("--context-length", type=int, default=512)
parser.add_argument("--num-layers", type=int, default=12)
parser.add_argument("--hidden-dim", type=int, default=768)
parser.add_argument("--num-heads", type=int, default=12)
parser.add_argument("--warmup-steps", type=int, default=5, help="steps to run before measuring time")
parser.add_argument("--measuring-steps", type=int, default=10, help="steps to measure time on")
parser.add_argument("--run-backward", type=bool, default=True)
parser.add_argument("--use-mixed-precision", type=bool, default=True)
parser.add_argument("--record-memory", type=bool, default=False)
args = parser.parse_args()

# ----------------------------------------------------------------------------------------------------
# device init
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"{device=}\n")
synchronize = torch.cuda.synchronize if device=="cuda" else lambda: None
use_amp = args.use_mixed_precision
autocast_ctx = torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp)
record_memory = args.record_memory and device=="cuda"

# data init
vocab_size = 10_000
bs = 2
context_length = args.context_length
dummy_ids = torch.randint(vocab_size, size=(bs, context_length), dtype=torch.long, device=device)
dummy_targs = torch.randint(vocab_size, size=(bs, context_length), dtype=torch.long, device=device)

# model init
num_layers, d_model = args.num_layers, args.hidden_dim
d_ff = 4*d_model
num_heads = args.num_heads
model = layers.TransformerLM(
    vocab_size, context_length, num_layers, d_model, d_ff, num_heads=num_heads, device=device
)

@nvtx.range("scaled dot product attention")
def sdpa(
    Q: Float[Tensor, "bs ... queries d_k"], K: Float[Tensor, "bs ... keys d_k"],
    V: Float[Tensor, "bs ... values d_v"], mask: Bool[Tensor, "queries keys"] | None = None
) -> Float[Tensor, "... queries d_v"]:
    d_k = Q.size(-1)
    with nvtx.range("computing attention scores"):
        attn_weights = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
        attn_weights = attn_weights/d_k**0.5
    if mask is not None:
        attn_weights = attn_weights.masked_fill(~mask, float("-inf"))
    with nvtx.range("computing softmax"):
        attn_scores = layers.softmax(attn_weights, dim=-1)
    with nvtx.range("output projection"):
        out_proj = einsum(attn_scores, V, "... queries keys, ... keys d_v -> ... queries d_v")
    return out_proj

layers.scaled_dot_product_attention = sdpa

# optimization inits
lr = 1e-4
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.0)

# ----------------------------------------------------------------------------------------------------
def training_step(do_backward: bool=True):
    step_start = timeit.default_timer()

    for g in optimizer.param_groups:
        g["lr"] = lr
    logits = model(dummy_ids)
    loss = cross_entropy(logits, targets=dummy_ids)
    fwd_time = timeit.default_timer() - step_start

    bwd_start = timeit.default_timer()
    if do_backward:
        loss.backward()
        grad_norm = clip_gradient(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    synchronize()
    bwd_time = timeit.default_timer() - bwd_start

    step_time = timeit.default_timer() - step_start
    return step_time, fwd_time, bwd_time

# warmup
print("Warmup")
for it in range(args.warmup_steps):
    step_time, fwd_time, bwd_time = training_step(args.run_backward)
    print(f"{it=}/{args.warmup_steps} | {step_time=:.3f}s | {fwd_time=:.3f}s | {bwd_time=:.3f}s")

if record_memory:
    torch.cuda.memory._record_memory_history(max_entries=1000000)

step_times = torch.zeros(args.measuring_steps, device=device)
fwd_times = torch.zeros(args.measuring_steps, device=device)
bwd_times = torch.zeros(args.measuring_steps, device=device)

print("\nMeasuring steps")
for it in range(args.measuring_steps):
    step_time, fwd_time, bwd_time = training_step(args.run_backward)
    print(f"{it=}/{args.measuring_steps} | {step_time=:.3f}s | {fwd_time=:.3f}s | {bwd_time=:.3f}s")
    step_times[it] = step_time
    fwd_times[it] = fwd_time
    bwd_times[it] = bwd_time

if record_memory:
    memory_path = os.path.join("memory_snapshots", f"d{num_layers}_c{context_length}.pickle")
    torch.cuda.memory._dump_snapshot(memory_path)
    torch.cuda.memory._record_memory_history(enabled=None)

print(f"\nStatistics for measuring steps")
print(f"{step_times.mean()=:.3f}s | {step_times.std()=:.3f}s")
print(f"{fwd_times.mean()=:.3f}s | {fwd_times.std()=:.3f}s")
print(f"{bwd_times.mean()=:.3f}s | {bwd_times.std()=:.3f}s")