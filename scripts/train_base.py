from pathlib import Path
import argparse
import time, wandb
from tqdm import trange

import numpy.typing as npt
import torch, numpy as np
from base_model.tiktoken_port import load_tokenizer
from base_model.layers import TransformerLM
from base_model.training_utils import *

# ----------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Base language model")
parser.add_argument("--run", type=str, default="initial", help="wandb run name")
parser.add_argument(
    "-d", "--data-dir", default="data/tokenized_tinystories",
    help="Directory containing the tokenized training (train.npy) and validation (valid.npy) texts"
)
parser.add_argument("-t", "--tokenizer", default="data/tinystories_merges.txt")
# model paramters
parser.add_argument("--context-length", type=int, default=512)
parser.add_argument("--num-layers", type=int, default=12)
parser.add_argument("--aspect-ratio", type=int, default=64, help="d_model = num_layers*aspect_ratio")
parser.add_argument("--num-heads", type=int, default=12)
# optimization paramters
parser.add_argument("--bs", type=int, default=2)
parser.add_argument("--grad-accum-steps", type=int, default=1)
parser.add_argument("--lr-max", type=float, default=3e-2, help="The scheduler warms-up to this lr")
parser.add_argument("--warmup-ratio", type=float, default=0.01, help="Ratio of iterations for lr warmup")
parser.add_argument("--wd", type=float, default=0.0)
parser.add_argument("--max-grad-norm", type=float, default=1.0)
parser.add_argument("--betas", type=float, nargs=2, default=[0.9,0.95], help="The betas parameter for AdamW")
# training horizon (only one is used, in this order)
parser.add_argument("--max-steps", type=int, default=-1, help="-1 = disable")
parser.add_argument("--target-flops", type=float, default=-1, help="-1 = disable")
parser.add_argument("--data-param-ratio", type=int, default=20, help="-1 = disable")
# eval parameters
parser.add_argument("--eval-interval", type=int, default=100)
parser.add_argument("--eval-tokens", type=int, default=524_288)
parser.add_argument("--save-interval", type=int, default=-1, help="-1 = saves only at the end")
args = parser.parse_args()

# ----------------------------------------------------------------------------------------------------
print("----------Config----------")

# device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"{device=}")
use_amp = device=="cuda"
autocast_ctx = torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp)

# data
data_path = Path(args.data_dir)
train_data = np.load(data_path/"train.npy", mmap_mode='r')
eval_data = np.load(data_path/"valid.npy", mmap_mode='r')

# tokenizer
tok_path = Path(args.tokenizer)
tokenizer = load_tokenizer(tok_path)
print("\n# Data")
vocab_size = tokenizer.n_vocab
bs, context_length = args.bs, args.context_length
grad_accum_steps = args.grad_accum_steps
print(f"{vocab_size=:,}, {context_length=}, {bs=}, {grad_accum_steps=}")
tok_per_step = bs*grad_accum_steps*context_length
print(f"Tokens per mini-batch: {tok_per_step:,}")

# model
print("\n# Model")
num_layers = args.num_layers
d_model = num_layers*args.aspect_ratio
d_ff = 4*d_model
num_heads = args.num_heads
assert d_model%num_heads==0,  "`d_model` must be divisible by `num_heads`"
print(f"{num_layers=}, {d_model=}, {d_ff=}, {num_heads=}, d_h={d_model//num_heads}")

orig_model = TransformerLM(
    vocab_size, context_length, num_layers, d_model, d_ff, num_heads, device=device
)
model = compile_model(orig_model, bs=bs, seq_len=context_length, device=device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params:,}")
num_flops_per_tok = orig_model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_tok:e}")

# training horizon
print("\n# Training Horizon")
if args.max_steps > 0:
    max_steps = args.max_steps
    print(f"Provided number of iterations: {max_steps:,}")
elif args.target_flops > 0:
    max_steps = round(args.target_flops/(num_flops_per_tok*tok_per_step))
    print(f"Calculated number of iterations from target FLOPs: {max_steps:,}")
elif args.data_param_ratio > 0:
    max_steps = (args.data_param_ratio*num_params)//tok_per_step
    print(f"Calculated number of iterations from the data param ratio: {max_steps:,}")
else:
    raise ValueError("No training horizon specified")
train_tokens = tok_per_step*max_steps
print(f"Total training tokens: {train_tokens:e}")
print(f"Total training FLOPs estimate: {(num_flops_per_tok*train_tokens):e}")
print(f"\nTokens to params ratio: {train_tokens/num_params:.2f}")

eval_tok_per_step = bs*context_length
eval_steps = max(1, args.eval_tokens//eval_tok_per_step)
ckpt_dir = Path(f"models/{args.run}")
ckpt_dir.mkdir(parents=True, exist_ok=True)

# optimizer
print("\n# Optimization")
lr_min = 1e-4
lr_max = args.lr_max
warmup_iters = int(max_steps*args.warmup_ratio)
wd = args.wd
max_grad_norm = args.max_grad_norm
print(f"{lr_min=:.4f}, {lr_max=:.4f}, {warmup_iters=:,}, {wd=:.4f}, {max_grad_norm=}")
optimizer = AdamW(model.parameters(), lr=lr_max, weight_decay=wd, betas=tuple(args.betas))

# wandb
config = {
    "num_layers":num_layers, "aspect_ratio":args.aspect_ratio,
    "num_heads":num_heads, "context_length":context_length,
    "bs":bs, "grad_accum_steps":grad_accum_steps, "wd":wd,
    "lr_max":lr_max, "warmup_ratio":args.warmup_ratio,
    "max_steps":max_steps, "train_tokens":train_tokens
}
run = wandb.init(project="cs336", name=args.run, config=config)

# ----------------------------------------------------------------------------------------------------
@torch.inference_mode()
def evaluate(data: npt.NDArray, model:torch.nn.Module, steps: int) -> float:
    model.eval()
    loss = 0
    for _ in trange(steps, desc="Evaluating", leave=False):
        ids, targets = get_batch(data, bs, context_length, device=device)
        logits = model(ids)
        loss += cross_entropy(logits, targets).item()
    model.train()
    return loss/steps

# ----------------------------------------------------------------------------------------------------
print("\n----------Training loop----------")
train_start = time.time()

for step in range(1, max_steps+1):
    t0 = time.time()
    lr = cosine_schedule_lr(step, lr_max, lr_min, warmup_iters, max_steps)
    for g in optimizer.param_groups:
        g["lr"] = lr
    
    # gradient accumulation
    train_loss = 0
    for micro_step in range(grad_accum_steps):
        ids, targets = get_batch(train_data, bs, context_length, device=device)
        with autocast_ctx:
            logits = model(ids)
            loss = cross_entropy(logits, targets)
        loss = loss/grad_accum_steps
        train_loss += loss.detach().item()
        loss.backward()
    
    grad_norm = clip_gradient(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # logging
    dt = time.time()-t0
    tok_per_sec = tok_per_step/dt
    print(f"step {step}/{max_steps} | {train_loss=:.4f} | {grad_norm=:.3f} | {lr=:.3f} | {dt=:.2f}s | {tok_per_sec=:.2f}")
    log = {
        "train/loss":train_loss, "train/grad_norm":grad_norm, "train/lr":lr, "train/dt":dt
    }

    if step%args.eval_interval == 0:
        eval_start = time.time()
        with autocast_ctx:
            eval_loss = evaluate(eval_data, model, eval_steps)
        eval_time = time.time()-eval_start
        print(f"eval_loss: {eval_loss:.3f} | dt: {eval_time:.2f}s")
        log["eval/loss"] = eval_loss
        log["eval/dt"] = eval_time

    wandb.log(log)
    
    if args.save_interval != -1 and step%args.save_interval == 0:
        save_checkpoint(model, optimizer, step, ckpt_dir/str(step))

save_checkpoint(model, optimizer, max_steps, ckpt_dir/str(max_steps))
tot_time = time.time()-train_start
print(f"\nTraining run complete! Total time taken: {tot_time/60:,} minutes")
run.finish()