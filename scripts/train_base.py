from pathlib import Path
import argparse
import time, wandb
from tqdm import trange

import numpy.typing as npt
import torch, numpy as np
from base_model.tiktoken_port import load_tokenizer
from base_model.layers import TransformerLM
from base_model.training_utils import *

# ---------------command-line arguments---------------
parser = argparse.ArgumentParser(description="Base language model")
parser.add_argument("--run", type=str, default="initial", help="wandb run name")
parser.add_argument(
    "-d", "--data-dir", default="data/tokenized_tinystories",
    help="Directory containing the tokenized training (train.npy) and validation (valid.npy) texts"
)
parser.add_argument("-t", "--tokenizer", default="data/tinystories_merges.txt")
# model paramters
parser.add_argument("--context-length", type=int, default=512)
parser.add_argument("--num-layers", type=int, default=4)
parser.add_argument("--aspect-ratio", type=int, default=96, help="d_model = num_layers*aspect_ratio")
parser.add_argument("--num-heads", type=int, default=4)
# optimization paramters
parser.add_argument("--bs", type=int, default=32)
parser.add_argument("--grad-accum-steps", type=int, default=1)
parser.add_argument("--lr-max", type=float, default=3e-2, help="The scheduler warms up to this lr")
parser.add_argument("--warmup-ratio", type=float, default=0.01, help="Ratio of iterations for lr warmup")
parser.add_argument("--wd", type=float, default=0.0)
parser.add_argument("--max-grad-norm", type=float, default=1.0)
parser.add_argument("--betas", type=float, nargs=2, default=[0.9,0.95], help="The betas parameter for AdamW")
# training horizon (only one is used, in this order)
parser.add_argument("--max-steps", type=int, default=-1, help="-1 = disable")
parser.add_argument("--target-flops", type=int, default=-1, help="-1 = disable")
parser.add_argument("--train-tokens", type=int, default=327_680_000, help="bs*context_length*steps")
# eval parameters
parser.add_argument("--eval-interval", type=int, default=100)
parser.add_argument("--eval-tokens", type=int, default=524_288)
parser.add_argument("--save-interval", type=int, default=-1, help="-1 = saves only at the end")
args = parser.parse_args()

# ---------------various initializations---------------
print("----------Config----------")

# device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = 'cpu'
print(f"{device=}")

# tokenizer
tok_path = Path(args.tokenizer)
tokenizer = load_tokenizer(tok_path)
vocab_size = tokenizer.n_vocab
bs, context_length = args.bs, args.context_length

# data
data_path = Path(args.data_dir)
train_data = np.load(data_path/"train.npy", mmap_mode='r')
eval_data = np.load(data_path/"valid.npy", mmap_mode='r')

# training horizon
grad_accum_steps = args.grad_accum_steps
tok_per_step = bs*grad_accum_steps*context_length
if args.max_steps > 0:
    max_steps = args.max_steps
else:
    max_steps = args.train_tokens//tok_per_step
train_tokens = tok_per_step*max_steps

eval_tok_per_step = bs*context_length
eval_steps = max(1, args.eval_tokens//eval_tok_per_step)
ckpt_dir = Path(f"models/{args.run}")
ckpt_dir.mkdir(parents=True, exist_ok=True)

print("\n# Data")
print(f"{vocab_size=:,}, {context_length=}, {bs=}, {grad_accum_steps=}")
print(f"Tokens per mini-batch: {tok_per_step:,}")
print(f"Total training tokens: {train_tokens:,}")

# model
print("\n# Model")
num_layers = args.num_layers
d_model = num_layers*args.aspect_ratio
d_ff = 8*d_model//3
num_heads = args.num_heads
assert d_model%num_heads==0,  "`d_model` must be divisible by `num_heads`"
print(f"{num_layers=}, {d_model=}, {d_ff=}, {num_heads=}, d_h={d_model//num_heads}")

orig_model = TransformerLM(
    vocab_size, context_length, num_layers, d_model, d_ff, num_heads, device=device
)
model = compile_model(orig_model, device=device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params:,}")

print(f"\nTokens to params ratio: {train_tokens/num_params:.2f}")

# optimizer
lr_min = 1e-4
lr_max = args.lr_max
warmup_iters = int(max_steps*args.warmup_ratio)
optimizer = AdamW(model.parameters(), lr=lr_max, weight_decay=args.wd, betas=tuple(args.betas))

# wandb
config = {
    "num_layers":num_layers, "aspect_ratio":args.aspect_ratio,
    "num_heads":num_heads, "context_length":context_length,
    "bs":bs, "grad_accum_steps":grad_accum_steps, "wd":args.wd,
    "lr_max":lr_max, "warmup_ratio":args.warmup_ratio,
    "max_steps":max_steps, "train_tokens":train_tokens
}
run = wandb.init(project="cs336", name=args.run, config=config)

# ---------------evaluation loop---------------
@torch.inference_mode()
def evaluate(data: npt.NDArray, model:torch.nn.Module, steps: int) -> float:
    model.eval()
    loss = 0
    for _ in trange(steps, desc="Evaluating", leave=False):
        ids, targets = get_batch(
            data, bs, context_length, dtype=torch.long, device=device
        )
        logits = model(ids)
        loss += cross_entropy(logits, targets).item()
    model.train()
    return loss/steps

# ---------------training loop---------------
print("\n----------Starting training loop----------")
train_start = time.time()

for step in range(1, max_steps+1):
    t0 = time.time()
    lr = cosine_schedule_lr(step, lr_max, lr_min, warmup_iters, max_steps)
    for g in optimizer.param_groups:
        g["lr"] = lr
    
    # gradient accumulation
    train_loss = 0
    for micro_step in range(grad_accum_steps):
        ids, targets = get_batch(
            train_data, bs, context_length, dtype=torch.long, device=device
        )
        logits = model(ids)
        loss = cross_entropy(logits, targets)
        loss = loss/grad_accum_steps
        train_loss += loss.detach().item()
        loss.backward()
    
    grad_norm = clip_gradient(model.parameters(), args.max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()

    # logging
    dt = time.time()-t0
    tok_per_sec = tok_per_step/dt
    print(f"step {step}/{max_steps} | loss: {train_loss:.4f} | grad_norm: {grad_norm:.3f} | lr: {lr:.3f} | dt: {dt:.2f}s | tok/sec: {tok_per_sec:.2f}")
    log = {
        "train/loss":train_loss, "train/grad_norm":grad_norm, "train/lr":lr, "train/dt":dt
    }

    if step%args.eval_interval == 0:
        eval_start = time.time()
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
print(f"Training run complete! Total time taken: {tot_time/60:,} minutes")
run.finish()