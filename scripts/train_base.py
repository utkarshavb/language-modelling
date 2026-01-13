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
    "-d", "--data_dir", default="data/tokenized_tinystories",
    help="Directory containing the tokenized training (train.npy) and validation (valid.npy) texts"
)
parser.add_argument("-t", "--tokenizer", default="data/tinystories_merges.txt")
# model paramters
parser.add_argument("--context_length", type=int, default=512)
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--aspect_ratio", type=int, default=96, help="d_model = num_layers*aspect_ratio")
# optimization paramters
parser.add_argument("--bs", type=int, default=32)
parser.add_argument("--grad_accum_steps", type=int, default=1)
parser.add_argument("--lr_max", type=float, default=3e-2, help="The scheduler warms up to this lr")
parser.add_argument("--warmup_ratio", type=float, default=0.01, help="Ratio of iterations for lr warmup")
parser.add_argument("--wd", type=float, default=0.0)
parser.add_argument("--max_grad_norm", type=float, default=1.0)
parser.add_argument("--betas", type=float, nargs=2, default=[0.9,0.95], help="The betas parameter for AdamW")
# training horizon
parser.add_argument("--max_steps", type=int, default=-1, help="When > 0, takes precedence over `training_tokens`")
parser.add_argument("--training_tokens", type=int, default=327_680_000, help="bs*context_length*steps")
# eval parameters
parser.add_argument("--eval_interval", type=int, default=100)
parser.add_argument("--eval_tokens", type=int, default=524_288)
parser.add_argument("--save_interval", type=int, default=-1, help="-1: saves only at the end")
args = parser.parse_args()

# ---------------various initializations---------------
# tokenizer
tkn_path = Path(args.tokenizer)
tokenizer = load_tokenizer(tkn_path)
vocab_size = tokenizer.n_vocab
print(f"{vocab_size=:,}")

# data
data_path = Path(args.data_dir)
train_data = np.load(data_path/"train.npy", mmap_mode='r')
eval_data = np.load(data_path/"valid.npy", mmap_mode='r')

# device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = 'cpu'
print(f"Using device: {device}")

# training related hyperparamters
bs = args.bs
context_length = args.context_length
grad_accum_steps = args.grad_accum_steps
tok_per_step = bs*grad_accum_steps*context_length
print(f"Tokens per mini-batch: {tok_per_step:,}")
if args.max_steps > 0:
    max_steps = args.max_steps
else:
    max_steps = args.training_tokens//tok_per_step
training_tokens = tok_per_step*max_steps
print(f"Total training tokens: {training_tokens:,}")
eval_steps = args.eval_tokens//tok_per_step
ckpt_dir = Path(f"models/{args.run}")
ckpt_dir.mkdir(parents=True, exist_ok=True)

# model
num_layers = args.num_layers
d_model = num_layers*args.aspect_ratio
d_ff = 8*d_model//3
num_heads = (d_model+127)//128  # ceil div
model = TransformerLM(
    vocab_size, context_length, num_layers, d_model, d_ff, num_heads, device=device
)
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params:,}")
print(f"Tokens : params ratio: {training_tokens/num_params:.2f}")

# optimizer
lr_min = 1e-4
lr_max = args.lr_max
warmup_iters = int(max_steps*args.warmup_ratio)
optimizer = AdamW(model.parameters(), lr=lr_max, weight_decay=args.wd, betas=tuple(args.betas))

# wandb
config = {
    "num_layers":num_layers, "context_length":context_length, "bs":bs,
    "grad_accum_steps":grad_accum_steps, "lr_max":lr_max, "wd":args.wd,
    "warmup_ratio":args.warmup_ratio, "max_steps":max_steps,
    "training_tokens":training_tokens, "eval_tokens":args.eval_tokens
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
print("Starting training loop")
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