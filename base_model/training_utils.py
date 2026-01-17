import os
import numpy as np, torch
from typing import Callable, cast
from collections.abc import Iterable
import numpy.typing as npt
from torch import Tensor
from jaxtyping import Float, Int
from einops import reduce

def cross_entropy(logits: Float[Tensor, "... L vocab_size"], targets: Int[Tensor, "... L"]):
    logits = logits-reduce(logits, "... vocab_size -> ... 1", "max")
    base = reduce(logits.exp(), "... vocab_size -> ... 1", "sum")
    ll = logits.gather(dim=-1, index=targets[..., None])
    ll = ll - base.log()
    return -reduce(ll, "... 1 -> 1", "mean")

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas=(0.9, 0.95), eps=1e-8):
        defaults = dict(lr=lr, wd=weight_decay, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[],float]|None=None) -> float|None:
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for g in self.param_groups:
            beta1, beta2 = g['betas']
            for p in g['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if not state:
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                    state['t'] = 1
                m, v, t = state['m'], state['v'], state['t']
                state['t'] += 1
                
                # update
                m.mul_(beta1).add_(grad, alpha=(1-beta1))
                v.mul_(beta2).addcmul_(grad, grad, value=(1-beta2))
                lr_ = g['lr'] * ((1-beta2**t)**0.5 / (1-beta1**t))
                p.mul_(1-g['lr']*g['wd'])
                p.addcdiv_(m, v.sqrt().add(g['eps']), value=-lr_)
                # p.mul_(1-g['lr']*g['wd'])
        
        return loss

def cosine_schedule_lr(it, lr_max, lr_min, warmup_iters, cosine_iters):
    if it < warmup_iters:
        return it/warmup_iters * lr_max
    elif it <= cosine_iters:
        angle = (it-warmup_iters)/(cosine_iters-warmup_iters) * np.pi
        return lr_min + 0.5 * (1+np.cos(angle)) * (lr_max-lr_min)
    else:
        return lr_min

def clip_gradient(params: Iterable[Tensor], max_norm, eps=1e-6):
    norms = [p.grad.norm() for p in params if p.grad is not None]
    if not norms:
        return
    total_norm = torch.stack(norms).norm()
    if total_norm < max_norm:
        return total_norm
    for p in params:
        if p.grad is not None:
            p.grad.mul_(max_norm/(total_norm+eps))
    return max_norm

def get_batch(
        data: npt.NDArray, bs, context_length, dtype=None, device=None
) -> tuple[Int[Tensor, "bs context_length"], Int[Tensor, "bs context_length"]]:
    start_idxs = np.random.randint(data.shape[0]-context_length, size=bs)
    idxs = start_idxs[:, None] + np.arange(context_length)
    x, y = torch.from_numpy(data[idxs]), torch.from_numpy(data[idxs+1])
    if device and 'cuda' in device:
        x = x.pin_memory().to(dtype=dtype, device=device, non_blocking=True)
        y = y.pin_memory().to(dtype=dtype, device=device, non_blocking=True)
    else:
        x = x.to(dtype=dtype, device=device)
        y = y.to(dtype=dtype, device=device)
    return x, y

def save_checkpoint(
        model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str|os.PathLike
):
    obj = dict(
        model=model.state_dict(), optimizer=optimizer.state_dict(), iteration=iteration
    )
    torch.save(obj, out)

def load_checkpoint(src, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    obj = torch.load(src)
    model.load_state_dict(obj['model'])
    optimizer.load_state_dict(obj['optimizer'])
    return obj['iteration']

def compile_model(model: torch.nn.Module, device) -> torch.nn.Module:
    dummy_ids = torch.zeros((1,2), dtype=torch.long, device=device)
    try:
        compiled = torch.compile(model, dynamic=False)
        _ = compiled(dummy_ids)
        print("Compiled the model with inductor backend")
    except Exception as e:
        compiled = torch.compile(model, dynamic=False, backend="aot_eager")
        print("torch.compile failed, using 'aot_eager' backend")
        print(f"Reason: {type(e).__name__}: {e}")
    compiled = cast(torch.nn.Module, compiled)
    return compiled

@torch.inference_mode()
def generate(
    model: torch.nn.Module, ids: Int[Tensor, "... L"], endoftext_id: int,
    max_tokens: int|None=None, p: float=1.0, temperature: float=1.0, 
):
    pass