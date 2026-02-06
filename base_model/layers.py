import torch
from torch import nn, Tensor
from einops import einsum, reduce, rearrange
from jaxtyping import Bool, Float, Int

class Linear(nn.Module):
    def __init__(self, ni, nf, dtype: torch.dtype|None=None, device: torch.device|None=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(nf, ni, dtype=dtype, device=device))
        std = (2/(ni+nf))**0.5
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, x: Float[Tensor, "... ni"]) -> Float[Tensor, "... nf"]:
        return einsum(x, self.weight, "... ni, nf ni -> ... nf")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dtype=None, device=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(
            num_embeddings, embedding_dim, dtype=dtype, device=device
        ))
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, ids: Int[Tensor, "... L"]) -> Int[Tensor, "... L embedding_dim"]:
        return self.weight[ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, dtype=None, device=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = reduce(x*x, "... d_model -> ... 1", "mean")
        rms = (rms + self.eps).sqrt()
        x = x/rms*self.weight
        return x.to(in_dtype)

def silu(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    return torch.sigmoid(x) * x

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, dtype=None, device=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, dtype=dtype, device=device)
        self.w2 = Linear(d_ff, d_model, dtype=dtype, device=device)
        self.w3 = Linear(d_model, d_ff, dtype=dtype, device=device)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        return self.w2(silu(self.w1(x)) * self.w3(x))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_k: int, max_seq_len, theta: float=1e5, device=None):
        assert d_k%2 == 0, "`d_k` should be even"
        super().__init__()
        ps = torch.arange(max_seq_len, device=device)
        inv_freq = 1 / theta**(torch.arange(0, d_k, 2, device=device)/d_k)
        angles = einsum(ps, inv_freq, "i, j -> i j")[..., None]
        self.register_buffer('cos_cached', angles.cos(), persistent=False)
        self.register_buffer('sin_cached', angles.sin(), persistent=False)
    
    def forward(
        self, x: Float[Tensor, "... L d_k"], token_positions: Int[Tensor, "... L"]
    ) -> Float[Tensor, "... L d_k"]:
        cos = self.cos_cached[token_positions] # pyright: ignore[reportIndexIssue]
        sin = self.sin_cached[token_positions] # pyright: ignore[reportIndexIssue]
        x = rearrange(x, "... (d_k2 t) -> ... d_k2 t", t=2)
        x1, x2 = x.unbind(dim=-1)
        x_rot = rearrange([-x2, x1], "t ... d_k2 -> ... d_k2 t")
        return rearrange(x*cos + x_rot*sin, "... d_k2 t -> ... (d_k2 t)")

def softmax(tensor: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    tensor = tensor-tensor.amax(dim=dim, keepdim=True)
    tensor = tensor.exp()
    return tensor/tensor.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(
    Q: Float[Tensor, "bs ... queries d_k"], K: Float[Tensor, "bs ... keys d_k"],
    V: Float[Tensor, "bs ... values d_v"], mask: Bool[Tensor, "queries keys"] | None = None
) -> Float[Tensor, "... queries d_v"]:
    d_k = Q.size(-1)
    attn_weights = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    attn_weights = attn_weights/d_k**0.5
    if mask is not None:
        attn_weights = attn_weights.masked_fill(~mask, float("-inf"))
    attn_scores = softmax(attn_weights, dim=-1)
    out_proj = einsum(attn_scores, V, "... queries keys, ... keys d_v -> ... queries d_v")
    return out_proj

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len=None, theta=1e5, dtype=None, device=None):
        super().__init__()
        assert d_model % num_heads == 0, "`d_model` must be divisible by `num_heads`"
        self.h = num_heads
        self.W = Linear(d_model, 3*d_model, dtype=dtype, device=device)
        self.rope = None
        if max_seq_len:
            self.rope = RotaryPositionalEmbedding(d_model//self.h, max_seq_len, theta=theta, device=device)
        self.output_proj = Linear(d_model, d_model, dtype=dtype, device=device)

    def forward(self, x: Float[Tensor, "... L d_model"]) -> Float[Tensor, "... L d_model"]:
        x = self.W(x)   # (..., L, 3*d_model)
        x = rearrange(x, "... L (t h d) -> ... h L t d", h=self.h, t=3)
        Q, K, V = x.unbind(dim=-2)
        L = x.size(-3)
        if self.rope:
            positions = torch.arange(L, device=x.device).view(*([1]*(Q.ndim-2)+[L]))
            Q, K = (self.rope(t, positions) for t in (Q, K))
        mask = torch.ones((L, L), dtype=torch.bool, device=x.device)
        attn_out = scaled_dot_product_attention(Q, K, V, mask=torch.tril(mask))   # (..., h, L, d)
        attn_out = rearrange(attn_out, "... h L d -> ... L (h d)")
        return self.output_proj(attn_out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, max_seq_len=None, theta=1e5, dtype=None, device=None):
        super().__init__()
        self.attn = MultiHeadSelfAttention(
            d_model, num_heads, max_seq_len, theta=theta, dtype=dtype, device=device
        )
        self.ln1 = RMSNorm(d_model, dtype=dtype, device=device)
        self.ffn = SwiGLU(d_model, d_ff, dtype=dtype, device=device)
        self.ln2 = RMSNorm(d_model, dtype=dtype, device=device)

    def forward(self, x: Float[Tensor, "... L d_model"]) -> Float[Tensor, "... L d_model"]:
        x = x+self.attn(self.ln1(x))
        x = x+self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(
        self, vocab_size: int, context_length: int, num_layers: int, d_model: int,
        d_ff: int, num_heads: int, rope_theta: int=100_000, dtype=None, device=None
    ):
        self.config: dict[str, int] = dict(
            vocab_size=vocab_size, context_length=context_length, d_model=d_model,
            d_ff=d_ff, num_layers=num_layers, num_heads=num_heads, rope_theta=rope_theta 
        )
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, dtype=dtype, device=device)
        self.layers = nn.ModuleList([TransformerBlock(
            d_model, d_ff, num_heads, context_length, theta=rope_theta, dtype=dtype, device=device
        ) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, dtype=dtype, device=device)
        self.lm_head = Linear(d_model, vocab_size, dtype=dtype, device=device)

    def forward(self, ids: Int[Tensor, "... L"]) -> Float[Tensor, "... L vocab_size"]:
        x = self.token_embeddings(ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_final(x))
    
    def estimate_flops(self) -> int:
        """
        Returns the estimated flops per token for the model (forward + backward)
        Dense matmul FLOPs:
            MHSA: 6*4*d_model^2 (QKV projection + Output projection) per layer
            SwiGLU: 6*3*d_model*d_ff (W1, W2, W3) per layer
            lm head: 6*d_model*vocab_size
        Attention FLOPs:
            QK^T is (L, d_k) @ (d_k, L) and Attn@V is (L, L) @ (L, d_k)
            This costs 12*L*d_model per layer
        """
        d_model, vocab_size = self.config['d_model'], self.config['vocab_size']
        num_layers, d_ff = self.config['num_layers'], self.config['d_ff']
        seq_len = self.config['context_length']

        ffn = 6 * num_layers * (3*d_model*d_ff)
        mhsa_dense = 6 * num_layers * (4*d_model**2)
        mhsa_attn = 12 * seq_len * d_model * num_layers
        lm_head = 6*d_model*vocab_size
        
        return (ffn + mhsa_dense + mhsa_attn + lm_head)
    
    @torch.inference_mode()
    def generate(
            self, ids: Int[Tensor, "bs L"], max_new_tokens: int, p: float=1.0,
            temperature: float=1.0, eos_tok_id: int|None=None
        ) -> Int[Tensor, "bs max_new_tokens"]:
        assert 0<p<=1.0
        orig_seq_len = ids.size(-1)
        context_len = self.config["context_length"]

        # track which sequences are still generating
        alive = torch.ones(ids.shape[:-1], dtype=torch.bool, device=ids.device)

        for _ in range(max_new_tokens):
            if eos_tok_id is not None and not alive.any():
                break

            # crop the ids to fit the context length of the model
            inp = ids[..., -context_len:] if ids.size(-1)>context_len else ids
            logits: Float[Tensor, "... vocab_size"] = self(inp)[..., -1, :]
            logits /= temperature
            probs = softmax(logits, dim=-1)

            # select minimum probs such that sum(probs)>=p
            sorted_probs, sorted_idxs = probs.sort(descending=True)
            mask: Bool[Tensor, "bs context_len"] = sorted_probs.cumsum(dim=-1) <= p
            mask[..., 0] = True    # keep atleast one token
            sorted_probs *= mask
            choices: Int[Tensor, "bs 1"] = torch.multinomial(sorted_probs, 1)
            next_tok_ids: Int[Tensor, "bs 1"] = sorted_idxs.gather(dim=-1, index=choices)

            if eos_tok_id is not None:
                eos = torch.full_like(next_tok_ids, eos_tok_id)
                next_tok_ids = torch.where(alive[...,None], next_tok_ids, eos)
                # update alive
                alive = alive & (next_tok_ids.squeeze()!=eos_tok_id)

            ids = torch.cat((ids, next_tok_ids), dim=-1)
        return ids[..., orig_seq_len:]