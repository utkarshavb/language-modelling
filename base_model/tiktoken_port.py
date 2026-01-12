import os, tiktoken

PAT = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

def bytes_to_unicode() -> dict[int, str]:
    """Returns a dict b/w every possible byte(int) to a printable unicode char"""
    # These 188 ints can be used as-is, as they are not space or control chars
    bs = (
        list(range(ord("!"), ord("~") + 1)) +
        list(range(ord("¡"), ord("¬") + 1)) +
        list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs.copy()
    n = 0
    for b in range(2**8):
        if b in bs: continue
        bs.append(b); cs.append(2**8 + n)
        n += 1
    characters = [chr(n) for n in cs]
    return dict(zip(bs, characters))

def get_mergeable_ranks(path: str|os.PathLike) -> dict[bytes, int]:
    mergeable_ranks = {bytes([i]):i for i in range(256)}
    byte_dec = {v:k for k,v in bytes_to_unicode().items()}
    rank = 256
    with open(path, 'r') as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            tok1, tok2 = line.split()
            b = bytes(byte_dec[c] for c in tok1)+bytes(byte_dec[c] for c in tok2)
            mergeable_ranks[b] = rank
            rank += 1
    return mergeable_ranks

def load_tokenizer(path: str|os.PathLike):
    mergeable_ranks = get_mergeable_ranks(path)
    special_tokens = {"<|endoftext|>": len(mergeable_ranks)}
    tokenizer = tiktoken.Encoding(
        name="", pat_str=PAT, mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )
    return tokenizer