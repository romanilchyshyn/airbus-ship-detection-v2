import torch

def rle_to_mask(rle: str, h: int, w: int) -> torch.Tensor:
    if not rle or not rle.strip():
        return torch.zeros((h, w), dtype=torch.bool)

    s = torch.tensor(list(map(int, rle.split())), dtype=torch.int64)
    
    if s.numel() % 2 != 0:
        raise ValueError(f"Malformed RLE: odd number of tokens ({s.numel()})")

    starts  = s[0::2] - 1 # 1-based -> 0-based
    lengths = s[1::2]

    total = int(lengths.sum())
    offsets = torch.ones(total, dtype=torch.int64)
    offsets[0] = 0
    offsets[torch.cumsum(lengths, dim=0)[:-1]] = 1 - lengths[:-1]
    offsets = torch.cumsum(offsets, dim=0)

    idx = torch.repeat_interleave(starts, lengths) + offsets

    mask = torch.zeros(h * w, dtype=torch.bool)
    mask[idx] = True

    return mask.view(w, h).t() # column-major -> (h, w)

def rle_list_to_mask(rles: list[str], h: int, w: int) -> torch.Tensor:
    mask = torch.zeros((h, w), dtype=torch.bool)
    
    for rle in rles:
        mask |= rle_to_mask(rle, h, w)
    
    return mask

def mask_to_rle(mask: torch.Tensor) -> str:
    pass