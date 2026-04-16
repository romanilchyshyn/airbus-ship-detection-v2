import torch
from scipy.ndimage import label

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
    # transpose + flatten mirrors the .view(w,h).t() in rle_to_mask
    flat = mask.t().reshape(-1)
    padded = torch.cat([torch.tensor([False]), flat, torch.tensor([False])])
    changes = torch.where(padded[1:] != padded[:-1])[0] + 1  # 1-based
    starts  = changes[0::2]
    lengths = changes[1::2] - starts
    
    return ' '.join(map(str, torch.stack([starts, lengths], dim=1).reshape(-1).tolist()))

def mask_to_rle_list(mask: torch.Tensor) -> list[str]:
    labeled, n = label(mask.numpy())
    labeled = torch.from_numpy(labeled)

    return [mask_to_rle(labeled == i) for i in range(1, n + 1)]
