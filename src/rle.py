import torch

def rle_to_mask(rle: str, h: int, w: int) -> torch.Tensor:
    if not rle:
        return torch.zeros((h, w), dtype=torch.bool)

    s = torch.tensor(list(map(int, rle.split())), dtype=torch.int64)

    starts = s[0::2] - 1
    lengths = s[1::2]

    idx = torch.repeat_interleave(starts, lengths) + \
          torch.cat([torch.arange(l) for l in lengths])

    mask = torch.zeros(h * w, dtype=torch.bool)
    mask[idx] = True

    return mask.view(w, h).t()

def mask_to_rle(mask: torch.Tensor) -> str:
    pass
