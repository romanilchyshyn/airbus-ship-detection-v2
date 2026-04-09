from typing import Tuple
import torch

def rle_to_mask(rle: str, h: int, w: int) -> torch.Tensor:
    if not rle:
        return torch.zeros((h, w), dtype=torch.bool)

    s = torch.tensor(list(map(int, rle.split())), dtype=torch.int64)

    starts = s[0::2] - 1 # start from zero
    lengths = s[1::2]
    ends = starts + lengths

    mask = torch.zeros(h * w, dtype=torch.bool)

    for start, end in zip(starts, ends):
        mask[start:end] = True

    return mask.view(w, h).t()

def mask_to_rle(mask: torch.Tensor) -> str:
    pass
