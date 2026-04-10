import torch
import torchvision.transforms.functional as F
from torchvision.utils import draw_segmentation_masks

def pil_image(image: torch.Tensor):
    pil = F.to_pil_image(image)
    return pil

def masked_pil_image(image: torch.Tensor, mask: torch.Tensor, alpha: float = 0.8):
    masked = draw_segmentation_masks(image, masks=mask, alpha=alpha)
    pil = pil_image(masked)
    return pil
