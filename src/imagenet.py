import torch

from device import get_device

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def normalize_batch(
    images: torch.Tensor,
    masks:  torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = get_device()

    images = normalize_images(images)
    masks  = masks.long().to(device)

    return images, masks

def normalize_images(
    images: torch.Tensor, 
) -> torch.Tensor:
    device = get_device()
    images = images.to(device=device, dtype=torch.float32) / 255.0

    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)

    images = (images - mean) / std

    return images

def denormalize_images(
    images: torch.Tensor,
) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(images.device)
    std = IMAGENET_STD.to(images.device)

    return (images * std + mean).clamp(0, 1)
