import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from imagenet import (normalize_batch, denormalize_images)

def build_summary_writer(log_dir: str) -> SummaryWriter:
    datestr = datetime.now().isoformat(timespec='minutes')

    return SummaryWriter(log_dir=os.path.join(log_dir, f"tb-{datestr}"))

@torch.no_grad()
def log_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
    n: int = 1,
) -> None:
    model.eval()

    raw_images, raw_masks = next(iter(loader))
    raw_images = raw_images[:n]
    raw_masks = raw_masks[:n]

    images, masks = normalize_batch(raw_images, raw_masks, device)

    preds = model(images)["out"].argmax(dim=1)

    imgs_display = denormalize_images(images).cpu()

    preds_display = preds.unsqueeze(1).float().cpu()
    masks_display = masks.unsqueeze(1).float().cpu()

    writer.add_images("preview/image", imgs_display, epoch)
    writer.add_images("preview/mask", masks_display, epoch)
    writer.add_images("preview/pred", preds_display, epoch)


def log_metrics(
    writer:     SummaryWriter,
    epoch:      int,
    train_loss: float,
    val_loss:   float,
    val_iou:    float,
    lr:         float,
) -> None:
    writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
    writer.add_scalar("mIoU/val", val_iou, epoch)
    writer.add_scalar("lr", lr, epoch)
