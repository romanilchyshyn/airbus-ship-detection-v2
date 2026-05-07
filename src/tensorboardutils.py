from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from imagenet import normalize_batch, denormalize_images

class TensorboardLogger:
    def __init__(self, log_dir: str) -> None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        self.writer = SummaryWriter(Path(log_dir) / f"tb-{timestamp}")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.writer.close()

    @torch.inference_mode()
    def log(
        self,
        model: nn.Module,
        loader,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_iou: float,
        lr: float,
        n: int = 1,
    ) -> None:
        was_training = model.training
        model.eval()

        try:
            raw_images, raw_masks = next(iter(loader))

            raw_images = raw_images[:n]
            raw_masks = raw_masks[:n]

            images, masks = normalize_batch(raw_images, raw_masks)
            preds = model(images)["out"].argmax(dim=1)

            self.writer.add_images(
                "preview/image",
                denormalize_images(images).cpu(),
                epoch,
            )
            self.writer.add_images(
                "preview/mask",
                masks.unsqueeze(1).float().cpu(),
                epoch,
            )
            self.writer.add_images(
                "preview/pred",
                preds.unsqueeze(1).float().cpu(),
                epoch,
            )

        finally:
            model.train(was_training)

        self.writer.add_scalars(
            "loss",
            {"train": train_loss, "val": val_loss},
            epoch,
        )
        self.writer.add_scalar("mIoU/val", val_iou, epoch)
        self.writer.add_scalar("lr", lr, epoch)
