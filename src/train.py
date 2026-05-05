import os
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)

from data import train_val_loader
from utils import get_device

CLASSES     = ['ship', 'background']
NUM_CLASSES = len(CLASSES)

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs           = torch.softmax(logits, dim=1)
        targets_one_hot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1)
        intersection    = (probs * targets_one_hot).sum(dim=(2, 3))
        union           = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        return 1 - ((2 * intersection + self.smooth) / (union + self.smooth)).mean()


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, targets) + self.dice(logits, targets)


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def compute_iou(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds, targets = preds.view(-1), targets.view(-1)
    ious = []
    for cls in range(NUM_CLASSES):
        pred_c, target_c = preds == cls, targets == cls
        union = (pred_c | target_c).sum().float()
        if union == 0:
            continue
        ious.append(((pred_c & target_c).sum().float() / union).item())
    return sum(ious) / len(ious) if ious else 0.0


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def build_model(freeze_backbone: bool = True) -> nn.Module:
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    model.classifier[4]     = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)

    if freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = "classifier" in name

    return model


# ─────────────────────────────────────────────
# TRAIN / EVAL
# ─────────────────────────────────────────────
def train_one_epoch(
    model:       nn.Module,
    loader:      torch.utils.data.DataLoader,
    optimizer:   torch.optim.Optimizer,
    criterion:   nn.Module,
    device:      torch.device,
    accum_steps: int = 1,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for i, (images, masks) in enumerate(loader):
        images, masks = normalize_batch(images, masks, device)

        output = model(images)
        loss   = criterion(output["out"], masks)
        if "aux" in output:
            loss = loss + 0.4 * criterion(output["aux"], masks)

        (loss / accum_steps).backward()

        if (i + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

    # Handle leftover batches
    if len(loader) % accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = total_iou = 0.0

    for images, masks in loader:
        images, masks = normalize_batch(images, masks, device)
        logits = model(images)["out"]
        total_loss += criterion(logits, masks).item()
        total_iou  += compute_iou(logits.argmax(dim=1).cpu(), masks.cpu())

    n = len(loader)
    return total_loss / n, total_iou / n


# ─────────────────────────────────────────────
# TENSORBOARD HELPERS
# ─────────────────────────────────────────────

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
    writer.add_images("preview/pred", preds_display, epoch)
    writer.add_images("preview/gt_mask", masks_display, epoch)


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


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def normalize_images(images: torch.Tensor, device: torch.device) -> torch.Tensor:
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

def normalize_batch(
    images: torch.Tensor,
    masks:  torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    images = normalize_images(images, device)
    masks  = masks.long().to(device)

    return images, masks

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    device = get_device()
    print(f"Device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    run_name = args.run_name or f"deeplabv3_bs{args.batch_size}_{'frozen' if args.freeze_backbone else 'full'}"
    writer   = SummaryWriter(log_dir=os.path.join(args.log_dir, run_name))
    print(f"TensorBoard run: {run_name}")

    # ── Data ──────────────────────────────────
    train_loader, val_loader = train_val_loader(
        args.data_dir,
        val_split=args.val_split,
        sample=args.sample,
        batch_size=args.batch_size,
    )
    print(f"Train: {len(train_loader) * args.batch_size} samples  |  Val: {len(val_loader) * args.batch_size} samples")

    # ── Model ─────────────────────────────────
    model = build_model(freeze_backbone=args.freeze_backbone).to(device)

    # ── Optimizer & scheduler ─────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = CombinedLoss()

    # ── Training loop ─────────────────────────
    best_iou = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss        = train_one_epoch(model, train_loader, optimizer, criterion, device, args.accum_steps)
        val_loss, val_iou = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr_now  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_mIoU={val_iou:.4f}  "
            f"lr={lr_now:.2e}  "
            f"time={elapsed:.1f}s"
        )

        log_metrics(writer, epoch, train_loss, val_loss, val_iou, lr_now)

        if epoch % args.log_img_every == 0:
            log_predictions(model, val_loader, writer, epoch, device)

        if val_iou > best_iou:
            best_iou  = val_iou
            ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "val_iou":     val_iou,
                "val_loss":    val_loss,
                "args":        vars(args),
            }, ckpt_path)
            print(f"  ✓ Saved best model  (mIoU={best_iou:.4f})")

    writer.close()
    print(f"\nTraining complete. Best val mIoU: {best_iou:.4f}")
    print(f"Checkpoint saved to: {args.checkpoint_dir}/best_model.pth")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--data-dir",      type=str,   default="data")
    p.add_argument("--val-split",     type=float, default=0.15)
    p.add_argument("--sample",        type=int,   default=None)

    p.add_argument("--freeze-backbone", type=bool, default=True)

    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--batch-size",    type=int,   default=10)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight-decay",  type=float, default=1e-4)
    p.add_argument("--accum-steps",   type=int,   default=1)

    p.add_argument("--checkpoint-dir", type=str,  default="checkpoints")
    p.add_argument("--log-dir",        type=str,  default="runs")
    p.add_argument("--run-name",       type=str,  default=None)
    p.add_argument("--log-img-every",  type=int,  default=1)

    return p.parse_args()

if __name__ == "__main__":
    main()