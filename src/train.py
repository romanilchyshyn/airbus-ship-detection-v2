import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)

from data import AirbusShipDetectionDataset

# ─────────────────────────────────────────────
# CONFIG  — edit these
# ─────────────────────────────────────────────
NUM_CLASSES    = 2               # background + 1 custom class
EPOCHS         = 30
BATCH_SIZE     = 10
LR             = 1e-4
WEIGHT_DECAY   = 1e-4
VAL_SPLIT      = 0.15            # 15% of data used for validation
CHECKPOINT_DIR = "checkpoints"   # where to save best model
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
FREEZE_BACKBONE = True           # True = only train the head (faster, less data needed)
                                 # False = train everything (better accuracy, needs more data)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: [B, C, H, W]   targets: [B, H, W]
        probs   = torch.softmax(logits, dim=1)
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union        = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice         = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """CrossEntropy + Dice — works well for binary segmentation."""
    def __init__(self):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return self.ce(logits, targets) + self.dice(logits, targets)


def compute_iou(preds, targets, num_classes=2):
    """Mean IoU over all classes."""
    ious = []
    preds   = preds.view(-1)
    targets = targets.view(-1)
    for cls in range(num_classes):
        pred_c   = preds   == cls
        target_c = targets == cls
        intersection = (pred_c & target_c).sum().float()
        union        = (pred_c | target_c).sum().float()
        if union == 0:
            continue
        ious.append((intersection / union).item())
    return sum(ious) / len(ious) if ious else 0.0


def build_model(num_classes, freeze_backbone=True):
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)

    model.classifier[4]     = nn.Conv2d(256, num_classes, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    if freeze_backbone:
        # Freeze the backbone — only the heads will update
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    return model


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        output = model(images)

        loss = criterion(output["out"], masks)

        # Auxiliary loss (helps training stability) — weighted lower
        if "aux" in output:
            loss += 0.4 * criterion(output["aux"], masks)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_iou  = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)

        output = model(images)
        logits = output["out"]

        loss = criterion(logits, masks)
        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        total_iou += compute_iou(preds.cpu(), masks.cpu(), NUM_CLASSES)

    return total_loss / len(loader), total_iou / len(loader)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    root = 'data'
    full_dataset = AirbusShipDetectionDataset(
        img_dir=os.path.join(root, 'train_v2'),
        masks_file=os.path.join(root, 'train_ship_segmentations_v2.csv')
    )

    val_size   = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"Train: {train_size} samples  |  Val: {val_size} samples")

    # ── Model ─────────────────────────────────
    model = build_model(NUM_CLASSES, freeze_backbone=FREEZE_BACKBONE).to(DEVICE)

    # ── Optimizer & scheduler ─────────────────
    # Only pass params that require grad
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    criterion = CombinedLoss()

    # ── Training loop ─────────────────────────
    best_iou = 0.0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss            = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_iou     = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{EPOCHS}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_mIoU={val_iou:.4f}  "
            f"lr={lr_now:.2e}  "
            f"time={elapsed:.1f}s"
        )

        # Save best checkpoint
        if val_iou > best_iou:
            best_iou = val_iou
            ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "val_iou":     val_iou,
                "val_loss":    val_loss,
            }, ckpt_path)
            print(f"  ✓ Saved best model  (mIoU={best_iou:.4f})")

    print(f"\nTraining complete. Best val mIoU: {best_iou:.4f}")
    print(f"Checkpoint saved to: {CHECKPOINT_DIR}/best_model.pth")


if __name__ == "__main__":
    main()