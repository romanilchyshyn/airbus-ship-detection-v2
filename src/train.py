import os
import time
import argparse
from datetime import datetime

import torch
import torch.nn as nn

from data import train_val_loader
from model import build_model, CLASSES
from imagenet import normalize_batch
from tensorboardutils import TensorboardLogger
from device import get_device

def compute_iou(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds, targets = preds.view(-1), targets.view(-1)
    num_classes = len(CLASSES)

    pred_oh   = preds.unsqueeze(0)   == torch.arange(num_classes, device=preds.device).unsqueeze(1)
    target_oh = targets.unsqueeze(0) == torch.arange(num_classes, device=preds.device).unsqueeze(1)

    intersection = (pred_oh & target_oh).sum(dim=1).float()  # [C]
    union        = (pred_oh | target_oh).sum(dim=1).float()  # [C]

    present = union > 0
    if not present.any():
        return 0.0

    return (intersection[present] / union[present]).mean().item() 

@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.eval()
    device = next(model.parameters()).device

    total_loss = torch.tensor(0.0, device=device)
    total_iou = 0.0
    total_samples = 0

    for images, masks in loader:
        images, masks = normalize_batch(images, masks)
        n = images.size(0)

        logits       = model(images)["out"]
        total_loss  += criterion(logits, masks) * n
        total_iou   += compute_iou(logits.argmax(dim=1), masks) * n
        total_samples += n

    if total_samples == 0:
        return 0.0, 0.0

    return total_loss.item() / total_samples, total_iou / total_samples

def train_one_epoch(
    model:       nn.Module,
    loader:      torch.utils.data.DataLoader,
    optimizer:   torch.optim.Optimizer,
    criterion:   nn.Module,
    scaler:      torch.amp.GradScaler,
    accum_steps: int = 1,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for i, (images, masks) in enumerate(loader):
        images, masks = normalize_batch(images, masks)

        with torch.autocast(device_type=get_device().type):
            loss = criterion(model(images)["out"], masks) / accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps

    return total_loss / len(loader)

def main() -> None:
    args = parse_args()
    
    datestr = datetime.now().strftime("%Y%m%d-%H%M")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    train_loader, val_loader = train_val_loader(
        args.data_dir,
        val_split=args.val_split,
        sample=args.sample,
        batch_size=args.batch_size,
    )
    print(f"Train: {len(train_loader) * args.batch_size} samples  |  Val: {len(val_loader) * args.batch_size} samples")

    model = build_model()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()

    best_iou = 0.0

    with TensorboardLogger(args.log_dir) as tb:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, args.accum_steps)
            val_loss, val_iou = evaluate(model, val_loader, criterion)
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

            tb.log(model, val_loader, epoch, train_loss, val_loss, val_iou, lr_now)

            if val_iou > best_iou:
                best_iou  = val_iou
                ckpt_path = os.path.join(args.checkpoint_dir, f"best_model-{datestr}.pth")
                torch.save({
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "optimizer":   optimizer.state_dict(),
                    "val_iou":     val_iou,
                    "val_loss":    val_loss,
                    "args":        vars(args),
                }, ckpt_path)
                print(f"Saved best model  (mIoU={best_iou:.4f})")

    print(f"\nTraining complete. Best val mIoU: {best_iou:.4f}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--data-dir",      type=str,   default="data")
    p.add_argument("--val-split",     type=float, default=0.15)
    p.add_argument("--sample",        type=int,   default=None)

    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--batch-size",    type=int,   default=10)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight-decay",  type=float, default=1e-4)
    p.add_argument("--accum-steps",   type=int,   default=1)

    p.add_argument("--checkpoint-dir", type=str,  default="checkpoints")
    p.add_argument("--log-dir",        type=str,  default="runs")

    return p.parse_args()

if __name__ == "__main__":
    main()