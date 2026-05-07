import os
import time
import argparse
from datetime import datetime

import torch
import torch.nn as nn

from utils import get_device
from data import train_val_loader
from model import build_model
from imagenet import normalize_batch
from tensorboardutils import (
    build_summary_writer, 
    log_predictions, 
    log_metrics,
)

CLASSES     = ['ship', 'background']
NUM_CLASSES = len(CLASSES)

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
        if "aux" in output: # fixme - don't use it
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

def main() -> None:
    args = parse_args()
    
    datestr = datetime.now().isoformat(timespec='minutes')

    device = get_device()
    print(f"Device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    writer = build_summary_writer(args.log_dir)

    train_loader, val_loader = train_val_loader(
        args.data_dir,
        val_split=args.val_split,
        sample=args.sample,
        batch_size=args.batch_size,
    )
    print(f"Train: {len(train_loader) * args.batch_size} samples  |  Val: {len(val_loader) * args.batch_size} samples")

    model = build_model().to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

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

    writer.close()
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
    p.add_argument("--log-img-every",  type=int,  default=1)

    return p.parse_args()

if __name__ == "__main__":
    main()