"""
train.py
~~~~~~~~

Training loop with the things a long run actually needs: mixed precision,
a warmup + cosine schedule, resumable checkpoints, early stopping, CSV logging
and a proper test-set evaluation at the end.

The models return raw logits, so nn.CrossEntropyLoss is used directly -- it
applies log-softmax internally. Never put a softmax in the model.
"""

import copy
import csv
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from scripts.evaluations import (
    confusion_matrix,
    format_confusion_matrix,
    format_report,
    macro_f1,
)


# --------------------------------------------------------------------------
# one epoch
# --------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None,
                    grad_clip=None):
    model.train()
    running_loss, correct, seen = 0.0, 0, 0
    use_amp = scaler is not None and scaler.is_enabled()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        seen += images.size(0)

    return running_loss / max(seen, 1), correct / max(seen, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, scaler=None):
    """Returns (loss, accuracy, predictions, targets)."""
    model.eval()
    running_loss, correct, seen = 0.0, 0, 0
    all_preds, all_targets = [], []
    use_amp = scaler is not None and scaler.is_enabled()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        seen += images.size(0)

        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())

    if not all_preds:
        return 0.0, 0.0, torch.tensor([]), torch.tensor([])

    return (
        running_loss / max(seen, 1),
        correct / max(seen, 1),
        torch.cat(all_preds),
        torch.cat(all_targets),
    )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def make_scheduler(optimizer, warmup_epochs, total_epochs):
    """Linear warmup, then cosine decay to zero."""
    warmup = max(int(warmup_epochs), 0)

    def factor(epoch):
        if warmup > 0 and epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(total_epochs - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def save_checkpoint(path, model, optimizer=None, scheduler=None, epoch=None,
                    best_acc=None, extra=None, include_optimizer=False):
    """
    Write a checkpoint.

    By default only the weights are stored (small, portable, enough to run the
    model). Pass include_optimizer=True to also store the AdamW moments and
    scheduler state, which is what makes a run resumable -- but that triples
    the file size, so it goes in a separate .state.pt file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    blob = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "epoch": epoch,
        "best_acc": best_acc,
    }
    if include_optimizer:
        if optimizer is not None:
            blob["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            blob["scheduler"] = scheduler.state_dict()
    if extra:
        blob.update(extra)

    torch.save(blob, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    """Restore a run saved by save_checkpoint. Returns the stored metadata."""
    blob = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(blob["state_dict"])

    if optimizer is not None and "optimizer" in blob:
        optimizer.load_state_dict(blob["optimizer"])
    if scheduler is not None and "scheduler" in blob:
        scheduler.load_state_dict(blob["scheduler"])

    return {
        "epoch": blob.get("epoch"),
        "best_acc": blob.get("best_acc"),
        "best_epoch": blob.get("best_epoch"),
    }


class CsvLogger:
    """Append one row per epoch so a run can be plotted afterwards."""

    FIELDS = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc",
              "lr", "seconds", "best_acc"]

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._started = False

    def log(self, row):
        with open(self.path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.FIELDS)
            if not self._started and self.path.stat().st_size == 0:
                writer.writeheader()
            self._started = True
            writer.writerow({k: row.get(k, "") for k in self.FIELDS})


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------

def fit(
    model,
    bundle,
    epochs=10,
    lr=1e-3,
    device="cpu",
    weight_decay=1e-4,
    amp=True,
    grad_clip=1.0,
    patience=None,
    warmup_epochs=0,
    scheduler=True,
    save_path=None,
    log_csv=None,
    resume_from=None,
    class_names=None,
    verbose=True,
):
    """
    Train `model` on `bundle` (a data.DataBundle) and return a summary dict.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    sched = None
    if scheduler:
        sched = make_scheduler(optimizer, warmup_epochs, epochs)

    # mixed precision only pays off on CUDA
    use_amp = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    start_epoch = 1
    best_acc, best_epoch, best_weights = -1.0, 0, None

    if resume_from is not None:
        meta = load_checkpoint(resume_from, model, optimizer, sched, device)
        start_epoch = (meta["epoch"] or 0) + 1
        best_acc = meta["best_acc"] or -1.0
        best_epoch = meta.get("best_epoch") or 0
        if verbose:
            print(f"resumed from {resume_from} at epoch {start_epoch - 1} (best {best_acc:.3f})")

    state_path = Path(save_path).with_suffix(".state.pt") if save_path is not None else None
    logger = CsvLogger(log_csv) if log_csv else None
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    started, stale = time.time(), 0

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, bundle.train, criterion, optimizer, device, scaler, grad_clip
        )
        val_loss, val_acc, _, _ = evaluate(model, bundle.val, criterion, device, scaler)

        current_lr = optimizer.param_groups[0]["lr"]
        for key, value in (
            ("train_loss", train_loss), ("train_acc", train_acc),
            ("val_loss", val_loss), ("val_acc", val_acc), ("lr", current_lr),
        ):
            history[key].append(value)

        improved = val_acc > best_acc
        if improved:
            best_acc, best_epoch = val_acc, epoch
            best_weights = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        if verbose:
            print(
                f"epoch {epoch:3d}/{epochs}  "
                f"train {train_loss:.4f}/{train_acc:.3f}  "
                f"val {val_loss:.4f}/{val_acc:.3f}  "
                f"lr {current_lr:.2e}  ({time.time() - t0:.1f}s)"
                + ("  *" if improved else "")
            )

        if logger:
            logger.log({
                "epoch": epoch, "train_loss": f"{train_loss:.6f}",
                "train_acc": f"{train_acc:.6f}", "val_loss": f"{val_loss:.6f}",
                "val_acc": f"{val_acc:.6f}", "lr": f"{current_lr:.8f}",
                "seconds": f"{time.time() - t0:.1f}", "best_acc": f"{best_acc:.6f}",
            })

        if save_path is not None:
            save_checkpoint(save_path, model, epoch=epoch, best_acc=best_acc,
                            extra={"classes": class_names, "best_epoch": best_epoch})
            if state_path is not None:
                save_checkpoint(state_path, model, optimizer, sched, epoch, best_acc,
                                extra={"classes": class_names, "best_epoch": best_epoch},
                                include_optimizer=True)

        if sched is not None:
            sched.step()

        if patience is not None and stale >= patience:
            if verbose:
                print(f"\nearly stopping: no improvement for {patience} epochs")
            break

    elapsed = time.time() - started

    if best_weights is not None:
        model.load_state_dict(best_weights)
        # the per-epoch saves hold whichever epoch just ran; make sure the file
        # left on disk is the weights that actually scored best
        if save_path is not None:
            save_checkpoint(save_path, model, epoch=best_epoch, best_acc=best_acc,
                            extra={"classes": class_names, "best_epoch": best_epoch})

    if verbose:
        print(f"\nfinished in {elapsed:.1f}s -- best val acc {best_acc:.3f} at epoch {best_epoch}")

    return {
        "history": history,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "seconds": elapsed,
        "model": model,
    }


def evaluate_and_report(model, loader, device, class_names, scaler=None,
                        header="test"):
    """Run a full evaluation and print accuracy, per-class metrics, confusion matrix."""
    criterion = nn.CrossEntropyLoss()
    loss, acc, preds, targets = evaluate(model, loader, criterion, device, scaler)

    cm = confusion_matrix(targets, preds, len(class_names))
    print(f"\n=== {header} set ===")
    print(f"loss {loss:.4f}   accuracy {acc:.4f}   macro-F1 {macro_f1(cm):.4f}")
    print()
    print(format_report(cm, class_names))
    print()
    print(format_confusion_matrix(cm, class_names))

    return {
        "loss": loss,
        "accuracy": acc,
        "macro_f1": macro_f1(cm),
        "confusion_matrix": cm,
        "class_names": class_names,
    }
