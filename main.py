"""
main.py
~~~~~~~

Entry point: wires the dataset (scripts/data.py), the architectures
(scripts/architectures/*) and the training loop (scripts/train.py) together.

Examples
--------
    python main.py                                    # LeNet, binary, 15 epochs
    python main.py --arch resnet --device cuda        # ResNet-50 on GPU
    python main.py --labels type --arch alexnet       # 4-class defect type
    python main.py --limit 256 --epochs 2             # fast smoke test
    python main.py --resume models/resnet.state.pt    # continue a run
    python main.py --data data_256 --arch resnet      # pre-resized cache
"""

import argparse
import sys
from pathlib import Path

import torch

from scripts import configs
from scripts.architectures.alexnet import AlexNet
from scripts.architectures.leenet import LeeNet
from scripts.architectures.resnet import ResNet50
from scripts.data import DatasetError, build_dataloaders
from scripts.train import evaluate_and_report, fit

MODELS = {
    "lenet": LeeNet,
    "alexnet": AlexNet,
    "resnet": ResNet50,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a CNN on the phone defect dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # data
    parser.add_argument("--data", default=str(configs.DATA_DIR), help="dataset folder")
    parser.add_argument("--labels", default="binary", choices=["binary", "type"],
                        help="binary: clean/defective, type: clean/dot/scratch/crack")
    parser.add_argument("--limit", type=int, default=None,
                        help="train on only the first N images (smoke tests)")
    parser.add_argument("--validate", action="store_true",
                        help="scan images at startup and skip unreadable ones")
    parser.add_argument("--workers", type=int, default=configs.NUM_WORKERS)

    # model
    parser.add_argument("--arch", default="lenet", choices=sorted(MODELS))
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the model (PyTorch 2+)")

    # optimisation
    parser.add_argument("--epochs", type=int, default=configs.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="default: chosen per architecture")
    parser.add_argument("--lr", type=float, default=None,
                        help="default: chosen per architecture")
    parser.add_argument("--weight-decay", type=float, default=configs.WEIGHT_DECAY)
    parser.add_argument("--warmup", type=int, default=configs.WARMUP_EPOCHS)
    parser.add_argument("--no-scheduler", action="store_true",
                        help="disable the cosine schedule")
    parser.add_argument("--patience", type=int, default=configs.EARLY_STOP_PATIENCE,
                        help="early stop after N epochs without improvement (0 = off)")
    parser.add_argument("--no-amp", action="store_true",
                        help="disable mixed precision on CUDA")

    # bookkeeping
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--out", default=str(configs.MODELS_DIR), help="checkpoint folder")
    parser.add_argument("--save", default=None, help="checkpoint name (default <arch>.pt)")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument("--log", default=None, help="CSV log path (default logs/<arch>.csv)")
    parser.add_argument("--no-log", action="store_true", help="disable CSV logging")
    parser.add_argument("--test", action="store_true", help="evaluate the test split at the end")
    parser.add_argument("--seed", type=int, default=configs.SEED)

    return parser.parse_args(argv)


def pick_device(requested):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    spec = configs.ARCHITECTURES[args.arch]
    device = pick_device(args.device)
    lr = args.lr if args.lr is not None else spec.lr
    batch_size = args.batch_size if args.batch_size is not None else spec.batch_size

    print(f"architecture : {args.arch}  (input {spec.image_size}x{spec.image_size})")
    print(f"device       : {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"lr           : {lr:g}   batch {batch_size}   epochs {args.epochs}")

    # ---- data ------------------------------------------------------------
    try:
        bundle = build_dataloaders(
            args.data,
            image_size=spec.image_size,
            batch_size=batch_size,
            num_workers=args.workers,
            seed=args.seed,
            device=device,
            label_mode=args.labels,
            limit=args.limit,
            validate=args.validate,
            with_test=args.test,
        )
    except DatasetError as exc:
        print(f"\ncannot build the dataset:\n  {exc}", file=sys.stderr)
        return 1

    print(bundle.describe())
    print()

    # ---- model -----------------------------------------------------------
    model = MODELS[args.arch](num_classes=len(bundle.classes))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters   : {n_params / 1e6:.2f}M")
    if args.compile:
        model = torch.compile(model)
        print("compile      : on")
    print()

    # ---- train -----------------------------------------------------------
    save_path = Path(args.out) / (args.save or f"{args.arch}.pt")
    log_path = None if args.no_log else Path(args.log or configs.LOGS_DIR / f"{args.arch}.csv")

    result = fit(
        model,
        bundle,
        epochs=args.epochs,
        lr=lr,
        device=device,
        weight_decay=args.weight_decay,
        amp=not args.no_amp,
        patience=args.patience or None,
        warmup_epochs=args.warmup,
        scheduler=not args.no_scheduler,
        save_path=save_path,
        log_csv=log_path,
        resume_from=args.resume,
        class_names=bundle.classes,
    )

    # ---- test ------------------------------------------------------------
    if args.test and bundle.test is not None:
        evaluate_and_report(result["model"], bundle.test, device, bundle.classes, header="test")

    print(f"\nbest validation accuracy: {result['best_acc']:.3f} (epoch {result['best_epoch']})")
    print(f"checkpoint: {save_path}")
    if log_path:
        print(f"log       : {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
