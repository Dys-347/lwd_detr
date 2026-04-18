import argparse
from pathlib import Path

import torch

import lwd_detr.patch

from ultralytics import RTDETR


def parse_args():
    parser = argparse.ArgumentParser(description="Train LWD-DETR")
    parser.add_argument("--data", type=str, default="configs/strain_clamp.yaml", help="dataset config path")
    parser.add_argument("--cfg", type=str, default="configs/lwd-detr.yaml", help="model config path")
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="input image size")
    parser.add_argument("--device", type=str, default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="dataloader workers")
    parser.add_argument("--lr0", type=float, default=1e-4, help="initial learning rate")
    parser.add_argument("--lrf", type=float, default=1e-4, help="final learning rate fraction")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="optimizer")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="weight decay")
    parser.add_argument("--patience", type=int, default=50, help="early stopping patience")
    parser.add_argument("--name", type=str, default="lwd-detr-exp", help="experiment name")
    parser.add_argument("--resume", type=str, default="", help="resume from checkpoint")
    parser.add_argument("--pretrained", type=str, default="", help="pretrained weights path (optional)")
    parser.add_argument("--fuse-deploy", action="store_true", help="fuse DRBC3 blocks for deployment after training")
    return parser.parse_args()


def main():
    args = parse_args()

    model = RTDETR(args.cfg)

    if args.pretrained:
        model.load(args.pretrained)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        lr0=args.lr0,
        lrf=args.lrf,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        box=5.0,
        cls=2.0,
        name=args.name,
        resume=bool(args.resume),
        flipud=0.5,
        mixup=0.5,
        scale=0.5,
        warmup_epochs=5,
    )

    if args.fuse_deploy:
        from lwd_detr import fuse_drbc3
        fuse_drbc3(model.model)
        save_dir = Path(model.trainer.save_dir) if hasattr(model, "trainer") else Path("runs/detect") / args.name
        save_path = save_dir / "weights" / "fused_deploy.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.model}, str(save_path))
        print(f"[INFO] Fused deployment model saved to {save_path}")

    metrics = model.val()
    print("[INFO] Validation metrics:")
    print(metrics)


if __name__ == "__main__":
    main()
