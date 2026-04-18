import argparse

import lwd_detr.patch
from ultralytics import RTDETR


def parse_args():
    parser = argparse.ArgumentParser(description="Validate / Infer LWD-DETR")
    parser.add_argument("--weights", type=str, required=True, help="model weights path")
    parser.add_argument("--data", type=str, default="configs/strain_clamp.yaml", help="dataset config (for val)")
    parser.add_argument("--source", type=str, default="", help="image/video dir for inference")
    parser.add_argument("--imgsz", type=int, default=640, help="input size")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--device", type=str, default="0", help="device")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--save", action="store_true", help="save results")
    parser.add_argument("--save-txt", action="store_true", help="save predictions as YOLO txt")
    return parser.parse_args()


def main():
    args = parse_args()
    model = RTDETR(args.weights)

    if args.source:
        results = model.predict(
            source=args.source,
            imgsz=args.imgsz,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            save=args.save,
            save_txt=args.save_txt,
        )
        for r in results:
            print(r.path, r.boxes.cls.tolist(), r.boxes.conf.tolist())
    else:
        metrics = model.val(
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            save=args.save,
        )
        print("[INFO] Validation metrics:")
        print(metrics)


if __name__ == "__main__":
    main()
