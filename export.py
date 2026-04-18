import argparse

import lwd_detr.patch
from lwd_detr import fuse_drbc3
from ultralytics import RTDETR


def parse_args():
    parser = argparse.ArgumentParser(description="Export LWD-DETR")
    parser.add_argument("--weights", type=str, required=True, help="model weights path")
    parser.add_argument("--format", type=str, default="onnx", help="export format: onnx, engine, openvino, coreml, etc.")
    parser.add_argument("--imgsz", type=int, default=640, help="input image size")
    parser.add_argument("--device", type=str, default="cpu", help="device")
    parser.add_argument("--fuse", action="store_true", default=True, help="fuse DRBC3 reparameterization before export")
    parser.add_argument("--half", action="store_true", help="FP16 half precision")
    parser.add_argument("--simplify", action="store_true", default=True, help="simplify ONNX model")
    return parser.parse_args()


def main():
    args = parse_args()
    model = RTDETR(args.weights)

    if args.fuse:
        fuse_drbc3(model.model)
        print("[INFO] Fused DRBC3 blocks for deployment.")

    model.export(
        format=args.format,
        imgsz=args.imgsz,
        device=args.device,
        half=args.half,
        simplify=args.simplify,
    )


if __name__ == "__main__":
    main()
