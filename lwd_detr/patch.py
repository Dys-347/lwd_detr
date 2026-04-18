import inspect
import sys
import types

import torch
import torch.nn.functional as F

import ultralytics.nn.tasks as task_module
from ultralytics.utils.metrics import bbox_iou

from .pcir import PCIRLayer
from .drbc3 import DRBC3, DRBC3Block
from .mpdiou import mpdiou


task_module.PCIRLayer = PCIRLayer
task_module.DRBC3 = DRBC3


_parse_model_src = inspect.getsource(task_module.parse_model)

_parse_model_src = _parse_model_src.replace(
    "            RepC3,\n            PSA,",
    "            RepC3,\n            DRBC3,\n            PSA,",
)

_parse_model_src = _parse_model_src.replace(
    "            RepC3,\n            C2fPSA,",
    "            RepC3,\n            DRBC3,\n            C2fPSA,",
)

_parse_model_src = _parse_model_src.replace(
    "        elif m is ResNetLayer:\n            c2 = args[1] if args[3] else args[1] * 4",
    "        elif m is ResNetLayer:\n"
    "            c2 = args[1] if args[3] else args[1] * 4\n"
    "        elif m is PCIRLayer:\n"
    "            c2 = args[1] if args[3] else (args[1] * args[5] if len(args) > 5 else args[1])",
)

_exec_globals = task_module.__dict__.copy()
_exec_globals["__name__"] = task_module.__name__
exec(compile(_parse_model_src, inspect.getfile(task_module.parse_model), "exec"), _exec_globals)
task_module.parse_model = _exec_globals["parse_model"]


_original_bbox_iou = bbox_iou


def _patched_bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7, **kwargs):
    if kwargs.get("MPDIoU", False):
        return mpdiou(box1, box2, xywh=xywh, eps=eps)
    return _original_bbox_iou(box1, box2, xywh, GIoU, DIoU, CIoU, eps)


import ultralytics.utils.metrics

ultralytics.utils.metrics.bbox_iou = _patched_bbox_iou


import ultralytics.models.utils.loss as loss_module

_original_get_loss_bbox = loss_module.DETRLoss._get_loss_bbox


def _patched_get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=""):
    name_bbox = f"loss_bbox{postfix}"
    name_giou = f"loss_giou{postfix}"
    loss = {}
    if len(gt_bboxes) == 0:
        loss[name_bbox] = torch.tensor(0.0, device=self.device)
        loss[name_giou] = torch.tensor(0.0, device=self.device)
        return loss

    loss[name_bbox] = (
        self.loss_gain["bbox"] * F.l1_loss(pred_bboxes, gt_bboxes, reduction="sum") / len(gt_bboxes)
    )

    mpdiou_val = mpdiou(pred_bboxes, gt_bboxes, xywh=True)
    loss[name_giou] = (1.0 - mpdiou_val).sum() / len(gt_bboxes)
    loss[name_giou] = self.loss_gain["giou"] * loss[name_giou]
    return {k: v.squeeze() for k, v in loss.items()}


loss_module.DETRLoss._get_loss_bbox = _patched_get_loss_bbox


def fuse_drbc3(model):
    for m in model.modules():
        if isinstance(m, DRBC3Block):
            m.switch_to_deploy()
    return model
