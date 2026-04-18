import torch


def mpdiou(box1, box2, xywh=True, eps=1e-7):
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.T, box2.T
        b1_x1, b1_x2 = x1 - w1 / 2, x1 + w1 / 2
        b1_y1, b1_y2 = y1 - h1 / 2, y1 + h1 / 2
        b2_x1, b2_x2 = x2 - w2 / 2, x2 + w2 / 2
        b2_y1, b2_y2 = y2 - h2 / 2, y2 + h2 / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    inter = (
        (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0)
        * (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    )

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    d1 = (b1_x1 - b2_x1) ** 2 + (b1_y1 - b2_y1) ** 2
    d2 = (b1_x2 - b2_x2) ** 2 + (b1_y2 - b2_y2) ** 2

    c_w = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    c_h = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = c_w ** 2 + c_h ** 2 + eps

    mpdiou_val = iou - (d1 + d2) / c2
    return mpdiou_val


class MPDIoULoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_boxes, target_boxes):
        loss = 1.0 - mpdiou(pred_boxes, target_boxes, xywh=True).mean()
        return loss
