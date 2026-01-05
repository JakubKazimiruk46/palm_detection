import random

import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import torchvision.ops
from typing import List, Optional, Any
from torch.utils.data import Dataset
from config import CLASS_NAMES


def denormalize_image(tensor_img: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # [C, H, W] -> [H, W, C]
    image_np = tensor_img.permute(1, 2, 0).cpu().numpy()

    image_np = std * image_np + mean

    image_np = np.clip(image_np, 0, 1)
    return image_np


def draw_boxes(
        ax: Any,
        boxes: np.ndarray,
        labels: np.ndarray,
        scores: Optional[np.ndarray] = None,
        color: str = 'green'
) -> None:

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        label_idx = int(labels[i]) - 1
        if 0 <= label_idx < len(CLASS_NAMES):
            cls_name = CLASS_NAMES[label_idx]
        else:
            cls_name = f"Unknown({labels[i]})"

        label_text = f"{cls_name}"
        if scores is not None:
            label_text += f" {scores[i]:.2f}"

        rect = plt.Rectangle((x1, y1), w, h, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)

        ax.text(x1, y1, label_text,
                fontsize=8,
                color='white',
                bbox=dict(facecolor=color, alpha=0.7, edgecolor='none'))


def save_epoch_visualization(
        model: torch.nn.Module,
        dataset: Dataset,
        device: torch.device,
        epoch: int,
        output_dir: str,
        num_samples: int = 3
) -> None:
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    total_len = len(dataset)
    if total_len == 0:
        return

    indices = random.sample(range(total_len), min(num_samples, total_len))

    images = []
    targets = []

    for idx in indices:
        img, tgt = dataset[idx]
        images.append(img)
        targets.append(tgt)

    images_dev = [img.to(device) for img in images]

    with torch.no_grad():
        predictions = model(images_dev)

    for i, idx in enumerate(indices):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        img_np = denormalize_image(images[i])

        ax1.imshow(img_np)
        gt_boxes = targets[i]['boxes'].cpu().numpy()
        gt_labels = targets[i]['labels'].cpu().numpy()
        draw_boxes(ax1, gt_boxes, gt_labels, color='#00FF00')
        ax1.set_title(f"Ground Truth (Sample {idx})")
        ax1.axis('off')

        ax2.imshow(img_np)

        pred_boxes = predictions[i]['boxes']
        pred_scores = predictions[i]['scores']
        pred_labels = predictions[i]['labels']

        keep_score = pred_scores > 0.5
        pred_boxes = pred_boxes[keep_score]
        pred_scores = pred_scores[keep_score]
        pred_labels = pred_labels[keep_score]

        if len(pred_boxes) > 0:
            keep_nms = torchvision.ops.nms(pred_boxes, pred_scores, iou_threshold=0.3)

            final_boxes = pred_boxes[keep_nms].cpu().numpy()
            final_scores = pred_scores[keep_nms].cpu().numpy()
            final_labels = pred_labels[keep_nms].cpu().numpy()

            draw_boxes(ax2, final_boxes, final_labels, final_scores, color='#FF0000')

        ax2.set_title("Model Predictions (with NMS)")
        ax2.axis('off')

        save_path = os.path.join(output_dir, f"epoch_{epoch + 1}_sample_{idx}.jpg")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
