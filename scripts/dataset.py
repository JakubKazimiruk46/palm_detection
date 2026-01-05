import os
import torch
import json
import logging
from PIL import Image
from torchvision import transforms as T
from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class GestureDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            annotations_path: str,
            user_ids: List[str],
            class_names: List[str],
            images_path_map: Dict[str, str],
            transform: Optional[Any] = None
    ) -> None:

        self.transform = transform
        self.labels_map = {label: idx + 1 for idx, label in enumerate(class_names)}
        self.image_path_map = images_path_map

        try:
            with open(annotations_path, 'r') as f:
                self.annotations_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load annotations from {annotations_path}: {e}")
            raise

        self.samples = self._prepare_dataset(user_ids)
        logging.info(f"Initialized dataset with {len(self.samples)} samples.")

    def _prepare_dataset(self, allowed_ids: List[str]) -> List[Dict[str, any]]:
        samples = []
        missing_images = 0

        allowed_ids_set = set(allowed_ids)

        for img_id, anno in self.annotations_data.items():
            if img_id not in allowed_ids_set:
                continue

            img_path = self.image_path_map.get(img_id)

            if not img_path:
                missing_images += 1
                continue

            for box, label in zip(anno['bboxes'], anno['labels']):
                samples.append({
                    'image_path': img_path,
                    'bbox': box,
                    'label': label
                })

        if missing_images > 0:
            logger.warning(f"Skipped {missing_images} samples because images were not found on disk.")

        return samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample = self.samples[idx]
        try:
            image = Image.open(sample['image_path']).convert('RGB')
        except Exception as e:
            logger.error(f"Corrupt image at {sample['image_path']}: {e}")
            image = Image.new('RGB', (100, 100))

        width, height = image.size
        x, y, w, h = sample['bbox']

        x1 = max(0, x * width)
        y1 = max(0, y * height)
        x2 = min(width, (x + w) * width)
        y2 = min(height, (y + h) * height)

        boxes = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
        labels = torch.tensor([self.labels_map.get(sample['label'], 0)], dtype=torch.int64)

        if self.transform:
            image = self.transform(image)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
        }

        return image, target

    def __len__(self) -> int:
        return len(self.samples)
