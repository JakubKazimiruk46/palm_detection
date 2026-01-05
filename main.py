import json
import logging
import os
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from scripts.config import CLASS_NAMES, Config
from scripts.dataset import GestureDataset
from scripts.inference import run_webcam_inference
from scripts.model import get_object_detection_model
from scripts.trainer import Trainer
from scripts.utils import get_device, setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def collate_fn(batch: list[Any]) -> list[Any]:
    return list(zip(*batch))


def index_images(base_path: str) -> dict[str, str]:
    logger.info("Indexing images directory...")
    image_map = {}
    supported_formats = (".jpeg", ".jpg", ".png", ".bmp", ".webp")

    if not os.path.exists(base_path):
        logger.error(f"Images directory not found: {base_path}")
        return image_map

    for root, _, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(supported_formats):
                img_id = os.path.splitext(file)[0]
                image_map[img_id] = os.path.join(root, file)

    logger.info(f"Found {len(image_map)} images in {base_path}")
    return image_map


def get_dataloaders(config: Config) -> tuple[DataLoader, DataLoader, Any]:
    transform = T.Compose(
        [
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if not os.path.exists(config.SPLIT_FILE_PATH):
        logger.error(
            f"Split file not found at {config.SPLIT_FILE_PATH}. Run split_data.py first."
        )
        sys.exit(1)

    with open(config.SPLIT_FILE_PATH) as f:
        splits = json.load(f)

    full_image_map = index_images(config.IMAGES_BASE_PATH)

    logger.info("Creating datasets...")

    train_dataset = GestureDataset(
        annotations_path=config.ANNOTATIONS_PATH,
        images_path_map=full_image_map,
        user_ids=splits["train"],
        class_names=CLASS_NAMES,
        transform=transform,
    )

    test_dataset = GestureDataset(
        annotations_path=config.ANNOTATIONS_PATH,
        images_path_map=full_image_map,
        user_ids=splits["test"],
        class_names=CLASS_NAMES,
        transform=transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if config.NUM_WORKERS > 0 else False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if config.NUM_WORKERS > 0 else False,
    )

    return train_loader, test_loader, transform


def load_best_checkpoint(
    model: torch.nn.Module, checkpoint_dir: str, device: torch.device
) -> str | None:

    if not os.path.exists(checkpoint_dir):
        return None

    files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not files:
        return None

    files.sort(
        key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True
    )
    latest_file = os.path.join(checkpoint_dir, files[0])

    logger.info(f"Loading checkpoint: {latest_file}")
    checkpoint = torch.load(latest_file, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return latest_file


def main() -> None:
    config = Config()
    device = get_device()
    logger.info(f"Running on device: {device}")

    model = get_object_detection_model(num_classes=len(CLASS_NAMES))
    model = model.to(device)

    print("\n--- Gesture Detection System ---")
    print("1. Train Model")
    print("2. Run Webcam Inference")
    print("3. Exit")
    choice = input("Select option: ")

    if choice == "1":
        print("\n--- Training Mode ---")
        print("1. Start from scratch (New Training)")
        print("2. Resume from best checkpoint (Fine-tuning)")
        train_choice = input("Select option: ")

        train_loader, test_loader, _ = get_dataloaders(config)

        if train_choice == "2":
            ckpt = load_best_checkpoint(model, config.CHECKPOINT_DIR, device)
            if ckpt:
                print(f"Resuming training from {ckpt}")
            else:
                print("Warning: No checkpoint found. Starting from scratch.")

        elif train_choice == "1":
            print("Starting training from scratch...")

        else:
            print("Invalid option. Exiting.")
            sys.exit(1)

        trainer = Trainer(model, train_loader, test_loader, config, device)
        trainer.train()

    elif choice == "2":
        _, _, transform = get_dataloaders(config)

        ckpt = load_best_checkpoint(model, config.CHECKPOINT_DIR, device)
        if not ckpt:
            logger.warning("No checkpoint found! Using initialized (random) weights.")

        run_webcam_inference(
            model=model,
            device=device,
            class_names=CLASS_NAMES,
            transform=transform,
            confidence_threshold=config.CONFIDENCE_THRESHOLD,
        )

    elif choice == "3":
        sys.exit(0)


if __name__ == "__main__":
    main()
