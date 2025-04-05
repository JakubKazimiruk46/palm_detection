import os
import json
import logging
import random
from tqdm import tqdm
from collections import defaultdict
from typing import Tuple, List, Dict, Any
from glob import glob

import pandas as pd
import numpy as np

from PIL import Image, ImageOps
import cv2

import torch
from torch import nn, Tensor
from torchvision import models
from torchvision.transforms import Compose
from torchvision.transforms import functional as F
from torchvision import transforms as T
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import warnings

warnings.filterwarnings('ignore')

# Add these lines for Windows multiprocessing support
import multiprocessing

multiprocessing.freeze_support()

class_names = ['palm', 'no_gesture']  # Wszystkie obsługiwane klasy

FORMATS = (".jpeg", ".jpg", ".jp2", ".png", ".tiff", ".jfif", ".bmp", ".webp", ".heic")

# transform = T.ToTensor()
# Normalizacja do transformacji
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class GestureDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_path, images_base_path, is_train=True, transform=None):
        self.is_train = is_train
        self.transform = transform
        self.annotations_path = annotations_path
        self.images_base_path = images_base_path

        # Mapowanie nazw klas na ich indeksy
        self.labels = {label: idx + 1 for idx, label in enumerate(class_names)}

        # Wczytaj adnotacje
        with open(annotations_path, 'r') as f:
            self.annotations_data = json.load(f)

        # Przetwórz adnotacje z pliku JSON
        self.samples = self._prepare_dataset(is_train)
        print(f"Created dataset with {len(self.samples)} samples")

        # Wypisz kilka pierwszych przykładów, aby sprawdzić poprawność
        if len(self.samples) > 0:
            print("Sample data entries:")
            for i in range(min(3, len(self.samples))):
                print(f"  Sample {i + 1}: {self.samples[i]}")

    def _prepare_dataset(self, is_train):
        """Przetwarza adnotacje do formatu używanego przez dataset"""
        samples = []

        # Podziel użytkowników na grupy treningowe i testowe
        all_users = list(set(anno['user_id'] for anno in self.annotations_data.values()))
        random.seed(42)  # Dla powtarzalności
        random.shuffle(all_users)
        split_idx = int(len(all_users) * 0.8)
        train_users = set(all_users[:split_idx])
        test_users = set(all_users[split_idx:])

        target_users = train_users if is_train else test_users
        print(f"Total users: {len(all_users)}, {'Training' if is_train else 'Testing'} users: {len(target_users)}")

        # Dla każdego ID obrazu w adnotacjach
        for img_id, anno in self.annotations_data.items():
            if anno['user_id'] not in target_users:
                continue

            # Sprawdź, czy plik obrazu istnieje
            img_found = False
            img_path = None

            # Szukaj pliku we wszystkich podkatalogach images_base_path
            for subdir in os.listdir(self.images_base_path):
                subdir_path = os.path.join(self.images_base_path, subdir)
                if not os.path.isdir(subdir_path):
                    continue

                for ext in FORMATS:
                    potential_path = os.path.join(subdir_path, f"{img_id}{ext}")
                    if os.path.exists(potential_path):
                        img_path = potential_path
                        img_found = True
                        break

                if img_found:
                    break

            if not img_found:
                # Spróbuj znaleźć plik bezpośrednio w katalogu bazowym
                for ext in FORMATS:
                    potential_path = os.path.join(self.images_base_path, f"{img_id}{ext}")
                    if os.path.exists(potential_path):
                        img_path = potential_path
                        img_found = True
                        break

            if not img_found:
                # Nie znaleziono obrazu - pomijamy tę adnotację
                continue

            # Dla każdego bounding boxa w adnotacji
            for box_idx, (box, label) in enumerate(zip(anno['bboxes'], anno['labels'])):
                # Dodaj próbkę do listy
                samples.append({
                    'image_id': img_id,
                    'image_path': img_path,
                    'bbox': box,
                    'label': label,
                    'user_id': anno['user_id']
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Pobierz informacje o próbce
        sample = self.samples[idx]

        # Wczytaj obraz
        try:
            image = Image.open(sample['image_path']).convert('RGB')
        except Exception as e:
            print(f"Error loading image {sample['image_path']}: {e}")
            # Zwróć pusty obraz w przypadku błędu
            image = Image.new('RGB', (100, 100))

        width, height = image.size

        # Konwertuj bounding box do formatu [x1, y1, x2, y2] w pikselach
        x, y, w, h = sample['bbox']
        x1, y1 = x * width, y * height
        x2, y2 = (x + w) * width, (y + h) * height

        # Przygotuj etykietę
        label_idx = self.labels.get(sample['label'], 0)  # Domyślnie 0, jeśli etykieta nie jest znana

        # Przygotuj tensory dla detektora obiektów
        boxes = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
        labels = torch.tensor([label_idx], dtype=torch.int64)

        # Transformacja obrazu
        if self.transform:
            image = self.transform(image)

        # Zwróć dane w formacie wymaganym przez detektor obiektów
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
        }

        return image, target


def collate_fn(batch):
    """
    Funkcja collate_fn dla DataLoader, obsługująca przypadki z różnymi rozmiarami bounding boxów
    """
    images = []
    targets = []

    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)

    return images, targets


def create_datasets(images_base_path='images', annotations_path='annotations.json'):
    """
    Tworzy zbiory danych treningowych i testowych
    """
    # Sprawdź, czy ścieżki istnieją
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

    if not os.path.exists(images_base_path):
        raise FileNotFoundError(f"Images directory not found: {images_base_path}")

    print(f"Creating datasets from {annotations_path} and {images_base_path}")
    print(f"Available subdirectories in {images_base_path}: {os.listdir(images_base_path)}")

    # Utwórz zbiory danych
    train_dataset = GestureDataset(
        annotations_path=annotations_path,
        images_base_path=images_base_path,
        is_train=True,
        transform=transform
    )

    test_dataset = GestureDataset(
        annotations_path=annotations_path,
        images_base_path=images_base_path,
        is_train=False,
        transform=transform
    )

    return train_dataset, test_dataset


def create_dataloaders(train_data, test_data, batch_size=8):
    """
    Tworzy data loadery dla zbiorów treningowych i testowych
    """
    if len(train_data) == 0 or len(test_data) == 0:
        raise ValueError("Datasets are empty! Cannot create dataloaders.")

    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Ważne dla kompatybilności z Windows
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    return train_dataloader, test_dataloader


def initialize_model(num_classes):
    """
    Inicjalizuje model detekcji obiektów
    """
    # Używamy modelu SSDLite z MobileNetV3 jako backbone
    model = models.detection.ssdlite320_mobilenet_v3_large(
        pretrained=False,
        num_classes=3  # palm, no_gesture, background
    )

    # Alternatywnie, możesz użyć Faster R-CNN jeśli potrzebujesz lepszej dokładności (ale wolniejsze)
    # model = models.detection.fasterrcnn_resnet50_fpn(
    #     pretrained=False,
    #     num_classes=num_classes + 1  # +1 dla klasy tła
    # )

    return model


def train_model(model, train_dataloader, test_dataloader, num_epochs=30, device='cpu'):
    """
    Trenuje model detekcji obiektów
    """
    # Przenieś model na odpowiednie urządzenie
    model.to(device)

    # Inicjalizuj optymalizator
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0005
    )

    # Scheduler redukcji współczynnika uczenia
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.1
    )

    # Utwórz katalog na checkpointy
    os.makedirs('checkpoints', exist_ok=True)

    # Główna pętla treningu
    for epoch in range(num_epochs):
        # Tryb treningu
        model.train()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Zmienne do śledzenia straty
        running_loss = 0.0

        # Iteracja po batchu danych
        for i, (images, targets) in enumerate(tqdm(train_dataloader, desc="Training")):
            # Przenieś dane na urządzenie
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Zeruj gradienty
            optimizer.zero_grad()

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass i optymalizacja
            losses.backward()
            optimizer.step()

            # Aktualizuj zmienne śledzące
            running_loss += losses.item()

            # Wypisz bieżącą stratę co jakiś czas
            if i % 10 == 9:  # co 10 batchów
                print(f"  Batch {i + 1}, Loss: {running_loss / 10:.4f}")
                running_loss = 0.0

        # Aktualizuj scheduler
        lr_scheduler.step()

        # Ewaluacja
        print("Evaluating...")
        model.eval()

        # Użyj Mean Average Precision jako metryki
        # metric = MeanAveragePrecision()
        # Łagodniejsze dla testów
        metric = MeanAveragePrecision(iou_thresholds=[0.3])

        with torch.no_grad():
            for images, targets in tqdm(test_dataloader, desc="Evaluating"):
                images = [img.to(device) for img in images]
                targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]

                # Uzyskaj predykcje
                predictions = model(images)
                predictions_cpu = [{k: v.cpu() for k, v in p.items()} for p in predictions]

                # Aktualizuj metrykę
                metric.update(predictions_cpu, targets_cpu)

        # Oblicz metryki
        metrics = metric.compute()
        map_value = metrics['map'].item()

        print(f"  mAP: {map_value:.4f}")

        # Zapisz model
        checkpoint_path = f"checkpoints/model_epoch_{epoch + 1}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss,
            'map': map_value
        }, checkpoint_path)
        print(f"  Model saved to {checkpoint_path}")

    return model


def main():
    # Ustawienia
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Ustaw seed dla powtarzalności
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    try:
        # Sprawdź, czy istnieje katalog train i test
        images_base_path = 'images'  # Ścieżka do katalogu z obrazami

        if not os.path.exists(images_base_path):
            print(f"Error: Directory {images_base_path} does not exist!")
            return

        # Utwórz zbiory danych
        train_dataset, test_dataset = create_datasets(
            images_base_path=images_base_path,
            annotations_path='annotations.json'
        )

        if len(train_dataset) == 0:
            print("Error: Training dataset is empty!")
            return

        if len(test_dataset) == 0:
            print("Error: Test dataset is empty!")
            return

        # Utwórz data loadery
        train_dataloader, test_dataloader = create_dataloaders(
            train_dataset,
            test_dataset
        )

        # Inicjalizuj model
        model = initialize_model(num_classes=len(class_names))

        # Trenuj model
        trained_model = train_model(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            num_epochs=30,
            device=device
        )

        print("Training completed successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
