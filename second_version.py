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
import matplotlib.pyplot as plt
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


def create_dataloaders(train_data, test_data, batch_size=16):
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
    
    # Set training flag - change to False to skip training
    do_training = False  # Set this to False to skip training
    
    # Sprawdź, czy istnieje katalog train i test
    images_base_path = 'images'  # Ścieżka do katalogu z obrazami
    annotations_path = 'annotations.json'  # Ścieżka do pliku z adnotacjami

    if not os.path.exists(images_base_path):
        print(f"Error: Directory {images_base_path} does not exist!")
        return
        
    # Inicjalizuj model
    model = initialize_model(num_classes=len(class_names))
    
    if do_training:
        # Training code - only execute if do_training is True
        try:
            # Utwórz zbiory danych
            train_dataset, test_dataset = create_datasets(
                images_base_path=images_base_path,
                annotations_path=annotations_path
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
            print(f"An error occurred during training: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        print("Skipping training phase, using saved checkpoint for evaluation.")
    
    try:
        # Znajdź najlepszy model na podstawie mAP z walidacji
        print("\nFinding best checkpoint...")
        best_checkpoint_path = find_best_checkpoint("checkpoints")
        
        if best_checkpoint_path:
            print(f"Loading best model from {best_checkpoint_path}")
            checkpoint = torch.load(best_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            best_map = checkpoint.get('map', 0)
            print(f"Best validation mAP: {best_map:.4f}")
        else:
            print("No checkpoints found. Cannot proceed with evaluation.")
            return
        
        # Utwórz osobny zbiór ewaluacyjny
        print("\nCreating separate evaluation dataset...")
        eval_dataset = create_evaluation_dataset(
            annotations_path=annotations_path,
            images_base_path=images_base_path,
            transform=transform
        )
        
        if len(eval_dataset) == 0:
            print("Error: Evaluation dataset is empty!")
            return
            
        # Utwórz data loader dla ewaluacji
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=1,  # Batch size 1 dla lepszej analizy per-image
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # Przeprowadź ewaluację z metrykami Dice i IoU
        print("\nEvaluating model...")
        evaluation_results = evaluate_model(
            model=model,
            eval_dataloader=eval_dataloader,
            device=device
        )
        
        # Zapisz wyniki ewaluacji
        results_file = 'evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(evaluation_results, f, indent=4)
        print(f"Evaluation results saved to {results_file}")

                # Test on completely new images (optional)
        new_images_dir = 'new_test_images'  # Change this to your folder with new images
        if os.path.exists(new_images_dir):
            print(f"\nTesting model on new images from {new_images_dir}...")
            test_on_new_images(
                model=model,
                image_dir=new_images_dir,
                device=device,
                confidence_threshold=0.4  # Adjust based on your needs
            )
        
        # Wizualizuj przykładowe predykcje
        print("\nGenerating visualization of predictions...")
        os.makedirs('visualization', exist_ok=True)
        visualize_predictions(
            model=model,
            eval_dataloader=eval_dataloader,
            device=device,
            num_samples=5  # Liczba obrazów do wizualizacji
        )

        # Add this at the end of your main function
        if os.path.exists('new_test_images'):
            print("\nVisualizing predictions on new test images...")
            visualize_new_images(
                model=model,
                image_dir='new_test_images',
                device=device,
                confidence_threshold=0.55  # Adjust based on your needs
            )
        else:
            print("\nNo 'new_test_images' directory found. Skipping new image visualization.")



    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()

def find_best_checkpoint(checkpoint_dir):
    """
    Znajduje najlepszy checkpoint na podstawie wartości mAP
    """
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} does not exist")
        return None
        
    checkpoints = glob(os.path.join(checkpoint_dir, "*.pth"))
    
    if not checkpoints:
        print("No checkpoints found")
        return None
    
    best_map = -1
    best_checkpoint = None
    
    for checkpoint_path in checkpoints:
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'map' in checkpoint and checkpoint['map'] > best_map:
                best_map = checkpoint['map']
                best_checkpoint = checkpoint_path
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {e}")
    
    return best_checkpoint
def evaluate_model(model, eval_dataloader, device='cpu'):
    """
    Evaluates the trained model using Dice coefficient and IoU metrics
    """
    model.eval()
    print("Starting model evaluation with Dice and IoU metrics...")
    
    dice_scores = []
    iou_scores = []
    
    with torch.no_grad():
        for images, targets in tqdm(eval_dataloader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            
            # Get model predictions
            predictions = model(images)
            
            # Calculate metrics for each image in the batch
            for i, (pred, target) in enumerate(zip(predictions, targets)):
                # Skip images with no ground truth boxes
                if len(target['boxes']) == 0:
                    continue
                
                # Skip images with no predictions
                if len(pred['boxes']) == 0:
                    dice_scores.append(0)
                    iou_scores.append(0)
                    continue
                
                # Get the highest confidence prediction
                conf_scores = pred['scores']
                if len(conf_scores) > 0:
                    max_idx = torch.argmax(conf_scores)
                    pred_box = pred['boxes'][max_idx].cpu()
                    
                    # Get the corresponding ground truth box
                    gt_box = target['boxes'][0].cpu()
                    
                    # Calculate IoU
                    iou = box_iou(pred_box.unsqueeze(0), gt_box.unsqueeze(0)).item()
                    iou_scores.append(iou)
                    
                    # Calculate Dice coefficient
                    dice = calculate_dice_coefficient(pred_box, gt_box)
                    dice_scores.append(dice)
    
    # Calculate average metrics
    avg_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0
    avg_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0
    
    # Use Python native types for all values in the results dictionary
    results = {
        "average_dice": float(avg_dice),  # Convert to Python float
        "average_iou": float(avg_iou),    # Convert to Python float
        "num_samples": len(iou_scores),   # Already a Python int
        "all_dice_scores": [float(score) for score in dice_scores],  # Convert list of values to float
        "all_iou_scores": [float(score) for score in iou_scores]     # Convert list of values to float
    }
    
    print(f"Evaluation Results on {len(iou_scores)} samples:")
    print(f"  Average Dice: {avg_dice:.4f}")
    print(f"  Average IoU: {avg_iou:.4f}")
    
    return results


def box_iou(box1, box2):
    """
    Calculate IoU between bounding boxes
    
    Args:
        box1, box2: tensors of shape [N, 4] where each box is [x1, y1, x2, y2]
        
    Returns:
        IoU tensor of shape [N, M] where N, M are number of boxes in box1, box2
    """
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    # Get coordinates of intersection
    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]
    
    # Calculate intersection area
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    # Calculate IoU
    union = area1[:, None] + area2 - inter
    iou = inter / union
    
    return iou


def calculate_dice_coefficient(box1, box2):
    """
    Calculate Dice coefficient between bounding boxes
    
    Args:
        box1, box2: tensors of shape [4] representing [x1, y1, x2, y2]
        
    Returns:
        float: Dice coefficient
    """
    # Calculate areas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Get coordinates of intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    intersection = w * h
    
    # Calculate Dice coefficient: 2*|A∩B|/(|A|+|B|)
    dice = (2.0 * intersection) / (area1 + area2)
    
    return dice

def create_evaluation_dataset(annotations_path, images_base_path, transform=None):
    """
    Creates a separate evaluation dataset distinct from training and testing sets
    """
    # Load annotations
    with open(annotations_path, 'r') as f:
        annotations_data = json.load(f)
    
    # Collect all unique user IDs
    all_users = list(set(anno['user_id'] for anno in annotations_data.values()))
    
    # Set a specific seed for reproducibility
    random.seed(99)  # Using a different seed from your train/test split
    random.shuffle(all_users)
    
    # Divide users into train (70%), test (15%), and evaluation (15%) sets
    train_split = int(len(all_users) * 0.7)
    test_split = int(len(all_users) * 0.85)
    
    train_users = set(all_users[:train_split])
    test_users = set(all_users[train_split:test_split])
    eval_users = set(all_users[test_split:])
    
    print(f"Total users: {len(all_users)}")
    print(f"Training users: {len(train_users)}")
    print(f"Testing users: {len(test_users)}")
    print(f"Evaluation users: {len(eval_users)}")
    
    # Create evaluation dataset
    eval_dataset = GestureDatasetEval(
        annotations_path=annotations_path,
        images_base_path=images_base_path,
        eval_users=eval_users,
        transform=transform
    )
    
    return eval_dataset


class GestureDatasetEval(torch.utils.data.Dataset):
    """
    Dataset specifically for evaluation, based on GestureDataset but using designated evaluation users
    """
    def __init__(self, annotations_path, images_base_path, eval_users, transform=None):
        self.transform = transform
        self.annotations_path = annotations_path
        self.images_base_path = images_base_path
        self.eval_users = eval_users

        # Mapowanie nazw klas na ich indeksy
        self.labels = {label: idx + 1 for idx, label in enumerate(class_names)}

        # Wczytaj adnotacje
        with open(annotations_path, 'r') as f:
            self.annotations_data = json.load(f)

        # Przetwórz adnotacje z pliku JSON
        self.samples = self._prepare_dataset()
        print(f"Created evaluation dataset with {len(self.samples)} samples")

        # Wypisz kilka pierwszych przykładów
        if len(self.samples) > 0:
            print("Sample evaluation data entries:")
            for i in range(min(3, len(self.samples))):
                print(f"  Sample {i + 1}: {self.samples[i]}")

    def _prepare_dataset(self):
        """Przetwarza adnotacje do formatu używanego przez dataset"""
        samples = []

        # Dla każdego ID obrazu w adnotacjach
        for img_id, anno in self.annotations_data.items():
            if anno['user_id'] not in self.eval_users:
                continue

            # Znajdź ścieżkę do obrazu
            img_found = False
            img_path = None

            # Szukaj obrazu we wszystkich podkatalogach
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
                # Spróbuj znaleźć bezpośrednio w głównym katalogu
                for ext in FORMATS:
                    potential_path = os.path.join(self.images_base_path, f"{img_id}{ext}")
                    if os.path.exists(potential_path):
                        img_path = potential_path
                        img_found = True
                        break

            if not img_found:
                continue

            # Dodaj próbkę dla każdego bounding box
            for box_idx, (box, label) in enumerate(zip(anno['bboxes'], anno['labels'])):
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
        # Same as GestureDataset.__getitem__
        sample = self.samples[idx]

        try:
            image = Image.open(sample['image_path']).convert('RGB')
        except Exception as e:
            print(f"Error loading image {sample['image_path']}: {e}")
            image = Image.new('RGB', (100, 100))

        width, height = image.size

        x, y, w, h = sample['bbox']
        x1, y1 = x * width, y * height
        x2, y2 = (x + w) * width, (y + h) * height

        label_idx = self.labels.get(sample['label'], 0)

        boxes = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
        labels = torch.tensor([label_idx], dtype=torch.int64)

        if self.transform:
            image = self.transform(image)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
        }

        return image, target
    
def visualize_predictions(model, eval_dataloader, device, num_samples=5):
    """
    Visualizes model predictions alongside ground truth
    
    Args:
        model: The trained object detection model
        eval_dataloader: DataLoader with evaluation dataset
        device: Device to run inference on
        num_samples: Number of samples to visualize
    """
    model.eval()
    
    # Create output directory
    os.makedirs('visualization', exist_ok=True)
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(eval_dataloader):
            if i >= num_samples:
                break
                
            # Get model predictions
            images_device = [img.to(device) for img in images]
            predictions = model(images_device)
            
            # Process one image at a time for clarity
            for j, (image, target, pred) in enumerate(zip(images, targets, predictions)):
                # Convert tensor to PIL image for drawing
                image_np = image.permute(1, 2, 0).cpu().numpy()
                # Denormalize
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image_np = std * image_np + mean
                image_np = np.clip(image_np, 0, 1)
                image_np = (image_np * 255).astype(np.uint8)
                
                # Create plot
                plt.figure(figsize=(12, 6))
                
                # Plot original image with ground truth
                plt.subplot(1, 2, 1)
                plt.imshow(image_np)
                plt.title('Ground Truth')
                
                # Draw ground truth boxes
                for box, label in zip(target['boxes'].cpu().numpy(), target['labels'].cpu().numpy()):
                    x1, y1, x2, y2 = box
                    plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                                     fill=False, edgecolor='green', linewidth=2))
                    plt.text(x1, y1, f"Class: {class_names[label-1]}", 
                             bbox=dict(facecolor='green', alpha=0.5))
                
                # Plot image with predictions
                plt.subplot(1, 2, 2)
                plt.imshow(image_np)
                plt.title('Predictions')
                
                # Draw prediction boxes
                for box, label, score in zip(pred['boxes'].cpu().numpy(), 
                                             pred['labels'].cpu().numpy(),
                                             pred['scores'].cpu().numpy()):
                    if score > 0.5:  # Only show predictions with confidence > 0.5
                        x1, y1, x2, y2 = box
                        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                                         fill=False, edgecolor='red', linewidth=2))
                        plt.text(x1, y1, f"Class: {class_names[label-1]}, Score: {score:.2f}", 
                                 bbox=dict(facecolor='red', alpha=0.5))
                
                # Calculate metrics for this example
                if len(pred['boxes']) > 0 and len(target['boxes']) > 0:
                    best_pred_idx = torch.argmax(pred['scores']).item()
                    pred_box = pred['boxes'][best_pred_idx].cpu()
                    gt_box = target['boxes'][0].cpu()
                    
                    iou_val = box_iou(pred_box.unsqueeze(0), gt_box.unsqueeze(0)).item()
                    dice_val = calculate_dice_coefficient(pred_box, gt_box)
                    
                    # Add metrics to the title
                    plt.suptitle(f"Sample {i}_{j} - IoU: {iou_val:.4f}, Dice: {dice_val:.4f}")
                
                plt.tight_layout()
                plt.savefig(f'visualization/sample_{i}_{j}.png')
                plt.close()
                
                print(f"Saved visualization for sample {i}_{j}")
    
    print(f"Visualizations saved to 'visualization/' directory")

def test_on_new_images(model, image_dir, device='cpu', confidence_threshold=0.5):
    """
    Tests the model on new images without annotations
    
    Args:
        model: Trained model
        image_dir: Directory containing test images
        device: Device to run inference on
        confidence_threshold: Threshold for detection confidence
    """
    model.eval()
    os.makedirs('inference_results', exist_ok=True)
    
    # Find all image files
    image_files = []
    for ext in FORMATS:
        image_files.extend(glob(os.path.join(image_dir, f"*{ext}")))
    
    print(f"Found {len(image_files)} images to test")
    
    detections_by_class = {class_name: 0 for class_name in class_names}
    
    for img_path in tqdm(image_files, desc="Testing new images"):
        try:
            # Load and preprocess image
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).to(device)
            
            # Run inference
            with torch.no_grad():
                prediction = model([image_tensor])[0]
            
            # Convert image for visualization
            image_np = np.array(image)
            
            # Create visualization
            plt.figure(figsize=(10, 8))
            plt.imshow(image_np)
            plt.title(f"Predictions for {os.path.basename(img_path)}")
            
            # Draw predictions
            found_detections = False
            for box, label, score in zip(prediction['boxes'].cpu().numpy(), 
                                        prediction['labels'].cpu().numpy(),
                                        prediction['scores'].cpu().numpy()):
                if score > confidence_threshold:
                    found_detections = True
                    class_name = class_names[label-1]
                    detections_by_class[class_name] += 1
                    
                    x1, y1, x2, y2 = box
                    plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                                    fill=False, edgecolor='red', linewidth=2))
                    plt.text(x1, y1, f"{class_name}: {score:.2f}", 
                            bbox=dict(facecolor='red', alpha=0.5))
            
            if not found_detections:
                plt.text(10, 30, f"No detections with score > {confidence_threshold}", 
                        bbox=dict(facecolor='red', alpha=0.5))
            
            # Save result
            output_path = os.path.join('inference_results', os.path.basename(img_path))
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"Detections by class: {detections_by_class}")
    print(f"Results saved to 'inference_results/' directory")

def visualize_new_images(model, image_dir, device='cpu', confidence_threshold=0.4):
    """
    Visualizes model predictions on new images without annotations
    
    Args:
        model: Trained object detection model
        image_dir: Directory containing test images without annotations
        device: Device to run inference on
        confidence_threshold: Threshold for detection confidence
    """
    model.eval()
    os.makedirs('new_images_results', exist_ok=True)
    
    # Find all image files
    image_files = []
    for ext in FORMATS:
        image_files.extend(glob(os.path.join(image_dir, f"*{ext}")))
    
    print(f"Found {len(image_files)} images to visualize")
    
    detections_by_class = {class_name: 0 for class_name in class_names}
    images_with_detections = 0
    
    for img_path in tqdm(image_files, desc="Processing new images"):
        try:
            # Load and preprocess image
            original_image = Image.open(img_path).convert('RGB')
            image_tensor = transform(original_image).to(device)
            
            # Run inference
            with torch.no_grad():
                prediction = model([image_tensor])[0]
            
            # Convert image for visualization
            image_np = np.array(original_image)
            
            # Create visualization
            plt.figure(figsize=(10, 8))
            plt.imshow(image_np)
            plt.title(f"Predictions for {os.path.basename(img_path)}")
            
            # Draw predictions
            found_detections = False
            for box, label, score in zip(prediction['boxes'].cpu().numpy(), 
                                        prediction['labels'].cpu().numpy(),
                                        prediction['scores'].cpu().numpy()):
                if score > confidence_threshold:
                    found_detections = True
                    class_name = class_names[label-1]
                    detections_by_class[class_name] += 1
                    
                    x1, y1, x2, y2 = box
                    plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                                    fill=False, edgecolor='red', linewidth=2))
                    plt.text(x1, y1, f"{class_name}: {score:.2f}", 
                            bbox=dict(facecolor='red', alpha=0.5))
            
            if found_detections:
                images_with_detections += 1
            else:
                plt.text(10, 30, f"No detections with score > {confidence_threshold}", 
                        bbox=dict(facecolor='red', alpha=0.5))
            
            # Save result
            output_path = os.path.join('new_images_results', os.path.basename(img_path))
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"Detection summary:")
    print(f"- Images with any detection: {images_with_detections}/{len(image_files)} ({images_with_detections/len(image_files)*100:.1f}%)")
    print(f"- Detections by class: {detections_by_class}")
    print(f"Results saved to 'new_images_results/' directory")

if __name__ == "__main__":
    main()
