import json
import os
import random
import warnings
from glob import glob
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision import models
from torchvision import transforms as T
from tqdm import tqdm
import multiprocessing
import cv2
import time
import torchvision.ops

IOU_THRESHOLDS = [0.3]

warnings.filterwarnings('ignore')

# Add these lines for Windows multiprocessing support

multiprocessing.freeze_support()

class_names = ['palm', 'no_gesture']  # Wszystkie obsługiwane klasy

FORMATS = (".jpeg", ".jpg", ".jp2", ".png", ".tiff", ".jfif", ".bmp", ".webp", ".heic")

# transform = T.ToTensor()
# Normalizacja do transformacji
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@dataclass
class Config:
    enable_training: bool = False
    annotations_path: str = 'annotations.json'
    images_base_path: str = 'images'
    checkpoint_dir: str = 'checkpoints'
    batch_size: int = 32
    num_epochs: int = 100
    num_visualizations: int = 5


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

            img_path = find_image_path(img_id, self.images_base_path)
            if img_path is None:
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
    return list(zip(*batch))


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
        metric = MeanAveragePrecision(iou_thresholds=IOU_THRESHOLDS)

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


def perform_training(model, config: Config, device):
    try:
        train_dataset, test_dataset = create_datasets(
            images_base_path=config.images_base_path,
            annotations_path=config.annotations_path
        )

        if len(train_dataset) == 0:
            print("Error: Training dataset is empty!")
            return
        if len(test_dataset) == 0:
            print("Error: Test dataset is empty!")
            return

        train_dataloader, test_dataloader = create_dataloaders(
            train_dataset, test_dataset
        )

        train_model(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            num_epochs=config.num_epochs,
            device=device
        )
        print("Training completed successfully!")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()


def perform_evaluation(model, config: Config, device):
    try:
        print("\nFinding best checkpoint...")
        best_checkpoint_path = find_best_checkpoint(config.checkpoint_dir)

        if best_checkpoint_path:
            print(f"Loading best model from {best_checkpoint_path}")
            checkpoint = torch.load(best_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Best validation mAP: {checkpoint.get('map', 0):.4f}")
        else:
            print("No checkpoints found. Cannot proceed with evaluation.")
            return

        print("\nCreating separate evaluation dataset...")
        eval_dataset = create_evaluation_dataset(
            annotations_path=config.annotations_path,
            images_base_path=config.images_base_path,
            transform=transform
        )

        if len(eval_dataset) == 0:
            print("Error: Evaluation dataset is empty!")
            return

        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )

        print("\nEvaluating model...")
        evaluation_results = evaluate_model(
            model=model,
            eval_dataloader=eval_dataloader,
            device=device
        )

        with open('evaluation_results.json', 'w') as f:
            json.dump(evaluation_results, f, indent=4)

        results_file = 'evaluation_results.json'

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
        visualize_predictions(
            model=model,
            eval_dataloader=eval_dataloader,
            device=device,
            num_samples=config.num_visualizations
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


def main():
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not os.path.exists(config.images_base_path):
        print(f"Error: Directory {config.images_base_path} does not exist!")
        return

    # Initialize model
    model = initialize_model(num_classes=len(class_names))
    model = model.to(device)  # Move to device immediately
    
    # If training is enabled, train the model
    if config.enable_training:
        perform_training(model, config, device)
    
    # Always load the best checkpoint (whether we just trained or not)
    best_checkpoint_path = find_best_checkpoint(config.checkpoint_dir)
    if best_checkpoint_path:
        print(f"Loading best model from {best_checkpoint_path}")
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Best validation mAP: {checkpoint.get('map', 0):.4f}")
    else:
        print("Warning: No checkpoints found. Using initialized model.")
    
    # Ask for webcam detection
    if os.path.exists(config.checkpoint_dir):
        print("\nDo you want to start live webcam detection? (y/n)")
        choice = input().lower()
        if choice == 'y' or choice == 'yes':
            print("Starting live webcam detection...")
            webcam_detection(
                model=model,
                device=device,
                confidence_threshold=0.65,  # Adjust based on your model
                enable_fps_display=True,
                mirror=True
            )
    
    # Run evaluation if needed
    print("\nDo you want to run evaluation? (y/n)")
    choice = input().lower()
    if choice == 'y' or choice == 'yes':
        perform_evaluation(model, config, device)


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
        "average_iou": float(avg_iou),  # Convert to Python float
        "num_samples": len(iou_scores),  # Already a Python int
        "all_dice_scores": [float(score) for score in dice_scores],  # Convert list of values to float
        "all_iou_scores": [float(score) for score in iou_scores]  # Convert list of values to float
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
            img_path = find_image_path(img_id, self.images_base_path)
            if img_path is None:
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


def find_image_path(img_id, images_base_path):
    for subdir in os.listdir(images_base_path):
        subdir_path = os.path.join(images_base_path, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for ext in FORMATS:
            potential_path = os.path.join(subdir_path, f"{img_id}{ext}")
            if os.path.exists(potential_path):
                return potential_path

    # Spróbuj znaleźć bezpośrednio
    for ext in FORMATS:
        potential_path = os.path.join(images_base_path, f"{img_id}{ext}")
        if os.path.exists(potential_path):
            return potential_path

    return None


def denormalize_image(tensor_img):
    # Convert tensor to PIL image for drawing
    image_np = tensor_img.permute(1, 2, 0).cpu().numpy()
    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = std * image_np + mean
    image_np = np.clip(image_np, 0, 1)
    return (image_np * 255).astype(np.uint8)


def draw_boxes(ax, boxes, labels, scores=None, color='green', title=''):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        label_text = f"Class: {class_names[labels[i] - 1]}"
        if scores is not None:
            label_text += f", Score: {scores[i]:.2f}"
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   fill=False, edgecolor=color, linewidth=2))
        ax.text(x1, y1, label_text, bbox=dict(facecolor=color, alpha=0.5))
    ax.set_title(title)
    ax.axis('off')


def save_visualization(image_np, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, sample_idx):
    plt.figure(figsize=(12, 6))

    # Ground truth
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(image_np)
    draw_boxes(ax1, gt_boxes, gt_labels, color='green', title='Ground Truth')

    # Predictions
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(image_np)
    draw_boxes(ax2, pred_boxes, pred_labels, pred_scores, color='red', title='Predictions')

    # Calculate and display metrics if possible
    if len(pred_boxes) > 0 and len(gt_boxes) > 0:
        pred_box = torch.tensor(pred_boxes[torch.argmax(torch.tensor(pred_scores))])
        gt_box = torch.tensor(gt_boxes[0])
        iou_val = box_iou(pred_box.unsqueeze(0), gt_box.unsqueeze(0)).item()
        dice_val = calculate_dice_coefficient(pred_box, gt_box)
        plt.suptitle(f"Sample {sample_idx} - IoU: {iou_val:.4f}, Dice: {dice_val:.4f}")

    plt.tight_layout()
    output_path = f'visualization/sample_{sample_idx}.png'
    plt.savefig(output_path)
    plt.close()
    print(f"Saved visualization for sample {sample_idx}")


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
                # Denormalize
                image_np = denormalize_image(image)

                gt_boxes = target['boxes'].cpu().numpy()
                gt_labels = target['labels'].cpu().numpy()

                pred_boxes = pred['boxes'].cpu().numpy()
                pred_labels = pred['labels'].cpu().numpy()
                pred_scores = pred['scores'].cpu().numpy()

                # Filtruj tylko predykcje z wysokim score
                mask = pred_scores > 0.5
                pred_boxes = pred_boxes[mask]
                pred_labels = pred_labels[mask]
                pred_scores = pred_scores[mask]

                save_visualization(
                    image_np=image_np,
                    gt_boxes=gt_boxes,
                    gt_labels=gt_labels,
                    pred_boxes=pred_boxes,
                    pred_labels=pred_labels,
                    pred_scores=pred_scores,
                    sample_idx=f"{i}_{j}"
                )

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



def webcam_detection(model, device, confidence_threshold=0.3, enable_fps_display=True, mirror=True):
    """
    Live webcam detection for palm gestures with proper filtering
    """
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    model.eval()
    model = model.to(device)
    print(f"Model on device: {next(model.parameters()).device}")
    
    fps_history = []
    
    print("Live Palm Detection Started")
    print(f"Using confidence threshold: {confidence_threshold}")
    print("Press 'q' to quit, 'p' to pause/resume, '+'/'-' to adjust threshold")
    
    paused = False
    
    # Function to filter and apply NMS to predictions
    def filter_predictions(prediction, confidence_threshold=0.3, nms_threshold=0.45):
        # Get prediction components
        boxes = prediction['boxes'].cpu()
        scores = prediction['scores'].cpu()
        labels = prediction['labels'].cpu()
        
        # Filter by confidence
        mask = scores > confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        # Apply NMS for each class separately
        result_boxes = []
        result_scores = []
        result_labels = []
        
        # Process each class
        for class_id in torch.unique(labels):
            class_mask = labels == class_id
            if not any(class_mask):
                continue
                
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]
            
            # Apply NMS
            keep_indices = torchvision.ops.nms(class_boxes, class_scores, nms_threshold)
            
            # Keep top 5 detections max
            keep_indices = keep_indices[:5]
            
            result_boxes.append(class_boxes[keep_indices])
            result_scores.append(class_scores[keep_indices])
            result_labels.append(torch.full((len(keep_indices),), class_id))
        
        # Combine results from all classes
        if result_boxes:
            result_boxes = torch.cat(result_boxes)
            result_scores = torch.cat(result_scores)
            result_labels = torch.cat(result_labels)
            return result_boxes, result_scores, result_labels
        else:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image from webcam")
            break
        
        if mirror:
            frame = cv2.flip(frame, 1)
        
        display_frame = frame.copy()
        
        if not paused:
            start_time = time.time()
            
            # Process frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            input_tensor = transform(pil_image).unsqueeze(0).to(device)
            
            # Perform detection
            with torch.no_grad():
                try:
                    predictions = model(input_tensor)
                    pred = predictions[0]
                    
                    # Apply filtering and NMS
                    filtered_boxes, filtered_scores, filtered_labels = filter_predictions(
                        pred, confidence_threshold, nms_threshold=0.45
                    )
                    
                    # Print detection info
                    if len(filtered_scores) > 0:
                        print(f"Filtered detections: {len(filtered_scores)}")
                        for i in range(len(filtered_scores)):
                            label = filtered_labels[i].item()
                            print(f"  {class_names[label-1]}: {filtered_scores[i].item():.4f}")
                    
                    # Draw filtered detections
                    palm_detected = False
                    for i in range(len(filtered_boxes)):
                        box = filtered_boxes[i].numpy()
                        score = filtered_scores[i].item()
                        label = filtered_labels[i].item()
                        class_name = class_names[label-1]
                        
                        x1, y1, x2, y2 = box.astype(int)
                        
                        # Use different colors for different classes
                        if class_name == 'palm':
                            color = (0, 255, 0)  # Green for palm
                            palm_detected = True
                        else:
                            color = (0, 0, 255)  # Red for no_gesture
                        
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(display_frame, f"{class_name}: {score:.2f}", (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Display status message
                    status_text = "Palm Detected" if palm_detected else "No Palm Detected"
                    status_color = (0, 255, 0) if palm_detected else (0, 0, 255)
                    cv2.putText(display_frame, status_text, (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                    
                except Exception as e:
                    print(f"Error during inference: {e}")
            
            # Calculate and display FPS
            if enable_fps_display:
                end_time = time.time()
                fps = 1 / (end_time - start_time + 1e-6)
                fps_history.append(fps)
                if len(fps_history) > 30:
                    fps_history.pop(0)
                avg_fps = sum(fps_history) / len(fps_history)
                
                cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            # If paused, display a message
            cv2.putText(display_frame, "PAUSED", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Show threshold
        cv2.putText(display_frame, f"Threshold: {confidence_threshold:.2f}", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display the result
        cv2.imshow('Palm Gesture Detection', display_frame)
        
        # Check for keypresses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('p'):  # Pause/Resume
            paused = not paused
        elif key == ord('+') or key == ord('='):  # Increase threshold
            confidence_threshold = min(confidence_threshold + 0.05, 1.0)
            print(f"Increased threshold to {confidence_threshold:.2f}")
        elif key == ord('-') or key == ord('_'):  # Decrease threshold
            confidence_threshold = max(confidence_threshold - 0.05, 0.05)
            print(f"Decreased threshold to {confidence_threshold:.2f}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam detection stopped")

def run_webcam_detection():
    """Run webcam detection with the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = initialize_model(num_classes=len(class_names))
    
    # Find and load the best checkpoint
    checkpoint_path = find_best_checkpoint("checkpoints")
    if checkpoint_path:
        print(f"Loading model from {checkpoint_path}")
        # Load with explicit device mapping
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Move model to device first
        model = model.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Verify model is on the correct device
        print(f"Model loaded successfully. First parameter device: {next(model.parameters()).device}")
        
        # Run webcam detection with a lower threshold
        webcam_detection(
            model=model,
            device=device,
            confidence_threshold=0.3,  # Lower threshold to see more detections
            enable_fps_display=True,
            mirror=True
        )
    else:
        print("No model checkpoint found. Please train the model first.")

if __name__ == "__main__":
    main()
