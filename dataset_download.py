import os
import zipfile
import shutil
import random
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as T  # This fixes the T.ToTensor() error
from torchvision import models
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm
import cv2
from glob import glob
from typing import Tuple

# Path to the local ZIP file
zip_file_path = 'subsample.zip'

# Check if the file exists
if not os.path.exists(zip_file_path):
    print(f"Error: Could not find {zip_file_path} in the current directory")
    print("Please make sure the file is in the same folder as this script")
    exit(1)

# Create directories
os.makedirs('dataset', exist_ok=True)

# Extract the ZIP file
try:
    print(f"Extracting {zip_file_path}...")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall('dataset/subsample')
    print("Successfully extracted the ZIP file to dataset/subsample")
except zipfile.BadZipFile:
    print(f"Error: {zip_file_path} is not a valid ZIP file")
    exit(1)
except Exception as e:
    print(f"Error extracting the ZIP file: {e}")
    exit(1)

# Copy the annotations if needed (this is equivalent to the scp command)
try:
    source_dir = './ann_subsample'  # Adjust this path if needed
    target_dir = './dataset/ann_subsample'
    if os.path.exists(source_dir):
        os.makedirs(target_dir, exist_ok=True)
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
        print(f"Copied annotations from {source_dir} to {target_dir}")
    else:
        print(f"Warning: Annotation directory {source_dir} not found")
        print("You may need to manually place annotations in the dataset/ann_subsample directory")
except Exception as e:
    print(f"Could not copy annotations: {e}")
    print("You may need to manually place annotations in the dataset/ann_subsample directory")

print("Setup completed")

class_names = [
   'call',
   'dislike',
   'fist',
   'four',
   'like',
   'mute',
   'ok',
   'one',
   'palm',
   'peace_inverted',
   'peace',
   'rock',
   'stop_inverted',
   'stop',
   'three',
   'three2',
   'two_up',
   'two_up_inverted',
   'no_gesture']

FORMATS = (".jpeg", ".jpg", ".jp2", ".png", ".tiff", ".jfif", ".bmp", ".webp", ".heic")

class_names = [
   'call',
   'dislike',
   'fist',
   'four',
   'like',
   'mute',
   'ok',
   'one',
   'palm',
   'peace_inverted',
   'peace',
   'rock',
   'stop_inverted',
   'stop',
   'three',
   'three2',
   'two_up',
   'two_up_inverted',
   'no_gesture']

FORMATS = (".jpeg", ".jpg", ".jp2", ".png", ".tiff", ".jfif", ".bmp", ".webp", ".heic")

transform = T.ToTensor()


class GestureDataset(torch.utils.data.Dataset):

    @staticmethod
    def __get_files_from_dir(pth: str, extns: Tuple):
        if not os.path.exists(pth):
            print(f"Dataset directory doesn't exist {pth}")
            return []
        files = [f for f in os.listdir(pth) if f.endswith(extns)]
        return files

    def __read_annotations(self, path):
      annotations_all = None
      exists_images = []
      found_annotations = False
         
      for target in class_names:
         path_to_csv = os.path.join(path, f"{target}.json")
         if os.path.exists(path_to_csv):
               found_annotations = True
               json_annotation = json.load(open(
                  os.path.join(path, f"{target}.json")
               ))

               json_annotation = [dict(annotation, **{"name": f"{name}.jpg"}) for name, annotation in
                                 zip(json_annotation, json_annotation.values())]

               annotation = pd.DataFrame(json_annotation)

               annotation["target"] = target
               if annotations_all is None:
                  annotations_all = annotation
               else:
                  annotations_all = pd.concat([annotations_all, annotation], ignore_index=True)
               
               exists_images.extend(
                  self.__get_files_from_dir(os.path.join(self.path_images, target), FORMATS))
         else:
               if target != 'no_gesture':
                  print(f"Database for {target} not found")
      
      if not found_annotations:
         raise ValueError(f"No annotation files found in {path}. Please ensure annotation files are in the correct location.")
         
      if annotations_all is None:
         raise ValueError("Could not load any annotations. Check that the annotation files exist and are in the correct format.")

      annotations_all["exists"] = annotations_all["name"].isin(exists_images)
      annotations_all = annotations_all[annotations_all["exists"]]

      users = annotations_all["user_id"].unique()
      users = sorted(users)
      random.Random(42).shuffle(users)
      train_users = users[:int(len(users) * 0.8)]
      val_users = users[int(len(users) * 0.8):]

      annotations_all = annotations_all.copy()

      if self.is_train:
         annotations_all = annotations_all[annotations_all["user_id"].isin(train_users)]
      else:
         annotations_all = annotations_all[annotations_all["user_id"].isin(val_users)]

      return annotations_all

    def __init__(self, path_annotation, path_images, is_train, transform=None):
        self.is_train = is_train
        self.transform = transform
        self.path_annotation = path_annotation
        self.path_images = path_images
        self.transform = transform
        self.labels = {label: num for (label, num) in
                       zip(class_names, range(len(class_names)))}
        self.annotations = self.__read_annotations(self.path_annotation)

    def __len__(self):
        return self.annotations.shape[0]

    def get_sample(self, index: int):
        row = self.annotations.iloc[[index]].to_dict('records')[0]
        image_pth = os.path.join(self.path_images, row["target"], row["name"])
        image = Image.open(image_pth).convert("RGB")

        labels = torch.LongTensor([self.labels[label] for label in row["labels"]])

        target = {}
        width, height = image.size

        bboxes = []

        for bbox in row["bboxes"]:
            x1, y1, w, h = bbox
            bbox_abs = [x1 * width, y1 * height, (x1 + w) * width, (y1 + h) * height]
            bboxes.append(bbox_abs)

        target["labels"] = labels
        target["boxes"] = torch.as_tensor(bboxes, dtype=torch.float32)
        target["orig_size"] = torch.as_tensor([int(height), int(width)])

        return image, target

    def __getitem__(self, index: int):
        image, target = self.get_sample(index)
        if self.transform:
            image = self.transform(image)
        return image, target
    
random_seed = 42
num_classes = len(class_names)
batch_size = 16
num_epoch = 15
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Change these paths to match your local setup
train_data = GestureDataset(path_images='./dataset/subsample',
                            path_annotation='./dataset/ann_subsample',
                            is_train=True, transform=transform)

test_data = GestureDataset(path_images='./dataset/subsample',
                            path_annotation='./dataset/ann_subsample',
                            is_train=False, transform=transform)

def collate_fn(batch):
    batch_targets = list()
    images = list()

    for b in batch:
        images.append(b[0])
        batch_targets.append({"boxes": b[1]["boxes"],
                              "labels": b[1]["labels"]})
    return images, batch_targets

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,collate_fn=collate_fn, shuffle=True, num_workers=4)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,collate_fn=collate_fn, shuffle=True, num_workers=4)

lr = 0.005
momentum = 0.9
weight_decay = 5e-4

model = models.detection.ssdlite320_mobilenet_v3_large(num_classes=len(class_names) + 1, pretrained_backbone=True)
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

warmup_factor = 1.0 / 1000
warmup_iters = min(1000, len(train_data) - 1)

lr_scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=warmup_factor, total_iters=warmup_iters
)

def eval(model, test_dataloader, epoch):
    model.eval()
    with torch.no_grad():
        mapmetric = MeanAveragePrecision()
        
        for images, targets in test_dataloader:
            images = list(image.to(device) for image in images)
            output = model(images)
            
            for pred in output:
                for key, value in pred.items():
                    pred[key] = value.cpu()
                    
            mapmetric.update(output, targets)

    metrics = mapmetric.compute()
    return metrics

os.makedirs('checkpoints', exist_ok=True)

for epoch in range(num_epoch):
    model.train()
    total = 0
    sum_loss = 0
    for images, targets in tqdm(train_dataloader):
        batch = len(images)
        images = list(image.to(device) for image in images)
        for target in targets:
            for key, value in target.items():
                target[key] = value.to(device)
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss = losses.item()
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        lr_scheduler_warmup.step()
        
        total = total + batch
        sum_loss = sum_loss + loss
    
    metrics = eval(model, test_dataloader, epoch)
    print(f"epoch : {epoch}  |||  loss : {sum_loss / total} ||| MAP : {metrics['map']}")

# Save the model from the final epoch
torch.save(model.state_dict(), f"checkpoints/{epoch}.pth")

images = []
for gesture in class_names[:-1]:
    image_path = glob(f'/kaggle/working/dataset/subsample/{gesture}/*.jpg')[0]
    images.append(Image.open(image_path))
    
images_tensors = images.copy()
images_tensors_input = list(transform(image).to(device) for image in images_tensors)

with torch.no_grad():
    model.eval()
    out = model(images_tensors_input)

bboxes = []
scores = []
labels = []

for pred in out:
    ids = pred['scores'] >= 0.2
    bboxes.append(pred['boxes'][ids][:2].cpu().numpy().astype(np.int))
    scores.append(pred['scores'][ids][:2].cpu().numpy())
    labels.append(pred['labels'][ids][:2].cpu().numpy())

short_class_names = []

for name in class_names:
    if name == 'stop_inverted':
        short_class_names.append('stop inv.')
    elif name == 'peace_inverted':
        short_class_names.append('peace inv.')
    elif name == 'two_up':
        short_class_names.append('two up')
    elif name == 'two_up_inverted':
        short_class_names.append('two up inv.')
    elif name == 'no_gesture':
        short_class_names.append('no gesture')
    else:
        short_class_names.append(name)

final_images = []
for bbox, score, label, image in zip(bboxes, scores, labels, images):
    image = np.array(image)
    for i, box in enumerate(bbox):
        _,width,_  = image.shape
        image = cv2.rectangle(image, box[:2], box[2:], thickness=3, color=[255, 0, 255])
        cv2.putText(image, f'{short_class_names[label[i]]}: {score[i]:0.2f}', (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX,
                        width / 780, (0, 0, 255), 2)
    final_images.append(Image.fromarray(image))

os.makedirs('out_images', exist_ok=True)

# Save each image
out_images = []
for i, image in enumerate(final_images):
    out_name = f"out_images/{i}.png"
    out_images.append(out_name)
    image.save(out_name)

out_dir = "out_images/"
@interact
def show_images(file=os.listdir(out_dir)):
    display(DImage(out_dir+file, width=600, height=300))