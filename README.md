# Palm Detection System

This project is a PyTorch-based hand gesture detection system designed to classify and localize gestures (specifically open palms) in images or webcam input. It uses a modular architecture with **SSDLite MobileNetV3** as the backbone.

The system distinguishes between two classes: `palm` (open hand) and `no_gesture`.

---

## 📂 Project Structure

The project is organized into a modular structure to separate logic from configuration and entry points.

```text
project_root/
│
├── main.py                 # Main entry point for Training and Inference
├── annotations.json        # Dataset annotations (COCO-like format)
├── images/                 # Directory containing raw images
├── dataset/                # Generated split files (split.json)
├── checkpoints/            # Saved model weights (.pth)
├── visualizations/         # Training progress visualizations (Ground Truth vs Prediction)
├── scripts/                # Core logic modules
│   ├── config.py           # Hyperparameters and paths
│   ├── dataset.py          # Custom PyTorch Dataset class
│   ├── model.py            # Model definition (SSDLite)
│   ├── trainer.py          # Training loop and evaluation logic
│   ├── inference.py        # Webcam inference logic
│   ├── visualization.py    # Plotting and NMS logic
│   ├── split_data.py       # Data splitting script
│   └── utils.py            # Helper functions (logging, device)
│
└── requirements.txt        # Python dependencies
```

---

## Requirements

* Python 3.8+
* PyTorch
* torchvision
* numpy
* matplotlib
* opencv-python
* tqdm
* Pillow
* torchmetrics

To install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Setup & Data Preparation

Before running the model, you need to prepare the data and generate the train/test split.
1. Prepare Images: Place your images in the `images/` directory.
2. Annotations: Ensure `annotations.json` is in the root directory. It should map images to bounding boxes, labels, and `user_id`.
3. Generate Split: Run the splitting script. This script groups data by `user_id` to prevent data leakage (ensuring the same person doesn't appear in both train and test sets).

```bash
# Run from the project root
python -m scripts.split_data
```

This will create `dataset/split.json`.

You will be prompted whether you want to:

* Run webcam detection
* Perform evaluation

---

## Usage

Run the main script to access the menu:

```bash
python main.py
```

You will be presented with the following options:

1. Train Model
Starts the training pipeline. You will be asked to choose:
* Start from scratch: Initializes a new model (Pre-trained Backbone + New Head).
* Resume from best checkpoint: Loads the best model from `checkpoints/` and continues training (Fine-tuning).
Key Training Features:
* Optimizer: AdamW with Cosine Annealing scheduler.
* Augmentation: ColorJitter, GaussianBlur, Normalization.
* Visualization: Saves comparison images (Ground Truth vs. Prediction) to `visualizations/` after every epoch.
* Metrics: Calculates mAP (Mean Average Precision).

2. Run Webcam Inference
Starts live detection using your webcam and the best trained model found in `checkpoints/`.

Controls:
* `q`: Quit the application.

---

## Configuration

You can tweak hyperparameters in`scripts/config.py`:
```bash
BATCH_SIZE = 16          # Lower this if you run out of VRAM/RAM
NUM_EPOCHS = 30          # Total training epochs
LEARNING_RATE = 0.0001   # Initial learning rate for AdamW
CONFIDENCE_THRESHOLD = 0.6 # Threshold for webcam detection
NUM_WORKERS = 2          # Adjust based on your CPU cores
```

---

## Model & Technical Details

* Architecture: SSDLite320 with MobileNetV3-Large backbone.
* Input Size: Images are resized to 320x320 internally by the model.
* Post-Processing:
  * NMS (Non-Maximum Suppression): Used during visualization and inference to remove overlapping bounding boxes and keep only the most confident detection per object.
* Performance:
  * Uses `pin_memory` and `persistent_workers` in DataLoaders for faster CPU-GPU transfer.
  * Dataset indexing uses Set lookups $O(1)$ for speed.

---

## Outputs

* Checkpoints: Saved as `checkpoints/model_epoch_X_map_Y.pth`.
* Visualizations: Saved as `.jpg` images in `visualizations/`. Green boxes indicate Ground Truth, Red boxes indicate Model Predictions.

---

## License

This is an academic/research project. Use it at your own discretion or adapt it to your organization's needs.

---
