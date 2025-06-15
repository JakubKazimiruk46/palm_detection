# Palm Detection

This project is a PyTorch-based hand gesture detection system designed to classify and localize gestures in images or webcam input. It distinguishes between two gesture classes: `palm` (open hand) and `no_gesture`.

The project includes training, evaluation, live webcam inference, and visualization utilities.

---

## 📦 Project Structure

```
project/
│
├── images/                 # Directory with user folders containing images
├── annotations.json        # JSON file with bounding boxes, labels, and user IDs
├── checkpoints/            # Saved model checkpoints during training
├── inference_results/      # Output predictions on new unlabeled images
├── visualization/          # Visual comparisons: ground truth vs predictions
├── new_test_images/        # Optional folder with unseen images for testing
├── second_version.py       # Main script with all logic
└── evaluation_results.json # Saved evaluation metrics
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

## How to Run

1. Place all images under `images/`, organized by user or category.
2. Add your `annotations.json` with bounding boxes, labels, and `user_id` for each image.
3. Open `second_version.py`, and set `enable_training = True` in the `Config` class if you wish to train the model.
4. Run the script:

```bash
python second_version.py
```

You will be prompted whether you want to:

* Run webcam detection
* Perform evaluation

---

## 🤖 Model

The script uses `SSDLite320_MobileNetV3_Large` from `torchvision` as the default object detector.

Features:

* Object detection with bounding boxes
* Class confidence scoring
* Supports switching to Faster R-CNN for higher accuracy (commented in code)

---

## 🧠 Training

* Dataset is split by unique `user_id`:

  * 80% of users for training
  * 20% for testing
* Separate 15% of users (using a different seed) are reserved for final evaluation
* During training, models are saved after each epoch to `checkpoints/`

Metrics:

* **IoU** (Intersection over Union)
* **Dice Coefficient**
* **mAP** (Mean Average Precision)

---

## 📷 Webcam Detection

After training, you can run live webcam detection:

```bash
Do you want to start live webcam detection? (y/n)
```

**Controls:**

* `q`: Quit
* `p`: Pause/Resume
* `+` / `-`: Adjust detection threshold

---

## 📊 Evaluation

To evaluate the best checkpoint on a separate validation set:

```bash
Do you want to run evaluation? (y/n)
```

* Results will be saved in `evaluation_results.json`
* Evaluates on IoU and Dice metrics
* Also supports visualizing predictions side-by-side with ground truth

---

## 🖼️ Visualization & Inference

* Visualization samples from the evaluation set are saved under `visualization/`
* You can also test the model on **new, unlabeled images** in `new_test_images/`

  * Results are saved to `inference_results/` and `new_images_results/`

---

## 📝 Notes

* Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.heic`, etc.
* Easy to extend to more gesture classes
* Default thresholds for detection and NMS are configurable in the script

---

## 📃 License

This is an academic/research project. Use it at your own discretion or adapt it to your organization's needs.

---
