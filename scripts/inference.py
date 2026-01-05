import cv2
import torch
import time
import logging
from PIL import Image
from torchvision import transforms as T
from typing import List, Any

logger = logging.getLogger(__name__)


def run_webcam_inference(
        model: torch.nn.Module,
        device: torch.device,
        class_names: List[str],
        transform: Any,
        confidence_threshold: float = 0.6
) -> None:

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Could not open webcam.")
        return

    model.eval()
    model.to(device)

    logger.info("Starting webcam inference. Press 'q' to exit.")

    prev_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror effect
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()

            # Preprocessing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            img_tensor = transform(pil_image).unsqueeze(0).to(device)

            start_infer = time.time()
            with torch.no_grad():
                prediction = model(img_tensor)[0]
            inference_time = time.time() - start_infer

            for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores'].cpu().numpy()):
                if score > confidence_threshold:
                    x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                    lbl_idx = label.item()
                    # Safe access to class names
                    cls_name = class_names[lbl_idx - 1] if 0 <= lbl_idx - 1 < len(class_names) else f"Unknown({lbl_idx})"

                    color = (0, 255, 0) if cls_name == 'palm' else (0, 0, 255)

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    label_text = f"{cls_name}: {score:.2f}"
                    cv2.putText(display_frame, label_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # FPS Calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time

            # HUD
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display_frame, f"Infer: {inference_time*1000:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            window_name = 'Gesture Detection'
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    except Exception as e:
        logger.critical(f"Inference loop crashed: {e}", exc_info=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()
