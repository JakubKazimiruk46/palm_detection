import torch.nn
import torchvision
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection.ssdlite import SSDLite320_MobileNet_V3_Large_Weights
import logging

logger = logging.getLogger(__name__)


def get_object_detection_model(num_classes: int) -> torch.nn.Module:
    """
    Returns the SSDLite model with MobileNetV3 backbone.
    """
    logger.info(f"Initializing model for {num_classes} classes (including background).")

    weights_backbone = MobileNet_V3_Large_Weights.DEFAULT

    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        weights_backbone=weights_backbone,
        num_classes=num_classes + 1
    )

    return model
