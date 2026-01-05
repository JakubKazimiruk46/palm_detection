import os
from dataclasses import dataclass


@dataclass
class Config:
    ANNOTATIONS_PATH: str = '../annotations.json'
    IMAGES_BASE_PATH: str = '../images'
    SPLIT_FILE_PATH: str = os.path.join('../dataset', 'split.json')
    CHECKPOINT_DIR: str = '../checkpoints'
    VISUALIZATION_DIR: str = '../visualizations'

    BATCH_SIZE: int = 16
    NUM_EPOCHS: int = 30
    LEARNING_RATE: float = 0.0001
    MOMENTUM: float = 0.9
    WEIGHT_DECAY: float = 0.0005

    NUM_WORKERS: int = 2
    DEVICE: str = 'cuda'

    CONFIDENCE_THRESHOLD: float = 0.6
    IOU_THRESHOLDS: list = None

    def __post_init__(self):
        if self.IOU_THRESHOLDS is None:
            self.IOU_THRESHOLDS = [0.5]


CLASS_NAMES = ['palm', 'no_gesture']
