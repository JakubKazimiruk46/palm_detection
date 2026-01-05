import logging
import sys

import torch


def setup_logging(log_file: str = 'app.log') -> None:
    """
    Configures logging to file and console.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def get_device() -> torch.device:
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
