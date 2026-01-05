import os
import torch
import logging
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import DataLoader

from config import Config
from visualization import save_epoch_visualization

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_loader: DataLoader,
            test_loader: DataLoader,
            config: Config,
            device: torch.device
    ) -> None:

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device

        #self.optimizer = torch.optim.SGD(
        #    model.parameters(),
        #    lr=config.LEARNING_RATE,
        #    momentum=config.MOMENTUM,
        #    weight_decay=config.WEIGHT_DECAY
        #)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )

        #self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
        #    self.optimizer,
        #    step_size=10,
        #    gamma=0.1
        #)

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.NUM_EPOCHS,
            eta_min=1e-6
        )

        os.makedirs(self.config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.config.VISUALIZATION_DIR, exist_ok=True)

    def train(self) -> None:
        logger.info(f"Starting training for {self.config.NUM_EPOCHS} epochs on {self.device}.")

        for epoch in range(self.config.NUM_EPOCHS):
            self.model.train()
            running_loss = 0.0

            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
            for images, targets in progress_bar:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                self.optimizer.zero_grad()

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                losses.backward()
                self.optimizer.step()

                running_loss += losses.item()
                progress_bar.set_postfix(loss=losses.item())

            self.lr_scheduler.step()
            avg_loss = running_loss / len(self.train_loader)
            logger.info(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")

            if (epoch + 1) % 5 == 0 or (epoch + 1) == self.config.NUM_EPOCHS:
                map_score = self.evaluate()
                self._save_checkpoint(epoch, avg_loss, map_score)

                logger.info(f"Saving visualizations to {self.config.VISUALIZATION_DIR}...")
                save_epoch_visualization(
                    model=self.model,
                    dataset=self.test_loader.dataset,
                    device=self.device,
                    epoch=epoch,
                    output_dir=self.config.VISUALIZATION_DIR,
                    num_samples=3
                )
            else:
                logger.info(f"Skipping evaluation for epoch {epoch+1}")
                self._save_checkpoint(epoch, avg_loss, map_value=0.0)

    def evaluate(self) -> float:
        self.model.eval()
        metric = MeanAveragePrecision(iou_thresholds=self.config.IOU_THRESHOLDS)

        logger.info("Starting evaluation...")
        with torch.no_grad():
            for images, targets in tqdm(self.test_loader, desc="Evaluating"):
                images = [img.to(self.device) for img in images]
                targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]

                predictions = self.model(images)
                predictions_cpu = [{k: v.cpu() for k, v in p.items()} for p in predictions]

                metric.update(predictions_cpu, targets_cpu)

        results = metric.compute()
        map_value = results['map'].item()
        logger.info(f"Validation mAP: {map_value:.4f}")
        return map_value

    def _save_checkpoint(self, epoch: int, loss: float, map_value: float) -> None:
        checkpoint_path = os.path.join(
            self.config.CHECKPOINT_DIR,
            f"model_epoch_{epoch+1}_map_{map_value:.3f}.pth"
        )
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'map': map_value
        }, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
