from typing import Any
from torchmetrics import Metric
from torch import Tensor, tensor
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import torch


# Structural Similarity
class SSIM(Metric):
    full_state_update: bool = True

    def __init__(self):
        super().__init__()
        self.add_state("similarity", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor):
        preds = preds[0].cpu().detach().numpy().squeeze()
        targets = targets.cpu().detach().numpy().squeeze()

        values = tensor(
            [
                ssim(pred, target, data_range=pred.max() - target.min())
                for pred, target in zip(preds, targets)
            ],
        )

        self.similarity += values.sum()
        self.total += targets.shape[0]

    def compute(self):
        return self.similarity.float() / self.total


# Mean Squared Error
class MSE(Metric):
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.add_state("similarity", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor):
        preds = preds[0].cpu().detach().numpy().squeeze()
        targets = targets.cpu().detach().numpy().squeeze()

        values = tensor(
            [mse(pred, target) for pred, target in zip(preds, targets)],
        )

        self.similarity += values.sum()
        self.total += targets.shape[0]

    def compute(self):
        return self.similarity.float() / self.total
