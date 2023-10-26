import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from mmseg.registry import MODELS
from .skip_head import BaseSkipHead
from mmengine.model import BaseModule

@MODELS.register_module()
class Threshold(BaseSkipHead):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def forward(self, input):
        """Forward function."""

        output = np.where(input >= self.threshold, 1, 0)
        output = torch.tensor(output)
        output = output[:, 0, :, :]

        return output

