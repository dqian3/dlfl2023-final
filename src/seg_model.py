
import torch
import torch.nn as nn


class SegmentationModel(nn.Module):
    def __init__(self, siam, dim=2048, pred_dim=512):
        self.encoder = siam.encoder
        self.predictor = siam.predictor

        self.decoder = 
