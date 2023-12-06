# https://github.com/facebookresearch/simsiam

import torch
import torch.nn as nn

from .r2plus1d import R2Plus1DNet
from .gsta import MidMetaNet

class SimSiamGSTA(nn.Module):
    """
    Build a SimSiam model.

    Makes the predictor layer much different...
    """
    def __init__(self, dim=1024, pred_dim=512):
        """
        dim: feature dimension (default: 1024)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiamGSTA, self).__init__()

        # create the backbone
        self.backbone = R2Plus1DNet()

        # build a gsta predictor
        self.predictor = MidMetaNet(2 * 256, 256, 3)


        # build a 2-layer projector
        self.dim = dim
        self.pool = nn.AdaptiveAvgPool3d((1, 2, 2))
        self.projector = nn.Sequential( nn.Linear(dim, dim, bias=False),
                                        nn.BatchNorm1d(dim),
                                        nn.ReLU(), 
                                        nn.Linear(dim, pred_dim, bias=False))

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        x1 = self.backbone(x1)
        x2 = self.backbone(x2)
        # N x 256 x 2 x 16 x 16

        p1 = self.predictor(x1)
        p2 = self.predictor(x2)
        # N x 256 x 2 x 16 x 16

        p1 = self.pool(p1).view((-1, self.dim))
        p2 = self.pool(p2).view((-1, self.dim))
        # N x 256 x 1 x 2 x 2 => N x 1024

        p1 = self.projector(p1)
        p2 = self.projector(p2)

        x1 = self.pool(x1).view((-1, self.dim))
        x2 = self.pool(x2).view((-1, self.dim))
        # N x 256 x 1 x 2 x 2 => N x 1024

        z1 = self.projector(x1) 
        z2 = self.projector(x2)

        return p1, p2, z1.detach(), z2.detach()