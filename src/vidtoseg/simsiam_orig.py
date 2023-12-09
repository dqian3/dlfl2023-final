# https://github.com/facebookresearch/simsiam

import torch
import torch.nn as nn


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, backbone, dim=1024, pred_dim=512):
        """
        dim: feature dimension (default: 1024)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the backbone
        self.backbone = backbone()

        # build a 3-layer projector
        self.projector = nn.Sequential(nn.AdaptiveAvgPool3d((1, 2, 2)),
                                       torch.nn.Flatten(start_dim=1, end_dim=-1),
                                       nn.Linear(dim, dim, bias=False),
                                        nn.BatchNorm1d(dim),
                                        nn.ReLU(), # first layer
                                        nn.Linear(dim, dim, bias=False),
                                        nn.BatchNorm1d(dim),
                                        nn.ReLU(), # second layer
                                        nn.Linear(dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer

        # build a 2-layer predictor
        # TODO: make this predictor smarter
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

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

        z1 = self.projector(x1) 
        z2 = self.projector(x2)

        p1 = self.predictor(z1) 
        p2 = self.predictor(z2) 

        return p1, p2, z1.detach(), z2.detach()