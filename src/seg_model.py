
import torch
import torch.nn as nn

from simsiam import SimSiam

'''
NN that takes a hidden representation of the first 11 frames of our video (size 512)
and outputs the semantic mask.

Should ouput a 240 x 160 x 49 mask 
'''
class SegmentationModel(nn.Module):

    '''
    Given pretrained simsiam model, construct the network
    '''
    def __init__(self, pretrained: SimSiam, prev_dim=1000, num_classes=49, finetune=False, use_predictor=True):
        super(SegmentationModel, self).__init__()
        
        self.finetune = finetune

        self.encoder = pretrained.encoder
        self.predictor = pretrained.predictor
        self.use_predictor = use_predictor

        # Project to 600 = 30 * 20 for easier upsampling later. Also allows network to unshuffle
        # Representation I guess?
        self.project = nn.Sequential(
            nn.Linear(prev_dim, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(), 
        )

        # 
        self.toclasses = nn.Conv1d(1, num_classes, kernel_size=1)

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(num_classes, num_classes, 8, stride=4, padding=2,
                output_padding=0, groups=num_classes,
                bias=False)
        )
        
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)


    def forward(self, x):
        if self.finetune:
            x = self.encoder(x)
            if (self.use_predictor):
                x = self.predictor(x)
        else: 
            with torch.no_grad():
                x = self.encoder(x)
                if (self.use_predictor):
                    x = self.predictor(x)

        x = self.project(x)

        # Add channels dim
        x = x.unsqueeze(1)
        x = self.toclasses(x)
        (B, C, D) = x.shape
        x = x.reshape(B, C, 20, 30)
        x = self.deconv1(x)
        x = self.upsample(x)

        return x