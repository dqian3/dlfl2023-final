
import torch
import torch.nn as nn

from simsiam import SimSiam

'''
NN that takes a hidden representation of the first 11 frames of our video (size 512)
and outputs the semantic mask.

Should ouput a 256 x 256 x 49 mask 
'''
class SegmentationModel(nn.Module):

    '''
    Given pretrained simsiam model, construct the network
    '''
    def __init__(self, pretrained: SimSiam, finetune=False):
        super(SegmentationModel, self).__init__()
        
        self.finetune = finetune

        self.encoder = pretrained.encoder
        self.predictor = pretrained.predictor

        # TODO

    def forward(self, x):
        if self.finetune:
            x = self.encoder(x)
            x = self.predictor(x)
        else: 
            with torch.no_grad():
                x = self.encoder(x)
                x = self.predictor(x)

        # TODO
        pass