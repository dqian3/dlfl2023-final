import os

import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image

# These numbers are mean and std values computed over a sample of 500 training samples
normalize = transforms.Normalize(mean=[0.5062, 0.5045, 0.5009],
                                 std=[0.0570, 0.0568, 0.0613])

# Inverse transformation: needed for plotting.
unnormalize = transforms.Normalize(
   mean=[-0.5062/0.0570, -0.5045/0.0568, -0.5009/0.0613],
   std=[1/0.0570, 1/0.0568, 1/0.0613]
)


def unresize_mask(mask):
    mask = torch.tensor(mask)
    mask = mask.unsqueeze(0) # add a channel dim
    mask = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)(mask)
    mask = mask.squeeze(0) # remove fake channel dim

    return mask


class VideoDataset(Dataset):
    def __init__(self, root_dir, size, idx_offset=0, use_label=False, has_label=True, validating=False, is_hidden=False, num_frames=22):
        self.root_dir = root_dir
        self.size = size
        self.idx_offset = idx_offset
        self.has_label = has_label
        self.use_label = use_label
        self.is_hidden = is_hidden
        self.num_frames = num_frames

        self.transform = transforms.Compose([
            # Skip some of the other transformations, since we are less worried about
            # scale and color variation
            transforms.ToTensor(),  # convert PIL to Pytorch Tensor
            normalize,
        ])
    
    def __len__(self):
        return self.size

    def __getitem__(self, i):
        img_dir = os.path.join(self.root_dir, f"video_{self.idx_offset + i}")

        frames = [Image.open(os.path.join(img_dir, f"image_{j}.png")) for j in range(self.num_frames)]
        
        # Load the frames into data
        data = [self.transform(img) for img in frames]
        data_train = data[0:11]
        data_train = torch.stack(data_train)
        
        if (not self.is_hidden):
            data_target = data[11:]
            data_target = torch.stack(data_target)
       
        # close file pointers
        for frame in frames:
            frame.close()
        
#         if (self.has_label):
        if (self.use_label):
            label = np.load(os.path.join(img_dir, "mask.npy"))
            return data_train, data_target, label
        
        if (self.is_hidden):
            return data_train

        return data_train, data_target


# Shortcuts for our data
def LabeledDataset_pred(base_dir):
    return VideoDataset(os.path.join(base_dir, "train"), 1000, idx_offset=0)

def UnlabeledDataset_pred(base_dir):
    return VideoDataset(os.path.join(base_dir, "unlabeled"), 13000, idx_offset=2000)

def ValidationDataset_pred(base_dir):
    return VideoDataset(os.path.join(base_dir, "val"), 1000, idx_offset=1000)

def HiddenDataset_pred(base_dir):
    return VideoDataset(os.path.join(base_dir, "hidden"), 2000, idx_offset=15000, is_hidden=True)
