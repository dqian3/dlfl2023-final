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
    def __init__(self, root_dir, size, idx_offset=0, has_label=True, num_frames=22):
        self.root_dir = root_dir
        self.size = size
        self.idx_offset = idx_offset
        self.has_label = has_label

        self.num_frames = num_frames

        self.transform = transforms.Compose([
            # Skip some of the other transformations, since we are less worried about
            # scale and color variation
            transforms.Resize((256, 256)),
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
        data = torch.stack(data)
        
        # close file pointers
        for frame in frames:
            frame.close()
        
        if (self.has_label):
            label = np.load(os.path.join(img_dir, "mask.npy"))
            return data, label

        return data


# Shortcuts for our data
def LabeledDataset(base_dir):
    return VideoDataset(os.path.join(base_dir, "train"), 1000, idx_offset=0)

def UnlabeledDataset(base_dir):
    return VideoDataset(os.path.join(base_dir, "unlabeled"), 13000, idx_offset=2000, has_label=False)

def ValidationDataset(base_dir):
    return VideoDataset(os.path.join(base_dir, "val"), 1000, idx_offset=1000)


