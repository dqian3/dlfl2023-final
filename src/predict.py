#!/usr/bin/env python
# coding: utf-8

# Local imports
from data import LabeledDataset, ValidationDataset, HiddenDataset
from validate import validate
from vidtoseg.parallel2dconv import Parallel2DResNet
from vidtoseg.simsiam import SimSiamGSTA
from vidtoseg.unet import UNetVidToSeg

from util import save_model

# DL packages
import torch
import torchmetrics

# Python packages
import os
import argparse
import time


def predict_segmentation(model, dataset, device, batch_size, channels_first=False):

    start_time = time.time()

    model.eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2)

    with torch.no_grad():
        masks = []
        for data in dataloader:
            data = data.to(device)

            if channels_first:
                data = data.transpose(1, 2)
            # Split video frames into first half
            mask = model(data)
            if not channels_first:
                mask = mask.transpose(1, 2)

            mask = torch.argmax(model(data).transpose(1, 2), dim=1)
            masks.append(mask[:,10].to("cpu"))

    print(f"Took {(time.time() - start_time):2f} s")
    result = torch.stack(masks)
    print(result.shape)
    return result


def main():
    parser = argparse.ArgumentParser(description="Running validation set.")

    # Data arguments
    parser.add_argument('--val_data', type=str, required=True, help='Path to the training data base folder')
    parser.add_argument('--hidden_data', default=None, help='Path to the hidden data folder')
    parser.add_argument('--model', default=None, help='Path to simvp mask model')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size')
    parser.add_argument('--channels_first', action='store_true', help='Wheter model expects channel dimension before all spatio temporal dims')

    # Parsing arguments
    args = parser.parse_args()

    print(f"Training data folder: {args.val_data}")
    print(f"Model: {args.model}")
    
    # Define model
    model = torch.load(args.model, map_location=torch.device('cpu'))
    print(f"model has {sum(p.numel() for p in model.parameters())} params")

    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.device("cuda:0")
        print("Using cuda!")
    else:
        device = torch.device("cpu")
        print("Using CPU!")

    # Load Data
    val_dataset = ValidationDataset(args.val_data)

    iou, result_val = validate(model, val_dataset, device=device, batch_size=args.batch_size, channels_first=args.channels_first)

    print(f"IOU: {iou}")

    torch.save(result_val, f"{os.path.basename(args.model)}_val.tensor")
    del result_val

    if args.hidden_data:
        hidden_dataset = HiddenDataset(args.hidden_data)
        result_hidden = predict_segmentation(model, hidden_dataset, device=device, batch_size=args.batch_size, channels_first=args.channels_first)
        torch.save(result_hidden, f"{os.path.basename(args.model)}_hidden.tensor")



if __name__ == "__main__":
    main()