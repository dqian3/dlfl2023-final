#!/usr/bin/env python
# coding: utf-8

# Local imports
from data import LabeledDataset, ValidationDataset
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


def predict_segmentation(dataloader, model, device):

    start_time = time.time()

    model.eval()

    with torch.no_grad():
        masks = []
        for batch in dataloader:
            data = batch
            data = data.to(device)

            # Split video frames into first half
            mask = torch.argmax(model(data).transpose(1, 2), dim=1)
            masks.append(mask[:,10])

    print(f"Took {(time.time() - start_time):2f} s")
    result = torch.stack(masks)
    print(result.shape)
    return result


def main():
    parser = argparse.ArgumentParser(description="Running validation set.")

    # Data arguments
    parser.add_argument('--data', type=str, required=True, help='Path to the training data (labeled) folder')
    parser.add_argument('--model', default=None, help='Path to pretrained simsiam network (or start a fresh one)')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size')

    # Parsing arguments
    args = parser.parse_args()

    print(f"Training data folder: {args.data}")
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
    hidden_dataset = LabeledDataset(args.data)
    val_dataset = ValidationDataset(args.data)
    hidden_dataloader = torch.utils.data.DataLoader(hidden_dataset, batch_size=args.batch_size, num_workers=2)

    iou, result_val = validate(model, val_dataset, device=device, batch_size=args.batch_size)
    # result_hidden = predict_segmentation(val_dataloader, model, device)

    torch.save(result_val, "val.tensor")
    # torch.save(result_hidden, "hidden.tensor")



if __name__ == "__main__":
    main()