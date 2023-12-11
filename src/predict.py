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

    masks = []
    for batch in dataloader:
        data = batch
        data = data.to(device)

        # Split video frames into first half
        masks.append(model(data))

    print(f"Took {(time.time() - start_time):2f} s")
    result = torch.stack(masks)
    print(result.shape)
    return result

def validate_segmentation(dataloader, model, device):
    iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=49).to(device)

    start_time = time.time()

    model.eval()

    masks = []
    labels = []
    for batch in dataloader:
        data, target = batch
        data = data.to(device)
        target = target.to(device)

        # Split video frames into first half
        masks.append(model(data))
        labels.append(target)

    print(f"Took {(time.time() - start_time):2f} s")
    masks = torch.stack(masks)
    print(masks.shape)
    labels = torch.stack(labels)
    print(labels.shape)

    print(f"IOU: {iou(masks, labels)}")
    return masks


def main():
    parser = argparse.ArgumentParser(description="Running validation set.")

    # Data arguments
    parser.add_argument('--data', type=str, required=True, help='Path to the training data (labeled) folder')
    parser.add_argument('--model', default=None, help='Path to pretrained simsiam network (or start a fresh one)')
    parser.add_argument('--checkpoint', default=None, help='Path to the model checkpoint to continue training off of')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size')

    # Parsing arguments
    args = parser.parse_args()

    print(f"Base training data folder: {args.train_data}")
    print(f"Model: {args.train_data}")
    print(f"Output file: {args.output}")
    
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"SGD learning rate: {args.lr}")

    # Define model
    model = torch.load(args.model)
    print(f"model has {sum(p.numel() for p in model.parameters())} params")

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        device = torch.device("cuda:0")
        print("Using cuda!")
    else:
        device = torch.device("cpu")
        print("Using CPU!")

    # Load Data
    hidden_dataset = LabeledDataset(args.data)
    val_dataset = ValidationDataset(args.data)
    hidden_dataloader = torch.utils.data.DataLoader(hidden_dataset, batch_size=args.batch_size, num_workers=2)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=2)

    result_val = validate_segmentation(val_dataloader, model, device)
    # result_hidden = predict_segmentation(val_dataloader, model, device)

    torch.save(result_val, "val.tensor")
    # torch.save(result_hidden, "hidden.tensor")



if __name__ == "__main__":
    main()