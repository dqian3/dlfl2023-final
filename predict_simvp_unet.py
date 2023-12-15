#!/usr/bin/env python
# coding: utf-8

# Local imports
from data import ValidationDataset, HiddenDataset
from unet import * 
from simvp.simvp import SimVP_Model
from simvp.modules import *

# DL packages
import torch
import torchmetrics

# Python packages
import os
import argparse
import time


def predict_simvp(model, dataset, device="cpu", batch_size=2, has_labels=True, target_frame=21):
    start_time = time.time()
    print(f"Predicting SimVP results")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2)

    frames = []
    labels = []

    with torch.no_grad():
        for (i, batch) in enumerate(dataloader):

            if (has_labels):
                data, target = batch
            else:
                data = batch

            data = data.to(device)
            data = data[:,:11]

            output = model(data)
            frames.append(output[:,target_frame - 11].to("cpu"))

            if (has_labels):
                target = target.to(device)
                labels.append(target[:,target_frame].to("cpu"))

            if (i + 1) % 100 == 0:
                print(f"After {time.time() - start_time:.2f} seconds finished predicting batch {i + 1} of {len(dataloader)}")


        frames = torch.cat(frames) # 1k x 3 x 160 x 240

        if (has_labels):
            labels = torch.cat(labels) # 1k x 160 x 240
            return frames, labels
        else:
            return frames



def predict_resnet50(model, frames, device="cpu", batch_size=2):
    start_time = time.time()
    print(f"Predicting Resnet50 results")

    dataset = torch.utils.data.TensorDataset(frames)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2)
    masks = []

    model.eval()
    with torch.no_grad():
        for (i, batch) in enumerate(dataloader):
            (batch,) = batch
            batch = batch.to(device)

            batch = torch.nn.functional.interpolate(batch, size=(256, 256), mode='bilinear', align_corners=False)
            mask = model(batch)
            mask = torch.nn.functional.interpolate(mask, size=(160, 240), mode='bilinear', align_corners=False)
            mask = torch.argmax(mask, dim=1)

            masks.append(mask.to("cpu"))

            if (i + 1) % 100 == 0:
                print(f"After {time.time() - start_time:.2f} seconds finished predicting sample {i + 1} of {len(dataset)}")

        masks = torch.cat(masks)
        return masks
    

def main():
    parser = argparse.ArgumentParser(description="Running validation set.")

    # Data arguments
    parser.add_argument('--data', type=str, required=True, help='Path to the training data (labeled) folder')
    parser.add_argument('--hidden_data', default=None, help='Path to hidden data')
    parser.add_argument('--simvp', required=True, help='Path to pretrained simvp')
    parser.add_argument('--unet', required=True, help='Path to pretrained unet')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size')
    parser.add_argument('--target_frame', type=int, default=21, help='index of frame in sequence we want to segment and validate')

    # Parsing arguments
    args = parser.parse_args()

    print(f"Training data folder: {args.data}")
    print(f"Model: {args.simvp}")
    
    # Define model
    model = torch.load(args.simvp, map_location=torch.device('cpu'))
    print(f"simvp has {sum(p.numel() for p in model.parameters())} params")

    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.device("cuda:0")
        print("Using cuda!")
    else:
        device = torch.device("cpu")
        print("Using CPU!")

    val_dataset = ValidationDataset(args.data)
    val_frames, val_labels = predict_simvp(model, val_dataset, device, batch_size=args.batch_size, target_frame=args.target_frame)

    if args.hidden_data:
        hidden_dataset = HiddenDataset(args.hidden_data)
        hidden_frames = predict_simvp(model, hidden_dataset, device, batch_size=args.batch_size, has_labels=False)

    del model

    model = torch.load(args.unet, map_location=torch.device('cpu'))
    print(f"unet has {sum(p.numel() for p in model.parameters())} params")

    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.device("cuda:0")
        print("Using cuda!")
    else:
        device = torch.device("cpu")
        print("Using CPU!")


    val_masks = predict_resnet50(model, val_frames, device, batch_size=args.batch_size)

    iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)
    print(f"IOU: {iou(val_masks, val_labels)}")

    if args.hidden_data:
        hidden_masks = predict_resnet50(model, hidden_frames, device, batch_size=args.batch_size) 
        print(hidden_masks.shape)
        torch.save(hidden_masks, "simvp_unet_hidden.pt")

if __name__ == "__main__":
    main()