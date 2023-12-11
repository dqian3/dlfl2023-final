#!/usr/bin/env python
# coding: utf-8

# Local imports
from data import ValidationDataset
from unet.unet import * 

# DL packages
import torch
import torchmetrics

# Python packages
import os
import argparse
import time


def predict_simvp(model, dataset, device="cpu", batch_size=2, has_labels=True):
    start_time = time.time()
    print(f"Predicting SimVP results")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2)

    frames = []
    labels = []

    with torch.no_grad():
        for (i, batch) in enumerate(dataloader):
            data, target = batch
            data = data.to(device)
            data = data[:,:11]

            # Split video frames into first half
            frame = model(data)
            frames.append(frame[:,10].to("cpu"))

            if (has_labels):
                target = target.to(device)
                labels.append(target[:,21].to("cpu"))

            if (i + 1) % 100 == 0:
                print(f"After {time.time() - start_time:.2f} seconds finished training batch {i + 1} of {len(dataloader)}")

            del data

        frames = torch.stack(frames)

        if (has_labels):
            labels = torch.stack(labels) # 1k x 160 x 240

        return frames, labels


def predict_resnet50(model, frames, device="cpu", batch_size=2):
    start_time = time.time()
    print(f"Predicting Resnet50 results")

    dataset = torch.utils.data.TensorDataset(frames)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2)

    masks = []

    with torch.no_grad():
        for (i, frame) in enumerate(dataloader):
            frame = frame.to(device)
            mask = model(frame)
            masks.append(mask.to("cpu"))
            
            if (i + 1) % 100 == 0:
                print(f"After {time.time() - start_time:.2f} seconds finished training batch {i + 1} of {len(dataloader)}")

            del data

        masks = torch.stack(masks)

        return masks
    

def main():
    parser = argparse.ArgumentParser(description="Running validation set.")

    # Data arguments
    parser.add_argument('--data', type=str, required=True, help='Path to the training data (labeled) folder')
    parser.add_argument('--simvp', default=None, help='Path to pretrained simvp')
    parser.add_argument('--unet', default=None, help='Path to pretrained unet')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size')

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

    # Load Data
    # hidden_dataset = LabeledDataset(args.data)
    val_dataset = ValidationDataset(args.data)
    # hidden_dataloader = torch.utils.data.DataLoader(hidden_dataset, batch_size=args.batch_size, num_workers=2)

    val_frames, val_labels = predict_simvp(model, val_dataset, device, batch_size=args.batch_size)
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


    val_masks = predict_resnet50(model, val_frames, device, args.batch_size)

    iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)
    print(f"IOU: {iou(val_masks, val_labels)}")

    # torch.save(result_val, "val.tensor")
    # torch.save(result_hidden, "hidden.tensor")



if __name__ == "__main__":
    main()