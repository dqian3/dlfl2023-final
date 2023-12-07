#!/usr/bin/env python
# coding: utf-8

# Local imports
from data import LabeledDataset, ValidationDataset
from validate import validate
from vidtoseg.simsiam import SimSiamGSTA
from vidtoseg.unet import UNetVidToSeg

from util import save_model

# DL packages
import torch
from tqdm import tqdm


# Python packages
import os
import argparse
import time

NUM_FRAMES = 22
SPLIT = 11


def train_segmentation(dataloader, model, criterion, optimizer, device, epoch, target_frame=21):
    total_loss = 0

    start_time = time.time()
    num_minutes = 0
    print(f"Starting epoch {epoch}")

    model.train()
    for (i, batch) in enumerate(dataloader):
        data, labels = batch
        data = data.to(device)
        labels = labels.to(device)

        # Split video frames into first half
        x = data[:, :SPLIT]
        # Transpose, since video resnet expects channels as first dim
        x = x.transpose(1, 2)

        # get mask by itself
        # Dim = (B x 160 x 240)
        label_masks = labels[:,target_frame,:,:].long()

        # Predict and backwards
        pred_masks = model(x)

        loss = criterion(pred_masks, label_masks)

        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ((time.time() - start_time) // 60 > num_minutes):
            num_minutes = (time.time() - start_time) // 60
            print(f"After {num_minutes} minutes, finished training batch {i + 1} of {len(dataloader)}")

    print(f"Loss at epoch {epoch} : {total_loss / len(dataloader)}")
    print(f"Took {(time.time() - start_time):2f} s")

    return total_loss / len(dataloader)



def main():
    parser = argparse.ArgumentParser(description="Process training data parameters.")

    # Data arguments
    parser.add_argument('--train_data', type=str, required=True, help='Path to the training data (labeled) folder')
    parser.add_argument('--output', type=str, default="final_model.pkl", help='Path to the output folder')
    parser.add_argument('--pretrained', default=None, help='Path to pretrained simsiam network (or start a fresh one)')
    parser.add_argument('--checkpoint', default=None, help='Path to the model checkpoint to continue training off of')

    # Hyperparam args
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

    # Other args
    parser.add_argument('--use_tqdm', action='store_true', help='Use tqdm in output')

    # Parsing arguments
    args = parser.parse_args()

    print(f"Base training data folder: {args.train_data}")
    print(f"Pretrained model: {args.train_data}")
    print(f"Output file: {args.output}")
    
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"SGD learning rate: {args.lr}")

    # Define model
    if args.checkpoint:
        model = torch.load(args.checkpoint, map_location=torch.device('cpu'))
        print(f"Initializing model from weights of {args.checkpoint}")
    else:
        if (args.pretrained is None):
            print(f"Initializing base model from random weights")
            base_model = SimSiamGSTA()
        else:
            print(f"Using pretraine base model {args.pretrained}")
            base_model = torch.load(args.pretrained)

        model = UNetVidToSeg(base_model, finetune=True)
        print(f"Initializing model from random weights")
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = torch.nn.DataParallel(model)

    print(f"model has {sum(p.numel() for p in model.parameters())} params")

    # Load Data
    dataset = LabeledDataset(args.train_data)
    val_dataset = ValidationDataset(args.train_data)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Try saving model and deleting, so we don't train an epoch before failing
    save_model(model, args.output)
    os.remove(args.output)

    # Train!
    # Weight criterion so that empty class matters less!
    weights = torch.ones(49)
    weights[0] = 1 / 50 # rough estimate of number of pixels that are background

    criterion = torch.nn.CrossEntropyLoss(weight=weights) 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) # TODO

    iterator = range(args.num_epochs)
    if (args.use_tqdm): 
        iterator = tqdm(iterator)

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        device = torch.device("cuda:0")
        print("Using cuda!")

    else:
        device = torch.device("cpu")
        print("Using CPU!")


    train_loss = []
    best_iou = 0

    for i in iterator:
        epoch_loss = train_segmentation(train_dataloader, model, criterion, optimizer, device, i + 1)
        train_loss.append(epoch_loss)

        val_iou = validate(model, val_dataset, device=device, sample=100)
        print(f"IOU of validation set at epoch {i + 1}: {val_iou:.4f}")

        # Save model if it has the best iou
        if best_iou > val_iou:
            save_model(model, args.output)



    print(train_loss)
    save_model(model, args.output)


if __name__ == "__main__":
    main()