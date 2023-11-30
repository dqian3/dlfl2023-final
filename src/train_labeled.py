#!/usr/bin/env python
# coding: utf-8

# Local imports
from data import VideoDataset
from seg_model import SegmentationModel

# DL packages
import torch
from tqdm import tqdm

# Python packages
import os
import argparse
import time

NUM_FRAMES = 22
SPLIT = 11


def train(dataloader, model, criterion, optimizer, device, epoch):
    total_loss = 0

    start_time = time.time()
    num_minutes = 0

    for (i, batch) in enumerate(dataloader):
        data, labels = batch
        data = data.to(device)
        labels = labels.to(device)

        # Split video frames into first half
        x = data[:, :SPLIT]

        # get last mask by itself
        # Dim = (B x 160 x 240)
        # Probably need to flatten to use cross entropy
        label_masks = labels[:,21,:,:]

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
    parser.add_argument('--pretrained', type=str, default="simsiam.pkl", help='Path to pretrained simsiam network')
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
    pretrained = torch.load(args.pretrained)
    if args.checkpoint:
        model = torch.load(args.checkpoint)
        print(f"Initializing model from weights of {args.checkpoint}")
    else:
        model = SegmentationModel(pretrained)
        print(f"Initializing model from random weights")

    # Load Data
    dataset = VideoDataset(args.train_data, 1000, idx_offset=0, has_label=True)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Try saving model and deleting, so we don't train an epoch before failing
    torch.save(model, args.output)
    os.remove(args.output)

    # Train!
    criterion = None # TODO
    optimizer = None # TODO

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
    for i in iterator:
        epoch_loss = train(train_dataloader, model, criterion, optimizer, device, i + 1)
        train_loss.append(epoch_loss)

        # Save model every 10 epochs, in case our job dies lol
        if i % 10 == 9:
            file, ext = os.path.splitext(args.output)
            torch.save(model, file + f"_{i}" + ext)

    print(train_loss)
    torch.save(model, args.output)


if __name__ == "__main__":
    main()