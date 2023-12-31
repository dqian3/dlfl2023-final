#!/usr/bin/env python
# coding: utf-8

# Local imports
from data import UnlabeledDataset
from vidtoseg.simsiam import SimSiamGSTA
from vidtoseg.r2plus1d import R2Plus1DNet 
from vidtoseg.parallel2dconv import Parallel2DResNet
from util import save_model

# DL packages
import torch
from torchvision.models.video import r2plus1d_18
from tqdm import tqdm

# Python packages
import os
import argparse
import time

from vidtoseg.simsiam_orig import SimSiam


NUM_FRAMES = 22
SPLIT = 11

def train(dataloader, model, criterion, optimizer, device, epoch):
    total_loss = 0

    start_time = time.time()
    num_minutes = 0

    model.train()
    for (i, batch) in enumerate(dataloader):
        data = batch

        data = data.to(device)

        # Split video frames into first and second half
        x1, x2 = data[:, :SPLIT], data[:, SPLIT:]
        # Transpose, since video resnet expects channels as first dim
        x1 = x1.transpose(1, 2)
        x2 = x2.transpose(1, 2)
    
        p1, p2, h1, h2 = model(x1, x2)

        # Note original simsiam uses
        #         loss = -(criterion(p1, h2).mean() + criterion(p2, h1).mean()) * 0.5
        # Which is using both views of the data to train the predictor. However, we only
        # want to have the predictor predict the view of the first 11 frames
        loss = -(criterion(p1, h2).mean() + criterion(p2, h1).mean()) 

        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (time.time() - start_time) // 60 > num_minutes:
            num_minutes = (time.time() - start_time) // 60
            print(f"After {num_minutes} minutes, finished training batch {i + 1} of {len(dataloader)}")


    print(f"Loss at epoch {epoch} : {total_loss / len(dataloader)}")
    print(f"Took {(time.time() - start_time):2f} s")

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Process training data parameters.")

    # Data arguments
    parser.add_argument('--train_data', type=str, required=True, help='Path to the training (unlabeled) folder')
    parser.add_argument('--output', type=str, default="simsiam.pkl", help='Path to the output folder')
    parser.add_argument('--checkpoint', default=None, help='Path to the model checkpoint to continue training off of')

    # Hyperparam args
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

    # Other args
    parser.add_argument('--use_tqdm', action='store_true', help='Use tqdm in output')
    parser.add_argument('--original', action='store_true', help='Use original simsiam algorithm')

    # Parsing arguments
    args = parser.parse_args()

    print(f"Base training data folder: {args.train_data}")
    print(f"Output file: {args.output}")

    print(f"Number of epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"SGD learning rate: {args.lr}")

    # Define model
    if args.checkpoint:
        model = torch.load(args.checkpoint)
        print(f"Initializing model from weights of {args.checkpoint}")

    else:
        if args.original:
            model = SimSiam(Parallel2DResNet)
        else:
            model = SimSiamGSTA(Parallel2DResNet, 256 * 11)

        print(f"Initializing model from random weights")
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = torch.nn.DataParallel(model)

    print(f"model has {sum(p.numel() for p in model.parameters())} params")

    # Load Data
    dataset = UnlabeledDataset(args.train_data)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Try saving model and deleting, so we don't train an epoch before failing
    save_model(model, args.output)
    os.remove(args.output)

    # Train!
    criterion = torch.nn.CosineSimilarity(dim=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
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

        # Save model every epoch, in case our job dies lol
        save_model(model, args.output)

    print(train_loss)
    save_model(model, args.output)


if __name__ == "__main__":
    main()