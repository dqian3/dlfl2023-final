#!/usr/bin/env python
# coding: utf-8

# Local imports
from data import UnlabeledDataset
from simsiam import SimSiam

# DL packages
import torch
from torchvision.models.video import r2plus1d_18
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

    for batch in dataloader:
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
        loss = -criterion(p1, h2).mean()

        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Loss at epoch {epoch} : {total_loss / len(dataloader)}")
    print(f"Took {(time.time() - start_time):2f} s")

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Process training data parameters.")

    # Data arguments
    parser.add_argument('--train_data', type=str, required=True, help='Path to the training (unlabeled) folder')
    parser.add_argument('--output', type=str, default="simsiam.pkl", help='Path to the output folder')

    # Hyperparam args
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

    # Other args
    parser.add_argument('--use_tqdm', action='store_true', help='Use tqdm in output')

    # Parsing arguments
    args = parser.parse_args()

    print(f"Base training data folder: {args.train_data}")
    print(f"Output file: {args.output}")

    print(f"Number of epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"SGD learning rate: {args.lr}")

    # Define model
    backbone = r2plus1d_18
    model = SimSiam(backbone)

    # Load Data
    dataset = UnlabeledDataset(args.train_data)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Try saving model and deleting, so we don't train an epoch before failing
    torch.save(model, args.output)
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
    else:
        device = torch.device("cpu")

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