#!/usr/bin/env python
# coding: utf-8

from torchvision.models.video import r2plus1d_18
from simsiam import SimSiam

from data import VideoDataset

import torch

from tqdm import tqdm

import argparse

NUM_FRAMES = 22
SPLIT = 11

def train(dataloader, model, criterion, optimizer, epoch):
    for batch in tqdm(dataloader):
        data = batch
    
        # Split video frames into first and second half
        x1, x2 = data[:, :SPLIT], data[:, SPLIT:]
        # Transpose, since video resnet expects channels as first dim
        x1 = x1.transpose(1, 2)
        x2 = x2.transpose(1, 2)
    
        p1, p2, h1, h2 = model(x1, x2)

        loss = -(criterion(p1, h2).mean() + criterion(p2, h1).mean()) * 0.5

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"epoch {epoch} loss: {loss}")



def main():
    parser = argparse.ArgumentParser(description="Process training data parameters.")

    # Adding arguments
    parser.add_argument('--train_data', type=str, required=True, help='Path to the training data folder')
    parser.add_argument('--output', type=str, default="simsiam.pkl", help='Path to the output folder')

    # Hyperparam args
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

    # Parsing arguments
    args = parser.parse_args()

    # You can now use args.training_data, args.output, and args.num_epochs in your program
    print(f"Training Data Folder: {args.train_data}")
    print(f"Output file: {args.output}")
    print(f"Number of Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"SGD Learning Rate: {args.lr}")

    # In[10]:

    backbone = r2plus1d_18
    model = SimSiam(backbone)


    dataset = VideoDataset(args.train_data, 13000, idx_offset=2000, has_label=False)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # simsiam criterion
    criterion = torch.nn.CosineSimilarity(dim=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)


    # In[16]:
    for i in tqdm(range(args.num_epochs)):
        train(train_dataloader, model, criterion, optimizer, i + 1)

        torch.save(model, args.output)


if __name__ == "__main__":
    main()