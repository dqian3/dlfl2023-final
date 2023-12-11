#!/usr/bin/env python
# coding: utf-8

# Local imports
from data import UnlabeledDataset
from simvp.modules import Decoder
from simvp.simvp import SimVP_Model
from util import save_model

# DL packages
import torch
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

        loss = None
        # Split video frames into multiple sequences of 11 and then 1 frame
        for target_frame in range(11, 22):
            x1 = data[:, target_frame-11:target_frame]
            x2 = data[:, target_frame]
            output = model(x1).squeeze(1) # squeeze last frame dim away
            
            if (loss is None):
                loss = criterion(output, x2)
            else:
                loss += criterion(output, x2)

            del output

        total_loss += loss.item() / 11
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (time.time() - start_time) // 60 > num_minutes:
            num_minutes = (time.time() - start_time) // 60
            print(f"After {num_minutes} minutes, finished training batch {i + 1} of {len(dataloader)}")

        if i == len(dataloader) - 1:
            # Split into first and second half
            x1, x2 = data[:,:11], data[:,11:]
            print(f"Autoregressive loss on last batch: {auto_regressive_loss(model, x1, x2)}")

    print(f"Loss at epoch {epoch} : {total_loss / len(dataloader)}")
    print(f"Took {(time.time() - start_time):2f} s")

    return total_loss / len(dataloader)


def auto_regressive_loss(model, data, target):
    criterion = torch.nn.MSELoss()
    with torch.no_grad():

        for i in range(11):
            output = model(data[:,i:])
            data = torch.cat((data, output), dim=1)

        return criterion(data[:,11:], target)


class DecoderProxy(torch.nn.Module):
    def __init__(self, decoder, hid_S):
        super(DecoderProxy, self).__init__()
        self.down_sample_skip = torch.nn.Conv2d(11 * hid_S, hid_S, 1)
        self.down_sample_hid = torch.nn.Conv2d(11 * hid_S, hid_S, 1)
        self.relu = torch.nn.ReLU()
        self.decoder = decoder
    
    def forward(self, hid, skip):
        hid = self.relu(self.down_sample_hid(hid))
        skip = self.relu(self.down_sample_skip(skip))
        return self.decoder(hid, skip)
    


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
    parser.add_argument('--size', type=str, default="small", help='Size of model')

    # Parsing arguments
    args = parser.parse_args()

    print(f"Base training data folder: {args.train_data}")
    print(f"Output file: {args.output}")

    print(f"Number of epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Adam learning rate: {args.lr}")

    # Define model
    if args.checkpoint:
        model = torch.load(args.checkpoint)
        print(f"Initializing model from weights of {args.checkpoint}")

    else:
        if args.size == "large":
            raise NotImplementedError()
            # model = SimVP_Model(in_shape=(11,3,160,240), hid_S=128, hid_T=512, N_T=10, N_S=8, drop_path=0.1)
        elif args.size == "med":
            raise NotImplementedError()
            # model = SimVP_Model(in_shape=(11,3,160,240), hid_S=128, hid_T=400, N_T=8, N_S=6, drop_path=0.1)
        else:
            model = SimVP_Model(in_shape=(11,3,160,240), hid_S=64, hid_T=512, N_T=8, N_S=6, drop_path=0.1)
            model.out_shape = (1, 3, 160, 240)
            model.dec = DecoderProxy(model.dec, 64)

        print(f"Initializing model from random weights")
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = torch.nn.DataParallel(model)

    print(f"model has {sum(p.numel() for p in model.parameters())} params")

    # Load Data
    dataset = UnlabeledDataset(args.train_data)
    val_dataset = UnlabeledDataset(args.train_data)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Try saving model and deleting, so we don't train an epoch before failing
    save_model(model, args.output)
    os.remove(args.output)

    # Train!
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

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