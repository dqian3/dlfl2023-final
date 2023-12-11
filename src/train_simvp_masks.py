#!/usr/bin/env python
# coding: utf-8

# Local imports
from data import LabeledDataset, ValidationDataset, UnetLabeledDataset
from simvp.modules import Decoder
from simvp.simvp import SimVP_Model
from validate import validate


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


def train_segmentation(dataloader, model, criterion, optimizer, device, epoch):
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

        # get mask by itself
        # Dim = (B x 160 x 240)
        label_masks = labels[:,SPLIT:].long()

        # Predict and backwards
        pred_masks = model(x).transpose(1, 2) # Switch channel and time

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
    parser.add_argument('--unet_labels', default=None, help='Path to the unet_labels (unlabeled data segmented by unet) folder')
    parser.add_argument('--output', type=str, default="final_model.pkl", help='Path to the output folder')
    parser.add_argument('--pretrained', default=None, help='Path to pretrained simsiam network (or start a fresh one)')
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
    print(f"Pretrained model: {args.train_data}")
    print(f"Output file: {args.output}")
    
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Adam learning rate: {args.lr}")

    if args.size == "large":
        hid_S = 128
        hid_T = 512
        N_T = 10
        N_S = 8
    elif args.size == "med":
        hid_S = 128
        hid_T = 256
        N_T = 8
        N_S = 6
    else:
        hid_S = 64
        hid_T = 512
        N_T = 8
        N_S = 4

    # Define model
    if args.checkpoint:
        print(f"Initializing model from weights of {args.checkpoint}")

        model = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    else:
        if (args.pretrained is None):
            print(f"Initializing base model from random weights")
            model = SimVP_Model(in_shape=(11,3,160,240), hid_S=hid_S, hid_T=hid_T, N_T=N_T, N_S=N_S, drop_path=0.1)
        else:
            print(f"Using pretrained base model {args.pretrained}")
            model = torch.load(args.pretrained)

        model.dec = Decoder(C_hid=hid_S, C_out=49, N_S=N_S, spatio_kernel=3)
        model.out_shape = (11,49,160,240)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = torch.nn.DataParallel(model)

    print(f"model has {sum(p.numel() for p in model.parameters())} params")

    # Load Data
    if args.unet_labels:
        dataset = LabeledDataset(args.train_data)
        unet_labeled_dataset = UnetLabeledDataset(args.train_data, args.unet_labels)
        dataset = torch.utils.data.ConcatDataset([dataset, unet_labeled_dataset])
    else:
        dataset = LabeledDataset(args.train_data)


    val_dataset = ValidationDataset(args.train_data)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Try saving model and deleting, so we don't train an epoch before failing
    save_model(model, args.output)
    os.remove(args.output)

    # Train!
    # Weight criterion so that empty class matters less!
    weights = torch.ones(49)
    weights[0] = 1 / 50 # This is just based on nothing lol

    criterion = torch.nn.CrossEntropyLoss(weight=weights) 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)

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

        val_iou = validate(model, val_dataset, device=device)
        print(f"IOU of validation set at epoch {i + 1}: {val_iou:.4f}")

        # Save model if it has the best iou
        if val_iou > best_iou:
            best_iou = val_iou
            save_model(model, args.output)

    print(train_loss)
    save_model(model, args.output)


if __name__ == "__main__":
    main()