#!/usr/bin/env python
# coding: utf-8

# Local imports
from data import LabeledDataset, ValidationDataset
from seg_model import SegmentationModel
from validate import validate
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
    parser.add_argument('--pretrained', type=str, default="simsiam.pkl", help='Path to pretrained simsiam network')
    parser.add_argument('--checkpoint', default=None, help='Path to the model checkpoint to continue training off of')

    # Hyperparam args
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

    # Other args
    parser.add_argument('--use_tqdm', action='store_true', help='Use tqdm in output')
    parser.add_argument('--skip_predictor', action='store_false', help='Skip prediction (i.e. predict 11th frame segmention, rather than 22nd)')

    # Parsing arguments
    args = parser.parse_args()

    print(f"Base training data folder: {args.train_data}")
    print(f"Pretrained model: {args.train_data}")
    print(f"Output file: {args.output}")
    
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"SGD learning rate: {args.lr}")

    target_frame = 10 if args.skip_predictor else 21
    print(f"Training segmentation for frame {target_frame}")

    # Define model
    if args.checkpoint:
        model = torch.load(args.checkpoint, map_location=torch.device('cpu'))
        print(f"Initializing model from weights of {args.checkpoint}")

        if (model.use_predictor == args.skip_predictor):
            if (model.use_predictor):
                print(f"Incompatibble training param: model uses predictor layer, but target is 11th frame")
            else:
                print(f"Incompatibble training param: model skips predictor layer, but target is 22nd frame")

    else:
        pretrained = torch.load(args.pretrained)
        model = SegmentationModel(pretrained, finetune=True, use_predictor=(not args.skip_predictor))
        print(f"Initializing model from random weights")

    print(f"model has {sum(p.numel() for p in model.parameters())} params")

    # Load Data
    dataset = LabeledDataset(args.train_data)
    val_dataset = ValidationDataset(args.train_data)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Try saving model and deleting, so we don't train an epoch before failing
    torch.save(model, args.output)
    os.remove(args.output)

    # Train!
    # Weight criterion so that empty class matters less!
    weights = torch.ones(49)
    weights[0] = 1 / 50 # rough estimate of number of pixels that are background

    criterion = torch.nn.CrossEntropyLoss(weight=weights) 
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr) # TODO

    iterator = range(args.num_epochs)
    if (args.use_tqdm): 
        iterator = tqdm(iterator)

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        device = torch.device("cuda:0")
        print("Using cuda!")
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = torch.nn.DataParallel(model)

    else:
        device = torch.device("cpu")
        print("Using CPU!")


    train_loss = []
    for i in iterator:
        epoch_loss = train_segmentation(train_dataloader, model, criterion, optimizer, device, i + 1, target_frame=target_frame)
        train_loss.append(epoch_loss)

        val_iou = validate(model, val_dataloader, device=device, target_frame=target_frame)
        print(f"IOU of validation set at epoch {i + 1}: {val_iou:.4f}")

        # Save model every 10 epochs, in case our job dies lol
        if i % 10 == 9:
            file, ext = os.path.splitext(args.output)
            torch.save(model, file + f"_{i + 1}" + ext)

    print(train_loss)
    torch.save(model.state_dict(), args.output)


if __name__ == "__main__":
    main()