#!/usr/bin/env python
# coding: utf-8

# Local imports
from data import LabeledDataset, ValidationDataset
from validate import validate
from vidtoseg.parallel2dconv import Parallel2DResNet
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


def train_segmentation(dataloader, model, criterion, optimizer, device, epoch, target_frames=(11, 22)):
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
        label_masks = labels[:,target_frames[0]:target_frames[1]].long()

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
    parser.add_argument('--no_prediction', action='store_true', help='Skip prediction (i.e. predict 11th frame segmention, rather than 22nd)')
    parser.add_argument('--use_model_predictor', action='store_true', help='Use models predictor, instead of initializing our own')
    parser.add_argument('--gsta', action='store_true', help='Use GSTA')

    # Parsing arguments
    args = parser.parse_args()

    print(f"Base training data folder: {args.train_data}")
    print(f"Pretrained model: {args.train_data}")
    print(f"Output file: {args.output}")
    
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"SGD learning rate: {args.lr}")

    target_frames = (0, 11) if args.no_prediction else (11, 22)
    print(f"Training segmentation for frames {target_frames}")


    # Define model
    if args.checkpoint:
        print(f"Initializing model from weights of {args.checkpoint}")

        model = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    else:
        if (args.pretrained is None):
            print(f"Initializing base model from random weights")
            base_model = SimSiamGSTA(Parallel2DResNet, 256 * 11)
        else:
            print(f"Using pretrained base model {args.pretrained}")
            base_model = torch.load(args.pretrained)

        encoder = base_model.backbone
        
        if args.no_prediction:
            print(f"Skipping prediction, using 1x1 conv for predictor")
            predictor = None

        elif args.use_model_predictor:
            print(f"Using pretrained predictor")
            predictor = base_model.predictor
        else:
            if (args.gsta):
                print(f"Using new instance of GSTA predictor")
                predictor = SimSiamGSTA(Parallel2DResNet, 256 * 11).predictor # This is dumb but whatever
            else:
                predictor = None

        model = UNetVidToSeg(encoder=encoder, predictor=predictor)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001) # TODO

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
        epoch_loss = train_segmentation(train_dataloader, model, criterion, optimizer, device, i + 1, target_frames=target_frames)
        train_loss.append(epoch_loss)

        val_iou = validate(model, val_dataset, device=device, target_frames=target_frames)
        print(f"IOU of validation set at epoch {i + 1}: {val_iou:.4f}")

        # Save model if it has the best iou
        if val_iou > best_iou:
            best_iou = val_iou
            save_model(model, args.output)



    print(train_loss)
    save_model(model, args.output)


if __name__ == "__main__":
    main()