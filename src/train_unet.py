import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchmetrics import JaccardIndex

import argparse
import time

from unet.unet import *
from data import UnlabeledDataset, LabeledDataset, ValidationDataset

parser = argparse.ArgumentParser(description="Process training data parameters.")
parser.add_argument('--train_data', type=str, required=True, help='Path to the training data folder')
parser.add_argument('--output', type=str, default="best_model.pkl", help='Path to the output model')

args = parser.parse_args()

# Load the data with Daniel's data.py
dataset = LabeledDataset(args.train_data)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=True, num_workers=2)

val_dataset = ValidationDataset(args.train_data)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=3, shuffle=False, num_workers=2)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = UNetWithResnet50Encoder(n_classes=49).to(device)
criterion = nn.CrossEntropyLoss()
# parameter chosen from YOLOv8 default value.
optim = Adam(model.parameters(), lr=0.002,weight_decay=0.0005)

num_epochs = 40 
best_train_acc = 1
best_val_acc = 0  # Use IOU 
jaccard = JaccardIndex(task="multiclass", num_classes=49).to(device)  
model_path = args.output # your model path, remember to modify

for epoch in range(1, num_epochs+1):
    print("Training epoch: ", epoch)
    train_loss = 0   
    val_IoU_accuracy = 0 

    start_time = time.time()
    num_minutes = 0

    model.train()
    for i, (input, label) in enumerate(train_dataloader): 
        input, label=input.to(device), label.to(device)
        input = input.reshape(-1,input.shape[2],input.shape[3],input.shape[4])
        label = label.reshape(-1,label.shape[2],label.shape[3])
        outputs = model(input)
        outputs = F.interpolate(outputs, size=(160, 240), mode='bilinear', align_corners=False)
        loss = criterion(outputs,label.long())
        loss.backward()   
        optim.step()                                               
        optim.zero_grad()                                           
        train_loss += loss.item()  

        if ((time.time() - start_time) // 60 > num_minutes):
            num_minutes = (time.time() - start_time) // 60
            print(f"After {num_minutes} minutes, finished training batch {i + 1} of {len(train_dataloader)}")

    train_loss /= len(train_dataloader.dataset)  

    # Timing                    
    print(f"Training took {(time.time() - start_time):2f} s")
    start_time = time.time()

    model.eval()
    for val_input, val_label in val_dataloader:
        input, label = val_input.to(device), val_label.to(device)
        input = input.reshape(-1,input.shape[2],input.shape[3],input.shape[4])
        label = label.reshape(-1,label.shape[2],label.shape[3])
        outputs = model(input)
        outputs = F.interpolate(outputs, size=(160, 240), mode='bilinear', align_corners=False)
        output = torch.argmax(output, dim=1)
        jac = jaccard(output, label.to(device))
        val_IoU_accuracy += jac

    val_IoU_accuracy /= len(val_dataloader.dataset)   
    print(f"Validation took {(time.time() - start_time):2f} s")
    print("Epoch {}: Training Loss:{:.6f} Val IoU: {:.6f}\n".format(epoch, train_loss, val_IoU_accuracy))

    if best_val_acc < val_IoU_accuracy:
        best_val_acc = val_IoU_accuracy
        torch.save(model, model_path)  
        print('Best model saved')