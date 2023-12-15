import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchmetrics import JaccardIndex
from data import LabeledDataset, ValidationDataset
from tqdm import tqdm

from unet import *
        
##################################################################################
# Load the data with Daniel's data.py

dataset = LabeledDataset('/scratch/py2050/Dataset_Student/')
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1)

val_dataset = ValidationDataset('/scratch/py2050/Dataset_Student/')
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=True, num_workers=1)


model_path = 'best_model_unet_50_py.pth' # your model path, remember to modify
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = UNetWithResnet50Encoder(n_classes=49)#.to(device)

# training preparations
# model = torch.load('best_model_unet_50_py.pth')
criterion = criterion = nn.CrossEntropyLoss()
optim = Adam(model.parameters(), lr=0.001,weight_decay=0.0001) 
num_epochs = 100
best_val_acc = 0 #Use IOU, baseline~=0.925097
jaccard = JaccardIndex(task="multiclass", num_classes=49).to(device)  

##################################################################################

for epoch in range(1, num_epochs+1):
    print("Training epoch: ", epoch)
    train_loss = 0   
    val_IoU_accuracy = 0 
    
    model.train()
    for data in tqdm(train_dataloader): 
        input, label = data
        input, label = input.to(device), label.to(device)
        input = input.reshape(-1,input.shape[2],input.shape[3],input.shape[4])
        label = label.reshape(-1,label.shape[2],label.shape[3])
        outputs = model(input)
        outputs = F.interpolate(outputs, size=(160, 240), mode='bilinear', align_corners=False)
        loss = criterion(outputs, label.long())
        loss.backward()   
        optim.step()                                               
        optim.zero_grad()                                           
        train_loss += loss.item()  
    train_loss /= (len(train_dataloader.dataset) * 22)   
    
    
    print("Validating epoch: ", epoch)
    model.eval()
    for idx, data in enumerate(tqdm(val_dataloader)):
        val_input, val_label = data
        input, label = val_input.to(device), val_label.to(device)
        input = input.reshape(-1,input.shape[2],input.shape[3],input.shape[4])
        label = label.reshape(-1,label.shape[2],label.shape[3])
        outputs = model(input)
        outputs = F.interpolate(outputs, size=(160, 240), mode='bilinear', align_corners=False)
        output = nn.LogSoftmax()(outputs)
        output = torch.argmax(output, dim=1)
        for i in range(label.shape[0]):
            jac = jaccard(output[i], label[i].to(device))
            val_IoU_accuracy += jac
        if idx == 49: # shorten val time, val on first 22*50*2 images
            break
    val_IoU_accuracy /= (22*50*2)  # remember to modify based on sample val length
    
    print("Epoch{}: Training Loss:{:.6f}; Val IoU: {:.6f}.\n".format(epoch, train_loss, val_IoU_accuracy))
    if best_val_acc < val_IoU_accuracy:
        best_val_acc = val_IoU_accuracy
        torch.save(model, model_path)  
        print('Best model saved to: ', model_path)