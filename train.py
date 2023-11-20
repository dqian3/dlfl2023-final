#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torchvision.models.video import r2plus1d_18
from simsiam import SimSiam

from data import VideoDataset

import torch


# In[2]:


backbone = r2plus1d_18
model = SimSiam(backbone)


# In[3]:


BATCH_SIZE=5

dataset = VideoDataset("Dataset_Student/unlabeled", 13000, idx_offset=2000, has_label=False)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


# In[9]:


from tqdm import tqdm

def train(dataloader, model, criterion, optimizer, epoch):
    for batch in tqdm(dataloader):
        data = batch
    
        # Split video frames into 
        x1, x2 = data[:, :11], data[:, 11:]
        # Transpose, since video resnet expects channels as first dim
        x1 = x1.transpose(1, 2)
        x2 = x2.transpose(1, 2)
    
        p1, p2, h1, h2 = model(x1, x2)

        loss = -(criterion(p1, h2).mean() + criterion(p2, h1).mean()) * 0.5

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"epoch {epoch} loss: {loss}")


# In[10]:


# simsiam criterion
criterion = torch.nn.CosineSimilarity(dim=1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


# In[16]:


NUM_EPOCHS = 10

for i in tqdm(range(NUM_EPOCHS)):
    train(train_dataloader, model, criterion, optimizer, i + 1)


# In[15]:


torch.save(model, "simsiam.pkl")


# In[ ]:




