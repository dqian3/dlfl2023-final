import torch
from torchmetrics import JaccardIndex

import time


# Sample None is whole dataset, int determines size of sample
@torch.no_grad()
def validate(model, dataset, device="cpu", batch_size=2, sample=None):
   
    start_time = time.time()
    print(f"Validating with sample={sample}")

    iou = JaccardIndex(task="multiclass", num_classes=49).to(device)

    if (sample):
        sampler = torch.utils.data.RandomSampler(dataset, num_samples=sample)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2, sampler=sampler)
    else:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2)

    masks = []
    labels = []
    for batch in data_loader:
        data, target = batch
        data = data.to(device)
        target = target.to(device)

        data = data[:,:11]

        # Split video frames into first half
        masks.append(model(data).detach())
        labels.append(target)

        del data

    masks = torch.stack(masks)
    labels = torch.stack(labels)

    print(f"Took {(time.time() - start_time):2f} s")

    return iou(masks, labels), masks
