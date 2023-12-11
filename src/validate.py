import torch
from torchmetrics import JaccardIndex

# Sample None is whole dataset, int determines size of sample
def validate(model, dataset, device="cpu", batch_size=2, sample=None):
    iou = JaccardIndex(task="multiclass", num_classes=49).to(device)

    if (sample):
        sampler = torch.utils.data.RandomSampler(dataset, num_samples=sample)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2, sampler=sampler)
    else:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2)

    total_iou = 0
    num_batches = 0

    masks = []
    labels = []
    for batch in data_loader:
        data, target = batch
        data = data.to(device)
        target = target.to(device)

        # Split video frames into first half
        masks.append(model(data))
        labels.append(target)

    masks = torch.stack(masks)
    print(masks.shape)
    labels = torch.stack(labels)
    print(labels.shape)

        
    return total_iou / num_batches
