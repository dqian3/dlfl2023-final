import torch
from torchmetrics import JaccardIndex

# Sample None is whole dataset, int determines size of sample
def validate(model, dataset, device="cpu", batch_size=2, target_frames=(11,22), sample=None):
    iou = JaccardIndex(task="multiclass", num_classes=49).to(device)

    if (sample):
        sampler = torch.utils.data.RandomSampler(dataset, num_samples=sample)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2, sampler=sampler)
    else:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2)

    total_iou = 0
    num_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            x, target = batch   
            x = x.to(device)
            target = target.to(device)

            x = x[:, :11]
            # Transpose, since video resnet expects channels as first dim
            x = x.transpose(1, 2)
            masks = torch.argmax(model(x), dim=1)
    
            total_iou += iou(masks, target[:,target_frames])
            num_batches += 1
        
    return total_iou / num_batches
