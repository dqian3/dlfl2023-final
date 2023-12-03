import torch
from torchmetrics import JaccardIndex

def validate(model, val_dataloader, device="cpu"):
    iou = JaccardIndex(task="multiclass", num_classes=49).to(device)

    total_iou = 0
    num_batches = 0

    with torch.no_grad():
        for batch in val_dataloader:
            x, target = batch   
            x = x.to(device)
            target = target.to(device)

            x = x[:, :11]
            # Transpose, since video resnet expects channels as first dim
            x = x.transpose(1, 2)
            masks = torch.argmax(model(x), dim=1)
    
            total_iou += iou(masks, target[:,21])
            num_batches += 1
        
    return total_iou / num_batches
