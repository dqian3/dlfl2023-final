import torch
from torchmetrics import JaccardIndex

import time


# Sample None is whole dataset, int determines size of sample
def validate(model, dataset, device="cpu", batch_size=2, sample=None, channels_first=False):
   
    start_time = time.time()
    print(f"Validating with sample={sample}")

    iou = JaccardIndex(task="multiclass", num_classes=49)

    if (sample):
        sampler = torch.utils.data.RandomSampler(dataset, num_samples=sample)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2, sampler=sampler)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2)

    masks = []
    labels = []

    with torch.no_grad():
        for (i, batch) in enumerate(dataloader):
            data, target = batch
            data = data.to(device)
            target = target.to(device)

            data = data[:,:11]
            if channels_first:
                data = data.transpose(1, 2)


            # Split video frames into first half
            mask = model(data)
            if not channels_first:
                mask = mask.transpose(1, 2)

            mask = torch.argmax(mask, dim=1)
            masks.append(mask[:,10].to("cpu"))
            labels.append(target[:,21].to("cpu"))
            
            if (i + 1) % 100 == 0:
                print(f"After {time.time() - start_time:.2f} seconds finished training batch {i + 1} of {len(dataloader)}")

            del data

        masks = torch.stack(masks) # 1k x 160 x 240
        labels = torch.stack(labels) # 1k x 160 x 240

        print(f"Took {(time.time() - start_time):2f} s")

        return iou(masks, labels), masks