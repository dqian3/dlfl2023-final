import torch
from torchmetrics import JaccardIndex

import time


# Sample None is whole dataset, int determines size of sample
def validate(model, dataset, device="cpu", batch_size=2, sample=None):
   
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

            # Split video frames into first half
            masks.append(model(data).to("cpu").detach()[:,10])
            labels.append(target[:,21].to("cpu"))
            
            if (i % 10 == 9):
                print(f"After {time.time() - start_time:.2f} seconds finished training batch {i + 1} of {len(dataloader)}")

            del data

        masks = torch.stack(masks)
        labels = torch.stack(labels)

        print(f"Took {(time.time() - start_time):2f} s")

        return iou(masks, labels), masks
