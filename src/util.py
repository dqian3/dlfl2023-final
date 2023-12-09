import torch

def save_model(model, name):
    print(f"Saving model to {name}")
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module, name)
    else:
        torch.save(model, name)