
def save_model(model, name):
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module, name)
    else:
        torch.save(model, name)