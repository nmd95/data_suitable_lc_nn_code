import torch
import torch.nn as nn
import models
import torch.backends.cudnn as cudnn

def get_model_and_criterion(baseline_params: dict):
    try:
        model_class = getattr(models, baseline_params["model_name"])
        model = model_class(**baseline_params["model_parameters"])
        criterion_class = getattr(nn, baseline_params["criterion_name"])
        criterion = criterion_class()
    except AttributeError as err:
        raise ValueError("Invalid model or criterion") from err

    return model, criterion

def get_optimizer_and_scheduler(model, baseline_params: dict):
    try:
        optimizer_class = getattr(torch.optim, baseline_params["optimizer_name"])
        optimizer = optimizer_class(model.parameters(), **baseline_params["optimizer_parameters"])

        scheduler =  baseline_params.get("scheduler_name")
        if scheduler:
            scheduler_class = getattr(torch.optim.lr_scheduler, baseline_params["scheduler_name"])
            scheduler = scheduler_class(optimizer, **baseline_params["scheduler_params"])

    except (AttributeError, KeyError) as err:
        raise ValueError("Invalid optimizer or scheduler") from err

    return optimizer, scheduler


def prepare_model(model, baseline_params: dict):
    if (baseline_params["multiple_gpu"]) and (baseline_params["model_name"] != "S4Model"):
        model = nn.DataParallel(model, device_ids=baseline_params["device_ids"])
    model.to(baseline_params["device"])
    if baseline_params["device"] != 'cpu':
        cudnn.benchmark = True
    return model

def make_layer(layer: str, *args) -> nn.Module:
    assert isinstance(layer, str)
    try:
        nn_layer = getattr(nn, layer)
    except AttributeError as err:
        raise ValueError(f'Unable to create layer {layer}') from err

    return nn_layer(*args)