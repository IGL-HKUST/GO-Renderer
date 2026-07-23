"""Parameter-placement helper adapted from ComfyUI-MochiWrapper."""

from torch import nn


def replace_parameters_by_name(module, name_keywords, device):
    """Move selected direct parameters to a device without re-registering them."""

    for name, param in list(module.named_parameters(recurse=False)):
        if any(keyword in name for keyword in name_keywords):
            if isinstance(param, nn.Parameter):
                tensor = param.data
                delattr(module, name)
                setattr(module, name, tensor.to(device=device))
    for child_name, child_module in module.named_children():
        replace_parameters_by_name(child_module, name_keywords, device)
