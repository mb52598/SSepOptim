from typing import Callable, Type

import torch.nn as nn


def replace_module(
    model: nn.Module,
    module_to_replace: Type[nn.Module],
    replace_function: Callable[[nn.Module], nn.Module],
):
    for child_name, child_module in model.named_children():
        if type(child_module) is module_to_replace:
            new_module = replace_function(child_module)
            setattr(model, child_name, new_module)
        else:
            replace_module(child_module, module_to_replace, replace_function)
