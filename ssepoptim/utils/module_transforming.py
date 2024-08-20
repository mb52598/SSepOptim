from typing import Callable, Type

import torch.nn as nn


def replace_module(
    module: nn.Module,
    module_to_replace: Type[nn.Module],
    replace_function: Callable[[nn.Module], nn.Module],
):
    for child_name, child_module in module.named_children():
        if type(child_module) is module_to_replace:
            new_module = replace_function(child_module)
            setattr(module, child_name, new_module)
        else:
            replace_module(child_module, module_to_replace, replace_function)


def for_module(
    module: nn.Module,
    search_module: Type[nn.Module],
    function: Callable[[nn.Module], None],
):
    for _, child_module in module.named_children():
        if type(child_module) is search_module:
            function(child_module)
        else:
            for_module(child_module, search_module, function)
