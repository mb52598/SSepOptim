import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


class WeightChangeUnstructured(prune.BasePruningMethod):
    r"""Prune (currently unpruned) units in a tensor by their gradients.

    Args:
        name (str): parameter name within ``module`` on which pruning
            will act.
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
    """

    PRUNING_TYPE = "unstructured"

    def __init__(self, amount: float | int):
        # Check range of validity of pruning amount
        prune._validate_pruning_amount_init(amount)
        self.amount = amount

    def compute_mask(self, t: torch.Tensor, default_mask: torch.Tensor):
        # Check that the tensor supports gradients
        if t.requires_grad and t.grad is not None:
            t = torch.abs(t.grad)
        else:
            raise RuntimeError(
                "WeightChangeUnstructured method is only possible for gradient tensors"
            )
        # Check that the amount of units to prune is not > than the number of
        # parameters in t
        tensor_size = t.nelement()
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = prune._compute_nparams_toprune(self.amount, tensor_size)
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        prune._validate_pruning_amount(nparams_toprune, tensor_size)
        # Clone the mask we are adding onto
        mask = default_mask.clone(memory_format=torch.contiguous_format)
        # If there are elements to prune, select `nparams_toprune` lowest gradients
        if nparams_toprune != 0:
            topk = torch.topk(
                t.view(-1), k=nparams_toprune, largest=False, sorted=False
            )
            mask.view(-1)[topk.indices] = 0
        # Return mask
        return mask


def weight_change_unstructured(
    module: nn.Module,
    name: str,
    amount: int | float,
    importance_scores: torch.Tensor | None = None,
):
    WeightChangeUnstructured.apply(
        module, name, amount=amount, importance_scores=importance_scores
    )
    return module


def _compute_sum(t: torch.Tensor, dim: int):
    # dims = all axes, except for the one identified by `dim`
    dims = list(range(t.dim()))
    # convert negative indexing
    if dim < 0:
        dim = dims[dim]
    dims.remove(dim)

    return torch.sum(t, dim=dims)


class WeightChangeStructured(prune.BasePruningMethod):
    r"""Prune entire (currently unpruned) channels in a tensor by their gradients.

    Args:
        amount (int or float): quantity of channels to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
        dim (int, optional): index of the dim along which we define
            channels to prune. Default: -1.
    """

    PRUNING_TYPE = "structured"

    def __init__(self, amount: int | float, dim: int = -1):
        # Check range of validity of amount
        prune._validate_pruning_amount_init(amount)
        self.amount = amount
        self.dim = dim

    def compute_mask(self, t: torch.Tensor, default_mask: torch.Tensor):
        # Check that the tensor supports gradients
        if t.requires_grad and t.grad is not None:
            t = torch.abs(t.grad)
        else:
            raise RuntimeError(
                "WeightChangeStructured method is only possible for gradient tensors"
            )
        # Check that tensor has structure (i.e. more than 1 dimension) such
        # that the concept of "channels" makes sense
        prune._validate_structured_pruning(t)
        # Check that self.dim is a valid dim to index t, else raise IndexError
        prune._validate_pruning_dim(t, self.dim)

        # Check that the amount of channels to prune is not > than the number of
        # channels in t along the dim to prune
        tensor_size = t.shape[self.dim]
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = prune._compute_nparams_toprune(self.amount, tensor_size)
        nparams_tokeep = tensor_size - nparams_toprune
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        prune._validate_pruning_amount(nparams_toprune, tensor_size)

        # Structured pruning prunes entire channels so we need to know the
        # gradient sum along each channel to then find the topk based on this
        # metric
        summation = _compute_sum(t, self.dim)
        # largest=True --> top k; largest=False --> bottom k
        # Keep the largest k channels along dim=self.dim
        topk = torch.topk(summation, k=nparams_tokeep, largest=True)
        # topk will have .indices and .values

        # Compute binary mask by initializing it to all 0s and then filling in
        # 1s wherever topk.indices indicates, along self.dim.
        # mask has the same shape as tensor t
        def make_mask(t: torch.Tensor, dim: int, indices: torch.Tensor):
            # init mask to 0
            mask = torch.zeros_like(t)
            # e.g.: slc = [None, None, None], if len(t.shape) = 3
            slc = [slice(None)] * len(t.shape)
            # replace a None at position=dim with indices
            # e.g.: slc = [None, None, [0, 2, 3]] if dim=2 & indices=[0,2,3]
            slc[dim] = indices
            # use slc to slice mask and replace all its entries with 1s
            # e.g.: mask[:, :, [0, 2, 3]] = 1
            mask[slc] = 1
            return mask

        if nparams_toprune == 0:
            mask = default_mask
        else:
            mask = make_mask(t, self.dim, topk.indices)
            mask *= default_mask.to(dtype=mask.dtype)

        return mask


def weight_change_structured(
    module: nn.Module,
    name: str,
    amount: int | float,
    dim: int,
    importance_scores: torch.Tensor | None = None,
):
    WeightChangeStructured.apply(
        module, name, amount=amount, dim=dim, importance_scores=importance_scores
    )
    return module
