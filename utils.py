import torch
from torch._six import inf
from typing import Union, Iterable

import os


import torch
from asteroid.losses import PITLossWrapper


class newPITLossWrapper(PITLossWrapper):
    def __init__(self, loss_func, pit_from="pw_mtx", perm_reduce=None, reduction=True):
        super(newPITLossWrapper, self).__init__(loss_func, pit_from, perm_reduce)
        self.reduction = reduction

    def forward(self, est_targets, targets, return_est=False, reduce_kwargs=None, **kwargs):
        n_src = targets.shape[1]
        assert n_src < 10, f"Expected source axis along dim 1, found {n_src}"
        if self.pit_from == "pw_mtx":
            # Loss function already returns pairwise losses
            pw_losses = self.loss_func(est_targets, targets, **kwargs)
        elif self.pit_from == "pw_pt":
            # Compute pairwise losses with a for loop.
            pw_losses = self.get_pw_losses(self.loss_func, est_targets, targets, **kwargs)
        elif self.pit_from == "perm_avg":
            # Cannot get pairwise losses from this type of loss.
            # Find best permutation directly.
            min_loss, batch_indices = self.best_perm_from_perm_avg_loss(
                self.loss_func, est_targets, targets, **kwargs
            )
            # Take the mean over the batch
            mean_loss = torch.mean(min_loss)
            if not return_est:
                return mean_loss
            reordered = self.reorder_source(est_targets, batch_indices)
            return mean_loss, reordered
        else:
            return

        assert pw_losses.ndim == 3, (
            "Something went wrong with the loss " "function, please read the docs."
        )
        assert pw_losses.shape[0] == targets.shape[0], "PIT loss needs same batch dim as input"

        reduce_kwargs = reduce_kwargs if reduce_kwargs is not None else dict()
        min_loss, batch_indices = self.find_best_perm(
            pw_losses, perm_reduce=self.perm_reduce, **reduce_kwargs
        )
        if self.reduction:
            mean_loss = torch.mean(min_loss)
        else:
            mean_loss = min_loss
        if not return_est:
            return mean_loss
        reordered = self.reorder_source(est_targets, batch_indices)
        return mean_loss, reordered


def makedir(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs, exist_ok=True)
        print(dirs)


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


'''
    gradient clips are based on pytorch codes.
'''
_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]

def norm(x, norm_type):
    return torch.sqrt(x.pow(2).sum() + 1e-6)

def clip_grad_norm_(
        parameters: _tensor_or_tensors, max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        norms = [p.grad.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = norm(torch.stack([norm(p.grad.detach().type(torch.float64), norm_type).to(device) for p in parameters]), norm_type)
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0).type(torch.float32)
    for p in parameters:
        p.grad.detach().mul_(clip_coef_clamped.to(p.grad.device))
    return total_norm

