# Ported from https://github.com/CasvandenBogaard/VBMF and altered to work with pytorch
# Implementation based on: Nakajima, Shinichi, et al. "Global analytic solution of fully-observed variational Bayesian matrix factorization." The Journal of Machine Learning Research 14.1 (2013): 1-37.

import math
from typing import Callable

import torch
import torch.optim as optim


def _minimize_scalar_bounded(
    func: Callable[[torch.Tensor], torch.Tensor],
    bounds: tuple[float, float],
    max_iters: int = 500,
    lr: float = 0.00001,
    delta: float = 0.0001,
) -> float:
    x = torch.tensor((bounds[1] + bounds[0]) / 2.0, requires_grad=True)
    prev_x = x.detach().clone()
    opt = optim.SGD([x], lr=lr)
    for _ in range(max_iters):
        out = func(x)
        opt.zero_grad(set_to_none=True)
        out.backward()
        opt.step()
        with torch.no_grad():
            x.clamp_(bounds[0], bounds[1])
        if torch.abs(x - prev_x) < delta:
            break
        prev_x = x.detach().clone()
    return x.item()


def tau(x: torch.Tensor, alpha: float):
    return 0.5 * (x - (1 + alpha) + torch.sqrt((x - (1 + alpha)) ** 2 - 4 * alpha))


def EVBsigma2(
    sigma2: torch.Tensor, L: int, M: int, s: torch.Tensor, residual: float, xubar: float
):
    H = len(s)

    alpha = L / M
    x = s.detach() ** 2 / (M * sigma2)

    z1 = x[x > xubar]
    z2 = x[x <= xubar]
    tau_z1 = tau(z1, alpha)

    term1 = torch.sum(z2 - torch.log(z2))
    term2 = torch.sum(z1 - tau_z1)
    term3 = torch.sum(torch.log(torch.divide(tau_z1 + 1, z1)))
    term4 = alpha * torch.sum(torch.log(tau_z1 / alpha + 1))

    obj = (
        term1
        + term2
        + term3
        + term4
        + residual / (M * sigma2)
        + (L - H) * torch.log(sigma2)
    )

    return obj


def EVBMF(Y: torch.Tensor, sigma2: float | None = None, H: int | None = None):
    L, M = Y.shape  # has to be L<=M

    if H is None:
        H = L

    alpha = L / M
    tauubar = 2.5129 * math.sqrt(alpha)

    # SVD of the input matrix, max rank of H
    U, s, V = torch.svd(Y)
    U = U[:, :H]
    s = s[:H]
    V = V[:H].T

    # Calculate residual
    residual = 0.0
    if H < L:
        residual = torch.sum(Y**2).item() - torch.sum(s**2).item()

    # Estimation of the variance when sigma2 is unspecified
    if sigma2 is None:
        xubar = (1 + tauubar) * (1 + alpha / tauubar)
        eH_ub = int(min(math.ceil(L / (1 + alpha)) - 1, H)) - 1
        upper_bound = (torch.sum(s**2).item() + residual) / (L * M)
        lower_bound = max(
            s[eH_ub + 1].item() ** 2 / (M * xubar),
            torch.mean(s[eH_ub + 1 :] ** 2).item() / M,
        )

        scale = 1.0  # /lower_bound
        s = s * math.sqrt(scale)
        residual = residual * scale
        lower_bound = lower_bound * scale
        upper_bound = upper_bound * scale

        sigma2 = _minimize_scalar_bounded(
            lambda x: EVBsigma2(x, L, M, s, residual, xubar),
            bounds=(lower_bound, upper_bound),
        )

    # Threshold gamma term
    threshold = math.sqrt(M * sigma2 * (1 + tauubar) * (1 + alpha / tauubar))
    pos = torch.sum(s > threshold).item()

    # Formula (15) from [2]
    d = torch.multiply(
        s[:pos] / 2,
        1
        - (((L + M) * sigma2) / (s[:pos] ** 2))
        + torch.sqrt(
            (1 - (((L + M) * sigma2) / (s[:pos] ** 2))) ** 2
            - 4 * L * M * sigma2**2 / s[:pos] ** 4
        ),
    )

    return U[:, :pos], torch.diag(d), V[:, :pos]
