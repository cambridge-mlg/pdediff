r"""Fast Sampling of Diffusion Models with Exponential Integrator"""

from abc import ABC, abstractmethod
import sys

sys.path.append("../")
import torch
from torch import Size, Tensor
from typing import *
from tqdm import tqdm

from pdediff.sde import VPSDE
from pdediff.sampler.sampler import Sampler
from pdediff.sampler.helper import *


class ExpInt(Sampler):
    def __init__(
            self,
            steps: int,
            corrections: int,
            tau: float,
            skip_type: str = "linear",
            correcting_x0_fn = None,
            thresholding_max_val: float = 1.,
            dynamic_thresholding_ratio: float = 0.995,
            denoise_to_zero = False,
    ):
        super().__init__(steps, corrections)
        self.tau = tau
        self.skip_type = skip_type
        if skip_type == "t":
            self.sampling_eps = 1e-3
            self.sampling_T = 1 - 1e-3
        elif skip_type == "linear":
            self.sampling_eps = 1e-3
            self.sampling_T = 1
        elif skip_type == "logSNR":
            self.sampling_eps = 1e-3
            self.sampling_T = 1
        else:
            raise ValueError("Unsupported skip_type {}, need to be 'logSNR' or 'linear' or 't'".format(skip_type))
        if correcting_x0_fn == "dynamic_thresholding":
            self.correcting_x0_fn = self.dynamic_thresholding_fn
        elif correcting_x0_fn == "None":
            self.correcting_x0_fn = None
        else:
            self.correcting_x0_fn = correcting_x0_fn
        self.thresholding_max_val = thresholding_max_val
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio 
        self.denoise_to_zero = denoise_to_zero

    def get_time_steps(self, sde, skip_type, t_T, t_0, N):
        """Compute the intermediate time steps for sampling.

        Args:
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'linear': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 't': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            N: A `int`. The total number of the spacing of the time steps.
            device: A torch device.
        Returns:
            A pytorch tensor of the time steps, with the shape (N + 1,).
        """
        if skip_type == 'logSNR':
            lambda_T = sde.marginal_lambda(torch.tensor(t_T).to(sde.device))
            lambda_0 = sde.marginal_lambda(torch.tensor(t_0).to(sde.device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(sde.device)
            return sde.inverse_lambda(logSNR_steps)
        elif skip_type == 'linear':
            return torch.linspace(t_T, t_0, N + 1).to(sde.device)
        elif skip_type == 't':
            t_order = 2
            t = torch.linspace(t_T ** (1. / t_order), t_0 ** (1. / t_order), N + 1).pow(t_order).to(sde.device)
            return t
        else:
            raise ValueError("Unsupported skip_type {}, need to be 'logSNR' or 'linear' or 't'".format(skip_type))

    def dynamic_thresholding_fn(self, x0, t):
        """
        The dynamic thresholding method.
        """
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(torch.maximum(s, self.thresholding_max_val * torch.ones_like(s).to(s.device)), dims)
        x0 = torch.clamp(x0, -s, s) / s
        return x0
    
    def denoise_to_zero_fn(self, sde: VPSDE, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization.
        """
        x0 = sde.data_prediction_fn(x, s)
        if self.correcting_x0_fn is not None:
            return self.correcting_x0_fn(x0, s)
        else:
            return x0
        
    def sample(
            self,
            sde: VPSDE,
            shape: Size = (),
    ) -> Tensor:
        r"""Samples from :math:`p(x(0))`.

        Arguments:
            shape: The batch shape.
            steps: The number of discrete time steps.
            corrections: The number of Langevin corrections per time steps.
            tau: The amplitude of Langevin steps.
        """
        x = torch.randn(shape + sde.shape).to(sde.device)
        x = x.reshape(-1, *sde.shape)

        # time = get_rev_ts_th(sde, self.steps, self.ts_order, sampling_eps = self.sampling_eps, sampling_T = self.sampling_T, ts_phase = self.ts_phase)
        time = self.get_time_steps(sde, skip_type=self.skip_type, t_T=self.sampling_T, t_0=self.sampling_eps,
                                   N=self.steps)
        dts = time[:-1] - time[1:]

        with torch.no_grad():
            for dt, t in tqdm(zip(dts, (time[:-1])), total=len(dts)):
                # for t in time[:-1]:
                # Predictor
                r = sde.mu(t - dt) / sde.mu(t)
                noise_pred = sde.noise_prediction_fn(x, t)
                x = r * x + (sde.sigma(t - dt) - r * sde.sigma(t)) * noise_pred
                # Corrector
                for _ in range(self.corrections):
                    eps = torch.randn_like(x)
                    # NOTE: since we are parametrizing eps(x(t),t) = - sigma(t) * score(x(t),t)
                    # if we need to use score(x(t),t) we have to always compute
                    # score(x(t),t) = - eps(x(t),t) / sigma(t)
                    noise_pred = sde.noise_prediction_fn(x, t - dt)
                    s = -noise_pred / sde.sigma(t - dt)
                    delta = self.tau / s.square().mean(dim=sde.dims, keepdim=True)

                    x = x + delta * s + torch.sqrt(2 * delta) * eps

            if self.denoise_to_zero:
                t = torch.ones((1,)).to(sde.device) * self.sampling_eps
                x = self.denoise_to_zero_fn(sde, x, t)
        return x.reshape(shape + sde.shape)