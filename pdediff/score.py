r"""Score modules"""

import torch
import torch.nn as nn

from torch import Tensor
from typing import *
from zuko.utils import broadcast

from pdediff.nn import TimeEmbedding
from pdediff.sde import VPSDE
from hydra.utils import instantiate
from pathlib import Path
from omegaconf import OmegaConf

import pdb


class ScoreNet(nn.Module):
    r"""Creates a simple score network made of residual blocks.

    Arguments:
        features: The number of features.
        embedding: The number of time embedding features.
    """
    def __init__(self, net, features: int, embedding: int = 16):
        super().__init__()

        self.embedding = TimeEmbedding(embedding)
        self.network = net(features + embedding, features)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t = self.embedding(t)
        x, t = broadcast(x, t, ignore=1)
        x = torch.cat((x, t), dim=-1)

        return self.network(x)


class ScoreUNet(nn.Module):
    r"""Creates a U-Net score network.

    Arguments:
        channels: The number of channels.
        embedding: The number of time embedding features.
    """

    def __init__(
            self,
            net,
            features: int,
            **kwargs):
        super().__init__()
        self.network = net(features, features)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        dims = self.network.spatial + 1
        y = x.reshape(-1, *x.shape[-dims:])
        return self.network(y, t).reshape(x.shape)


class AmortizedScoreUNet(nn.Module):
    r"""Creates a U-Net score network.

    Arguments:
        channels: The number of channels.
        embedding: The number of time embedding features.
    """

    def __init__(
            self,
            net: nn.Module,
            in_features: int = 64,
            out_features: int = 64,
            add_noise: bool = False,
            **kwargs):
        super().__init__()

        self.network = net(in_features + out_features, out_features)
        self.condition = None
        self.condition_dim = in_features
        self.mask_dim = self.condition_dim//2
        self.in_features = in_features + out_features
        self.out_features = out_features
        self.alpha = None
        self.add_noise = add_noise
        self.eta = None
       
    def set_condition(self, condition: Tensor):
        dims = self.network.spatial + 1
        assert condition.shape[-dims] == self.condition_dim
        self.condition = condition

    def set_zero_condition(self):
        dim = self.network.spatial
        self.condition = torch.zeros(1, self.condition_dim, *([1]*dim)).to(next(self.network.parameters()).device)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        dims = self.network.spatial + 1
        y = x.reshape(-1, *x.shape[-dims:])

        assert self.condition is not None

        condition = self.condition.expand(y.shape[0], -1, *y.shape[-self.network.spatial:])
        if self.add_noise:
            t = t.reshape(t.shape + (1,) * (self.network.spatial + 1))
            mask = condition[:, :self.mask_dim]
            condition_without_mask = torch.clone(condition[:, self.mask_dim:])
            mu = self.alpha(t).sqrt()
            sigma = (1 - mu ** 2 + self.eta ** 2).sqrt()
            condition_without_mask = mu * condition_without_mask + sigma * torch.randn_like(
                condition_without_mask).to(condition_without_mask.device)
            current_condition = torch.cat([mask, mask*condition_without_mask], dim=1)
        else:
            current_condition = condition

        y = torch.cat([current_condition, y], dim=1)
        new_shape = list(x.shape)
        new_shape[-dims] = self.out_features
        return self.network(y, t).reshape(new_shape)


class SpecialScoreUNet(ScoreUNet):
    r"""Creates a score U-Net with a forcing channel."""

    def __init__(
        self,
        net,
        features: int,
        size: int = 64, 
        **kwargs,
    ):
        super().__init__(net, features + 1, **kwargs)

        domain = 2 * torch.pi / size * (torch.arange(size) + 1 / 2)
        forcing = torch.sin(4 * domain).expand(1, size, size).clone()

        self.register_buffer("forcing", forcing)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x, f = broadcast(x, self.forcing, ignore=3)
        x = torch.cat((x, f), dim=-3)

        return super().forward(x, t)[..., :-1, :, :]
    

class AmortizedSpecialScoreUNet(AmortizedScoreUNet):
    r"""Creates a score U-Net with a forcing channel."""

    def __init__(
        self,
        net: nn.Module,
        in_features: int = 64,
        out_features: int = 64,
        add_noise: bool = False,
        size: int = 64,
        **kwargs):

        super().__init__(net=net, 
                         in_features=in_features, 
                         out_features=out_features + 1, 
                         add_noise=add_noise)

        domain = 2 * torch.pi / size * (torch.arange(size) + 1 / 2)
        forcing = torch.sin(4 * domain).expand(1, size, size).clone()

        self.register_buffer("forcing", forcing)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x, f = broadcast(x, self.forcing, ignore=3)
        x = torch.cat((x, f), dim=-3)

        return super().forward(x, t)[..., :-1, :, :]
    

class MCScoreWrapper(nn.Module):
    r"""Disguises a `ScoreUNet` as a score network for a Markov chain.
    
        Just a wrapper for our score network that is passed in the constructor.
        The forward just call the forward of the score network
    """

    def __init__(self, score: nn.Module):
        super().__init__()

        self.score = score

    def forward(
        self,
        x: Tensor,  # (B, L, C, H, W)
        t: Tensor,  # ()
    ) -> Tensor:
        return self.score(x.transpose(1, 2), t).transpose(1, 2)


class MCScoreNet(nn.Module):
    r"""Creates a score network for a Markov chain.

    Arguments:
        features: The number of features.
        order: The order of the Markov chain.
    """

    def __init__(self, kernel, order: int):
        super().__init__()

        self.order = order
        self.kernel = kernel
    
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        dims = self.kernel.network.spatial + 2
        num_channels = x.shape[-(dims - 1)]

        x = self.unfold(x, self.order)

        assert (
            x.shape[-(dims - 1)] == (2 * self.order + 1) * num_channels
        ), f"Dimensions {x.shape} are not consistent with the window size {2*self.order+1}"

        s = self.kernel(x, t)

        assert (
            s.shape[-(dims - 1)] == (2 * self.order + 1) * num_channels
        ), f"Dimensions {s.shape} are not consistent with the window size {2*self.order+1}"

        s = self.fold(s, self.order)
        return s

    # the tag is just compiling the function when it is first called during tracing
    @staticmethod
    @torch.jit.script_if_tracing
    def unfold(x: Tensor, order: int) -> Tensor:
        """
        This method take the batch of trajectories, and return all the psudo markov
        blanket described by Algorithm 2 in the paper.
        So it just create the following:
        - x_{1:2k+1}(t)
        - x_{i−k:i+k}(t) for i = k + 2 to L − k − 1
        - x_{L−2k:L}(t)

        These are all the input to our score network that are used to compute the approximate score.
        """

        x = x.unfold(1, 2 * order + 1, 1)
        x = x.movedim(-1, 2)
        x = x.flatten(2, 3)

        return x

    @staticmethod
    @torch.jit.script_if_tracing
    def fold(x: Tensor, order: int) -> Tensor:
        """
        Function that given all the scores computed in each markov blanket and
        compose the approximated score as described in Algorithm 2
        """
        x = x.unflatten(2, (2 * order + 1, -1))

        return torch.cat(
            (
                x[:, 0, :order],
                x[:, :, order],
                x[:, -1, -order:],
            ),
            dim=1,
        )

def make_score(
        score,
        net,
        window,
        spatial,
        condition_dim=0,
):
    # Partially create neural network
    net = instantiate(net, _partial_=True)

    # Create score wrapper to combine with time context
    score = instantiate(
        score,
        net=net,
        features=spatial * window,
        in_features=spatial*condition_dim,
        out_features=spatial*window,
    )

    # Construct full score network from markov blanket scores
    return MCScoreNet(
        kernel=score,
        order=window // 2,
    )


def load_score(file: Path, device: str = "cpu", **kwargs) -> nn.Module:
    state = torch.load(file, map_location=device)
    cfg = OmegaConf.load(file.parent.parent.joinpath(".hydra/config.yaml"))
    cfg.update(kwargs)

    if not cfg.amortized:
        condition_dim = 0
    else:
        condition_dim = cfg.window*2

    score = make_score(
        score=cfg.score,
        net=cfg.net,
        window=cfg.window,
        spatial=cfg.data.spatial,
        condition_dim=condition_dim,
    )
    score.load_state_dict(state)
    return score
