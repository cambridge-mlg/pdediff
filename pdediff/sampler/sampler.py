r"""Samplers"""

from abc import ABC, abstractmethod

from torch import Size, Tensor
from typing import *
from pdediff.sde import VPSDE

class Sampler(ABC):
    def __init__(self, steps: int, corrections: int):
        super().__init__()
        self.steps = steps
        self.corrections = corrections

    @abstractmethod
    def sample(self, sde: VPSDE, shape: Size) -> Tensor:
        pass
