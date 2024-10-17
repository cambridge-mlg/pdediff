from torch import nn


ACTIVATIONS = {
    "ReLU": nn.ReLU,
    "SiLU": nn.SiLU,
    "GELU": nn.GELU,
    "SELU": nn.SELU,
    "ELU": nn.ELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}