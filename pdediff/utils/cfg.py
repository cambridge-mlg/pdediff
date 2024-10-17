import math
import functools
from typing import List

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf


class NoneHydra:
    def __init__(self, *args, **kwargs):
        pass

    def __bool__(self):
        return False


def my_glob_impl(pattern: str, _root_: DictConfig) -> List[str]:
    """
    A simple glob implementation, takes a `pattern` with wildcard `*` at the
    end.  The return value is a set of full keys in the config which match the
    `pattern`.
    """
    assert pattern.endswith(".*")
    pattern = pattern.removesuffix(".*")
    node = OmegaConf.select(_root_, key=pattern)
    if node is None:
        return []
    if not isinstance(node, DictConfig):
        raise NotImplementedError
    else:
        return [f"{pattern}.{key}" for key in node.keys()]


OmegaConf.register_new_resolver(name="my_glob", resolver=my_glob_impl)


# Define useful resolver for hydra config
OmegaConf.register_new_resolver("int", lambda x: int(x), replace=True)
OmegaConf.register_new_resolver("eval", lambda x: eval(x), replace=True)
OmegaConf.register_new_resolver("str", lambda x: str(x), replace=True)
OmegaConf.register_new_resolver("prod", lambda x: np.prod(x), replace=True)
OmegaConf.register_new_resolver(
    "where", lambda condition, x, y: x if condition else y, replace=True
)
OmegaConf.register_new_resolver("isequal", lambda x, y: x == y, replace=True)
OmegaConf.register_new_resolver("pi", lambda x: x * math.pi, replace=True)
OmegaConf.register_new_resolver("min", min, replace=True)
OmegaConf.register_new_resolver("max", max, replace=True)
OmegaConf.register_new_resolver("sqrt", math.sqrt, replace=True)
OmegaConf.register_new_resolver(
    name="concat", resolver=lambda *lists: [elt for l in lists for elt in l]
)


def partialclass(cls, *args, **kwds):
    """Return a class instance with partial __init__
    Input:
        cls [str]: class to instantiate
    """
    cls = hydra.utils.get_class(cls)

    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)

    return NewCls


def partialfunction(func, *args, **kwargs):
    return functools.partial(func, *args, **kwargs)
