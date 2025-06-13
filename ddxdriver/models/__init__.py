import importlib

from .base import Model


def init_model(class_name: str, *args, **kwargs) -> Model:
    tokens = class_name.rsplit(".", maxsplit=1)
    mdl = importlib.import_module(tokens[0])
    _cls = getattr(mdl, tokens[1])
    return _cls(*args, **kwargs)
