import importlib

from .base import DDxDriver


def init_ddxdriver(class_name: str, *args, **kwargs) -> DDxDriver:
    tokens = class_name.rsplit(".", maxsplit=1)
    mdl = importlib.import_module(tokens[0])
    _cls = getattr(mdl, tokens[1])
    return _cls(*args, **kwargs)
