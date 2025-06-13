import importlib

from .base import Diagnosis


def init_diagnosis_agent(class_name: str, *args, **kwargs) -> Diagnosis:
    tokens = class_name.rsplit(".", maxsplit=1)
    mdl = importlib.import_module(tokens[0])
    _cls = getattr(mdl, tokens[1])
    return _cls(*args, **kwargs)
