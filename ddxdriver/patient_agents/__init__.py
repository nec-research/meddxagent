import importlib

from .base import PatientAgent


def init_patient_agent(class_name: str, *args, **kwargs) -> PatientAgent:
    tokens = class_name.rsplit(".", maxsplit=1)
    mdl = importlib.import_module(tokens[0])
    _cls = getattr(mdl, tokens[1])
    return _cls(*args, **kwargs)
