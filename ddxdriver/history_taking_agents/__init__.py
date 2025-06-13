import importlib

from .base import HistoryTaking


def init_history_taking_agent(class_name: str, *args, **kwargs) -> HistoryTaking:
    tokens = class_name.rsplit(".", maxsplit=1)
    mdl = importlib.import_module(tokens[0])
    _cls = getattr(mdl, tokens[1])
    return _cls(*args, **kwargs)
