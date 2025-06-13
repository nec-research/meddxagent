import importlib

from .base import RAG


def init_rag_agent(class_name: str, *args, **kwargs) -> RAG:
    tokens = class_name.rsplit(".", maxsplit=1)
    mdl = importlib.import_module(tokens[0])
    _cls = getattr(mdl, tokens[1])
    return _cls(*args, **kwargs)
