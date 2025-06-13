import importlib
from typing import Any, Dict

from .base import Bench


def init_bench(bench_cfg: Dict[str, Any]) -> Bench:
    """
    Initializes and returns the benchmark class.

    The given configuration should contain:
    - a `bench_class` field, containing the qualified class name for the benchmark
    - optionally a `bench_args` field, containing additional arguments to the
      benchmark init

    Args:
        bench_cfg (dict): Benchmark configuration

    Return:
        Bench: Benchmark class
    """
    class_qualified_name = bench_cfg["class_name"]
    class_args = bench_cfg.get("config", {})
    agent_name, class_name = class_qualified_name.rsplit(".", maxsplit=1)

    _agent = importlib.import_module(agent_name)
    _cls = getattr(_agent, class_name)
    return _cls(**class_args)
