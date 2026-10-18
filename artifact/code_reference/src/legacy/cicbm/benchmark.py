"""Compatibility layer for the historic ``stage4.refined.benchmark`` module.

The implementation is archived together with its experiment outputs and its
``run_config.json`` at::

    experiments/archive/stage4/refined/benchmark.py

It is loaded in place here so that ``from src.legacy.cicbm import benchmark``
keeps working for the archived experiment scripts, while the module continues
to read its config and write its artefacts next to the original experiment.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_TARGET = (
    Path(__file__).resolve().parents[3]
    / "experiments" / "archive" / "stage4" / "refined" / "benchmark.py"
)

_spec = importlib.util.spec_from_file_location(__name__ + "._impl", _TARGET)
_impl = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _impl
_spec.loader.exec_module(_impl)

sys.modules[__name__] = _impl
