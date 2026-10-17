#!/usr/bin/env python3
"""CLI entry point for the DAPI Stage-8 engine implementation."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ENGINE_PATH = Path(__file__).resolve().parent / "engine_dapi" / "8_refine_alignment.py"


def _load_engine():
    module_name = "pharaoh_engine_dapi_stage8"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Stage-8 engine: {ENGINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_ENGINE = _load_engine()
main = _ENGINE.main


def __getattr__(name):
    """Preserve imports of Stage-8 helpers while implementation lives in the engine."""
    return getattr(_ENGINE, name)


def __dir__():
    return sorted(set(globals()).union(dir(_ENGINE)))


if __name__ == "__main__":
    main()
