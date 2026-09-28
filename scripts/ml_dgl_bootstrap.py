"""Bootstrap DGL on Windows when GraphBolt C++ lib mismatches torch (e.g. 2.7.1).

ALIGNN inference uses core DGL graphs only — not GraphBolt distributed sampling.
Call ``ensure_dgl_importable()`` before ``import dgl``.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import sys
import types
import warnings


class _GraphBoltStub(types.ModuleType):
    """Minimal stand-in so ``from dgl import graphbolt as gb`` succeeds."""

    class FusedCSCSamplingGraph:  # noqa: D106
        pass

    @staticmethod
    def load_from_shared_memory(*_a, **_k):
        raise RuntimeError("GraphBolt disabled (Windows torch/DGL ABI stub)")

    @staticmethod
    def expand_indptr(*_a, **_k):
        raise RuntimeError("GraphBolt disabled (Windows torch/DGL ABI stub)")

    @staticmethod
    def fused_csc_sampling_graph(*_a, **_k):
        raise RuntimeError("GraphBolt disabled (Windows torch/DGL ABI stub)")

    @staticmethod
    def etype_tuple_to_str(x):
        return str(x)


class _GraphBoltStubFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "dgl.graphbolt":
            return importlib.util.spec_from_loader(fullname, _GraphBoltStubLoader())
        return None


class _GraphBoltStubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return _GraphBoltStub(spec.name)

    def exec_module(self, module):
        sys.modules["dgl.graphbolt"] = module


def ensure_dgl_importable() -> None:
    """Install GraphBolt stub finder once, then verify ``import dgl`` works."""
    if "dgl" in sys.modules:
        return
    if not any(isinstance(f, _GraphBoltStubFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _GraphBoltStubFinder())
    import dgl  # noqa: F401

    warnings.warn(
        "DGL GraphBolt C++ skipped (Windows + torch 2.7.x). "
        "ALIGNN / core DGL graphs OK; distributed GraphBolt sampling disabled.",
        UserWarning,
        stacklevel=2,
    )
