"""Import gCastle's PC, TTPM and MetricsDAG with or without PyTorch installed."""
from __future__ import annotations

import os
import sys
import types
from functools import lru_cache


@lru_cache(maxsize=None)
def load_castle():
    """Return (PC, TTPM, MetricsDAG).

    gCastle 1.0.x imports torch when `castle.algorithms` loads, although PC and TTPM do not use it.
    If only torch is missing, load the two submodules directly and skip the package __init__.
    """
    try:
        from castle.algorithms import PC, TTPM
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        import castle

        sys.modules.pop("castle.algorithms", None)
        pkg = types.ModuleType("castle.algorithms")
        pkg.__path__ = [os.path.join(os.path.dirname(castle.__file__), "algorithms")]
        sys.modules["castle.algorithms"] = pkg
        from castle.algorithms.pc.pc import PC
        from castle.algorithms.ttpm.ttpm import TTPM
    from castle.metrics import MetricsDAG

    return PC, TTPM, MetricsDAG
