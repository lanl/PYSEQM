import sys
from importlib import import_module

from . import seqm_functions as seqm_functions  # noqa: F401
from .api import *  # noqa: F401,F403

# ---------------------------------------------------------------------------
# Backward-compatible module aliases for helpers that moved into subpackages.
# This avoids littering the top-level directory while keeping legacy imports
# like `import seqm.nac_utils` working.
_MODULE_ALIASES = {
    "seqm.XLBOMD": "seqm.dynamics.xlbomd",
    "seqm.geometryOptimization": "seqm.optimization.geometry",
    "seqm.tools": "seqm.utils.profiling",
}

for old_name, new_name in _MODULE_ALIASES.items():
    if old_name not in sys.modules:
        try:
            module = import_module(new_name)
            sys.modules[old_name] = module
            setattr(sys.modules[__name__], old_name.split(".")[-1], module)
        except ModuleNotFoundError:
            # If an optional dependency is missing we let the import fail later.
            continue
