"""Shim onto the shared implementation in `scripts/my_utils.py`.

There used to be three near-identical copies of this file, one per engine plus the
one in `scripts/`. They drifted, and the drift was invisible: a helper added to only
one copy imports cleanly under the tests -- which put `scripts/` first on sys.path --
and then fails at run time for every dataset, which is how a nine-dataset batch died
with `ImportError: cannot import name 'scalar_mpp'`. Worse, `cli_params_block` read
`he0` in two copies and `he0fg` in the third; only the third matched the key actually
in parameters.json.

The engine still imports `my_utils` by bare name and still resolves it to this
sibling, so nothing about its import lines changes -- this file just forwards.

`cli_params_block()` reads the run's pair from the environment (`PHARAOH_PAIR`, which
`run_pharaoh.py` exports) rather than from any file's location, which is what lets
one engine serve every pair.
"""

import importlib.util as _importlib_util
import sys as _sys
from pathlib import Path as _Path

_SHARED_PATH = _Path(__file__).resolve().parents[1] / "my_utils.py"
_SHARED_NAME = "pharaoh_my_utils_shared"

_shared = _sys.modules.get(_SHARED_NAME)
if _shared is None:
    _spec = _importlib_util.spec_from_file_location(_SHARED_NAME, _SHARED_PATH)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"cannot load shared my_utils: {_SHARED_PATH}")
    _shared = _importlib_util.module_from_spec(_spec)
    _sys.modules[_SHARED_NAME] = _shared
    _spec.loader.exec_module(_shared)

globals().update({k: v for k, v in vars(_shared).items() if not k.startswith("__")})


def __getattr__(name):
    """Anything added to the shared module later stays reachable from here."""
    return getattr(_shared, name)
