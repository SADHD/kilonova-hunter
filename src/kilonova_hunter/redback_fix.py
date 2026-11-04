"""A wrapper around redback 1.20.0's kilonova diffusion integral.

redback's own files are not edited. This module patches one function
for the duration of a model call, then puts the original back.

The Arnett diffusion formula is::

    L(t) = exp(-t^2 / t_d^2) * integral Q(t') * (2 t' / t_d^2) * exp(t'^2 / t_d^2) dt'

``t_d`` is the diffusion time. ``Q`` is the thermalized heating rate.
redback 1.20.0 builds that integral with ``t / t_d^2`` where the
formula has ``2 t / t_d^2``, so the luminosity it returns is about half
of ``Q``. The integral is linear in ``Q``, so multiplying the function's
output by 2 puts the missing 2 back.

``t_d`` itself is already ``sqrt(2 * kappa * M / (beta * v * c))`` with
``beta = 13.7``. That square root is left alone. Doubling ``t_d`` would
be a different, wrong change: the time of the light curve would move.

The patch is only valid for redback 1.20.0. A later redback that already
includes the 2 would be doubled again. The test
``test_installed_redback_is_exactly_1_20_0`` fails in that case.

One process runs one patched call at a time. A lock makes a second
thread in that process wait, so one thread cannot restore the original
function while another thread is still inside it. The recall grid uses
a process pool. Each worker is its own process, with its own copy of
this module, so the workers do not share the lock.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

REQUIRED_REDBACK_VERSION = "1.20.0"

# Written on every cache manifest. A cache from before the wrapper does
# not have this marker, so it is not reused and it cannot score the test set.
DIFFUSION_FIX_ID = "arnett-factor-2"

_PATCH_LOCK = threading.RLock()
_PATCH_DEPTH = 0
_PATCH_ORIGINAL = None
_SAVED_USETEX: bool | None = None


def installed_redback_version() -> str:
    """The installed redback version, or ``not-installed``."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("redback")
    except PackageNotFoundError:
        return "not-installed"


def require_redback_1_20_0() -> None:
    """Stop if this is not the redback version the factor of 2 was written for."""
    found = installed_redback_version()
    if found != REQUIRED_REDBACK_VERSION:
        raise RuntimeError(
            "The diffusion wrapper multiplies redback 1.20.0's luminosity by 2. "
            f"This environment has redback {found}. "
            "Using the wrapper on another version could apply the factor twice "
            "or not at all."
        )


@contextmanager
def corrected_diffusion() -> Iterator[None]:
    """Use the factor-of-2 diffusion fix, then restore redback's function.

    Call the kilonova model inside this block. The model looks up
    ``_kilonova_diffusion_luminosity`` on the redback module when it
    runs, so replacing that name is enough. The original function is
    put back even if the model raises. Matplotlib's ``text.usetex``
    setting is put back to the value it had on the way in. redback
    turns that setting on during the call.
    """
    global _PATCH_DEPTH, _PATCH_ORIGINAL, _SAVED_USETEX

    require_redback_1_20_0()
    import matplotlib
    from redback.transient_models import kilonova_models

    with _PATCH_LOCK:
        if _PATCH_DEPTH == 0:
            current = kilonova_models._kilonova_diffusion_luminosity
            if getattr(current, "_hunter_diffusion_fix", False):
                raise RuntimeError(
                    "The diffusion fix is already applied. "
                    "Refusing to wrap it again, which would multiply by 4."
                )
            # Read usetex inside the lock. A second thread must not
            # record the value redback already changed.
            _SAVED_USETEX = bool(matplotlib.rcParams["text.usetex"])

            def fixed(time, thermalised_luminosity, diffusion_timescale):
                # The 2 is the missing piece of (2 t / t_d^2) in the integral.
                # t_d is the third argument and is not changed.
                return 2.0 * current(time, thermalised_luminosity, diffusion_timescale)

            fixed._hunter_diffusion_fix = True  # type: ignore[attr-defined]
            _PATCH_ORIGINAL = current
            kilonova_models._kilonova_diffusion_luminosity = fixed
        _PATCH_DEPTH += 1
        try:
            yield
        finally:
            _PATCH_DEPTH -= 1
            if _PATCH_DEPTH == 0 and _PATCH_ORIGINAL is not None:
                kilonova_models._kilonova_diffusion_luminosity = _PATCH_ORIGINAL
                _PATCH_ORIGINAL = None
                matplotlib.rcParams["text.usetex"] = bool(_SAVED_USETEX)
                _SAVED_USETEX = None
