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
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

REQUIRED_REDBACK_VERSION = "1.20.0"


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
    put back even if the model raises.
    """
    require_redback_1_20_0()
    from redback.transient_models import kilonova_models

    current = kilonova_models._kilonova_diffusion_luminosity
    if getattr(current, "_hunter_diffusion_fix", False):
        yield
        return

    original = current

    def fixed(time, thermalised_luminosity, diffusion_timescale):
        # The 2 is the missing piece of (2 t / t_d^2) in the integral.
        # t_d is the third argument and is not changed.
        return 2.0 * original(time, thermalised_luminosity, diffusion_timescale)

    fixed._hunter_diffusion_fix = True  # type: ignore[attr-defined]
    kilonova_models._kilonova_diffusion_luminosity = fixed
    try:
        yield
    finally:
        kilonova_models._kilonova_diffusion_luminosity = original
        # The model call can turn LaTeX back on. Put it back off.
        import matplotlib

        matplotlib.rcParams["text.usetex"] = False
