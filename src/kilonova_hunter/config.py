"""The numbers the kilonova rules compare against.

Every cut lives in this file. Changing a threshold means editing the
``THRESHOLDS`` value below, not hunting through the filter.

This is a frozen dataclass, not a separate ``thresholds.toml`` file.
A dataclass is ordinary Python, so a reader who is learning the rest
of this package does not also have to learn a second file format, and
the comment that explains a number can sit directly above it. A TOML
file inside the package would also need a path lookup after install.
Frozen means a running program cannot overwrite a cut by accident
(``THRESHOLDS.fade_faster_than_mag_per_day = 1`` raises). A test that
wants a different cut builds a new ``Thresholds`` instead.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    """One complete set of cuts for the v1 rules.

    Build a second instance to try a different cut. The fields are the
    only numbers the rules read.

    Attributes:
        fade_faster_than_mag_per_day: A band fades "fast enough" when
            its fitted slope is strictly above this, in magnitudes per
            day. The rule passes if any one of ``fade_bands`` does.
        redder_than_g_minus_r: The color rule passes when g minus r is
            strictly above this after the peak. A larger g minus r is
            redder, because g has become the fainter band.
        min_g_minus_r_rise: The reddening rule passes when g minus r
            climbs by strictly more than this, in magnitudes, between
            the earliest color and a later color. A positive number
            means "it has to get redder", not merely "it is red".
        min_points_per_band: How many points a band needs after its
            peak before a straight line is worth fitting. Two is the
            minimum, because a line is defined by two different times.
        fade_window_days: How many days after a band's peak are
            included in that band's fade fit.
        color_days_after_peak: How long after the r-band peak we read
            g minus r. Zero would read the color at the peak itself.
        reddening_window_days: How many days after the earliest color
            we still look for a later color. The later color has to
            fall inside this window.
        min_color_separation_days: The smallest gap, in days, between
            the two colors the reddening rule compares. Two snapshots
            of the same visit are not a change.
        fade_bands: The filters the fade rule is allowed to use. The
            rule passes when any one of these fades fast enough.
    """

    fade_faster_than_mag_per_day: float
    redder_than_g_minus_r: float
    min_g_minus_r_rise: float
    min_points_per_band: int
    fade_window_days: float
    color_days_after_peak: float
    reddening_window_days: float
    min_color_separation_days: float
    fade_bands: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject settings that cannot describe a fade, a color, or a rise."""
        if self.min_points_per_band < 2:
            raise ValueError(
                "min_points_per_band must be at least 2, because a straight "
                f"line needs two points, got {self.min_points_per_band}."
            )
        if self.fade_window_days <= 0:
            raise ValueError(
                "fade_window_days must be a positive number of days, "
                f"got {self.fade_window_days}."
            )
        if self.color_days_after_peak < 0:
            raise ValueError(
                "color_days_after_peak cannot be negative, "
                f"got {self.color_days_after_peak}."
            )
        if self.reddening_window_days <= 0:
            raise ValueError(
                "reddening_window_days must be a positive number of days, "
                f"got {self.reddening_window_days}."
            )
        if self.min_color_separation_days < 0:
            raise ValueError(
                "min_color_separation_days cannot be negative, "
                f"got {self.min_color_separation_days}."
            )
        if self.min_color_separation_days > self.reddening_window_days:
            raise ValueError(
                "min_color_separation_days cannot be longer than "
                "reddening_window_days, because the later color has to sit "
                "inside the window "
                f"({self.min_color_separation_days} days is longer than "
                f"{self.reddening_window_days} days)."
            )
        if not self.fade_bands:
            raise ValueError(
                "fade_bands needs at least one filter, such as g, r, or i."
            )


# The fade cut below is signed off. ZTFReST (Andreoni et al. 2021,
# arXiv:2104.06352, section II.2) uses a hard cut of fading at least
# 0.3 mag/day in any one of g, r, or i, and the science lead kept that
# cut for v1 on Oct 31, 2026. "Faster than" means the fitted slope must
# be above this number, not equal to it.
#
# PLACEHOLDER. TODO(Science Lead): replace redder_than_g_minus_r on
# Tuesday with the real cut for how large g minus r must be after the
# peak. 0.0 only means "g is fainter than r". It is here so the tests
# have a number to read. It is not a science result.
#
# PLACEHOLDER. TODO(Science Lead): replace the three reddening settings
# on Tuesday Nov 3, 2026, after checking AT2017gfo. They are here so the
# tests have numbers to read. They are not a science result.
#
# min_g_minus_r_rise is how many magnitudes g minus r must climb.
# 0.1 only means "a little redder than it started".
# reddening_window_days is how many days after the first color we still
# look for a later color. 2 days follows the demo's "turns red within
# about two days" shape, not a measured window.
# min_color_separation_days is the smallest gap between those two
# colors. 0.5 days matches the demo's half-day spacing, so two
# neighboring visits can be compared and two copies of one visit cannot.
#
# The remaining fields are helper settings, not science cuts. Two
# points is the smallest line. Three days is the "first few days" the
# demo's fade rate describes. One day after the r-band peak is long
# enough for the demo to have turned red; the color at the peak itself
# can still be blue. g, r, and i are the bands ZTFReST allows the fade
# rule to use.
THRESHOLDS = Thresholds(
    fade_faster_than_mag_per_day=0.3,
    redder_than_g_minus_r=0.0,
    min_g_minus_r_rise=0.1,
    min_points_per_band=2,
    fade_window_days=3.0,
    color_days_after_peak=1.0,
    reddening_window_days=2.0,
    min_color_separation_days=0.5,
    fade_bands=("g", "r", "i"),
)


def resolve_thresholds(thresholds: Thresholds | None) -> Thresholds:
    """Return the cuts to use for this call.

    ``None`` means "use the project defaults". Passing a ``Thresholds``
    lets a test change one number without editing this file.
    """
    if thresholds is None:
        return THRESHOLDS
    return thresholds
