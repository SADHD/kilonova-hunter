"""Kilonova Hunter: search Rubin alerts for kilonovae and explain the verdict.

The work is a pipeline. Each stage lives in its own module, takes data in,
and hands data to the next stage:

    fetch → clean → features → filter → report

``config`` holds the rule thresholds, in one place.
``inject`` builds a simple fake light curve for the demo.
``recall`` scores many fakes and reports how many the filter catches.
``tns`` looks up confirmed types on the Transient Name Server.
``cli`` is the ``kilohunt`` command, and the only module that talks to a person.
"""

__version__ = "0.1.0"
