"""Kilonova Hunter: search Rubin alerts for kilonovae and explain the verdict.

The work is a pipeline. Each stage lives in its own module, takes data in,
and hands data to the next stage:

    fetch → clean → features → filter → report

``inject`` builds fake kilonovae so the filter can be scored.
``tns`` looks up confirmed types on the Transient Name Server.
``cli`` is the ``kilohunt`` command, and the only module that talks to a person.
"""

__version__ = "0.1.0"
