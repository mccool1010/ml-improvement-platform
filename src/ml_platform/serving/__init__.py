"""The model tier.

:mod:`ml_platform.serving.scoring` is the one place a loaded pipeline is turned
into a probability. :mod:`ml_platform.serving.predictor` wraps it in the HTTP
server KServe runs.

The application tier lives in :mod:`ml_platform.api` and does not duplicate any
of this; it calls it, either in process or over HTTP. See ADR-002.
"""

from ml_platform.serving.scoring import score

__all__ = ["score"]
