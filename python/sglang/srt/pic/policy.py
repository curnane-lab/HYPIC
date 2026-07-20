"""PICPolicy: attribute-driven description of a PIC method.

Replaces scattered ``pic_mode`` string comparisons with three orthogonal
attributes. The four public method names remain as aliases (see
:data:`POLICIES`); code branches on attributes, not names, so adding a method
means adding one table row rather than editing every conditional.

Note: the *seam width* is NOT a policy attribute — it is an independent
runtime knob (``PIC_SEAM_SINK`` env / ``resolve_seam_sink_tokens``). The
policy only records *whether* a mode recomputes the seam.
"""
from __future__ import annotations

from enum import Enum, auto

import msgspec


class PICCompose(Enum):
    ADDITION = auto()    # cache and add S_{C|0} only, no transition operator
    TRANSITION = auto()  # transition-operator compose (T_C, S_{C|0})


class PICPolicy(msgspec.Struct, frozen=True):
    compose: PICCompose  # compose operator
    rope: bool           # rope re-rotation correction (also selects cache schema / mamba_idx)
    recompute: bool      # recompute the hit-segment seam window


# The four method names = four instances. Names stay as aliases (config /
# experiments / logs are unchanged); dispatch reads attributes.
POLICIES: dict[str, PICPolicy] = {
    "addition":                  PICPolicy(PICCompose.ADDITION,   rope=False, recompute=False),
    "transition":                PICPolicy(PICCompose.TRANSITION, rope=False, recompute=False),
    "transition_rope":           PICPolicy(PICCompose.TRANSITION, rope=True,  recompute=False),
    "transition_rope_recompute": PICPolicy(PICCompose.TRANSITION, rope=True,  recompute=True),
}


def resolve_policy(pic_mode: str) -> PICPolicy:
    """Look up the :class:`PICPolicy` for a mode name. Raises KeyError on unknown."""
    return POLICIES[pic_mode]
