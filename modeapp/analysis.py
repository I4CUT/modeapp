"""Mode classification helpers: excitability and recommended single-mode range."""
from __future__ import annotations
from .solver import ModeSet, Mode


def excitability(m: Mode, ms: ModeSet, source=("even", "even"), f: float | None = None):
    """Return (status, reason). `source` = parity of the source E-field under x and y mirror reflections.
    A mode can only be driven by a source of the same parity along every axis in which the geometry is symmetric."""
    if f is not None and m.fc > 0 and f < m.fc:
        return "NO", "Frequency below cutoff"
    for ax, sym, mp, sp_ in zip("xy", ms.geom_sym, (m.parity_x, m.parity_y), source):
        if not sym:
            continue
        if mp == "mixed":
            return "MAYBE", f"Mixed symmetry along {ax} (degenerate modes)"
        if mp != sp_:
            return "NO", f"Symmetry mismatch along {ax} (mode {mp}, source {sp_})"
    return "YES", "Symmetry compatible"


def single_mode_range(ms: ModeSet, source=("even", "even"), margin: float = 0.20, fstart: float = 10e6):
    higher = [m for m in ms.modes if m.fc > 0]
    if not higher:
        return {}
    first = min(higher, key=lambda m: m.fc)
    exc = [m for m in higher if excitability(m, ms, source)[0] in ("YES", "MAYBE")]
    first_exc = min(exc, key=lambda m: m.fc) if exc else None
    out = {"fundamental": ms.modes[0].kind, "first_higher_fc": first.fc, "first_higher": first,
           "first_excitable_fc": first_exc.fc if first_exc else None, "first_excitable": first_exc,
           "range_conservative": (fstart, first.fc), "range_conservative_margin": (fstart, first.fc * (1 - margin))}
    if first_exc:
        out["range_excitable"] = (fstart, first_exc.fc)
        out["range_excitable_margin"] = (fstart, first_exc.fc * (1 - margin))
    return out
