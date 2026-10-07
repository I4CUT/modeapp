"""Parametric 2D cross-sections. All lengths in millimetres.

Outer boundary: PEC box [0,width] x [0,height].
Conductors: rectangles; same `group` = one electrical body; `ground=True` = tied to the outer box.
A conductor with zero extent in one direction is an infinitely thin PEC wall.
Dielectrics: rectangles overriding the background eps_r / mu_r (later entries override earlier).
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float
    group: int = 1
    ground: bool = False


@dataclass
class Dielectric:
    x0: float
    y0: float
    x1: float
    y1: float
    eps_r: float = 1.0
    mu_r: float = 1.0
    name: str = ""


@dataclass
class CrossSection:
    width: float
    height: float
    conductors: list = field(default_factory=list)
    eps_r: float = 1.0          # background filling
    name: str = "cross-section"
    dielectrics: list = field(default_factory=list)
    mu_r: float = 1.0

    def is_homogeneous(self) -> bool:
        return all(abs(d.eps_r - self.eps_r) < 1e-12 and abs(d.mu_r - self.mu_r) < 1e-12 for d in self.dielectrics)


def rectangular_waveguide(a: float = 22.86, b: float = 10.16, eps_r: float = 1.0) -> CrossSection:
    return CrossSection(a, b, [], eps_r, "Rectangular waveguide")


def symmetric_stripline(width: float = 500.0, height: float = 205.0, strip_width: float = 280.0,
                        thickness: float = 1.0, eps_r: float = 1.0, y_offset: float = 0.0) -> CrossSection:
    """Strip centred in x (and in y unless y_offset != 0). thickness=0 -> infinitely thin strip."""
    xc, yc = width / 2.0, height / 2.0 + y_offset
    strip = Rect(xc - strip_width / 2, yc - thickness / 2, xc + strip_width / 2, yc + thickness / 2, 1)
    return CrossSection(width, height, [strip], eps_r, "Symmetric stripline")


def microstrip(h: float = 1.0, w: float = 1.0, t: float = 0.035, eps_r: float = 9.8,
               box_w: float | None = None, box_h: float | None = None) -> CrossSection:
    """Open-ish microstrip in a shielding box; ground = bottom wall; substrate fills [0,h]."""
    box_w = box_w or max(20 * h, 10 * w)
    box_h = box_h or 12 * h
    xc = box_w / 2
    sub = Dielectric(0, 0, box_w, h, eps_r, 1.0, "substrate")
    strip = Rect(xc - w / 2, h, xc + w / 2, h + t, 1)
    return CrossSection(box_w, box_h, [strip], 1.0, "Microstrip", [sub])


def cpw(w: float = 1.0, gap: float = 0.5, h: float = 1.0, t: float = 0.035, eps_r: float = 9.8,
        box_w: float | None = None, box_h: float | None = None) -> CrossSection:
    """Conductor-backed-free CPW: centre strip + coplanar grounds tied to the box, substrate on bottom wall."""
    box_w = box_w or max(20 * h, 10 * (w + 2 * gap))
    box_h = box_h or 12 * h
    xc = box_w / 2
    sub = Dielectric(0, 0, box_w, h, eps_r, 1.0, "substrate")
    strip = Rect(xc - w / 2, h, xc + w / 2, h + t, 1)
    gl = Rect(0.0, h, xc - w / 2 - gap, h + t, 0, True)
    gr = Rect(xc + w / 2 + gap, h, box_w, h + t, 0, True)
    return CrossSection(box_w, box_h, [strip, gl, gr], 1.0, "CPW", [sub])


def rect_coax(outer_w: float = 20.0, outer_h: float = 20.0, inner_w: float = 8.0, inner_h: float = 8.0,
              eps_r: float = 1.0) -> CrossSection:
    xc, yc = outer_w / 2, outer_h / 2
    inner = Rect(xc - inner_w / 2, yc - inner_h / 2, xc + inner_w / 2, yc + inner_h / 2, 1)
    return CrossSection(outer_w, outer_h, [inner], eps_r, "Rectangular coax")
