"""modeapp: 2D cross-section mode analysis (stage 1: homogeneous, stage 2: inhomogeneous + full-vector)."""
from .geometry import (CrossSection, Rect, Dielectric, rectangular_waveguide, symmetric_stripline,
                       microstrip, cpw, rect_coax)
from .solver import solve, ModeSet, Mode
from .analysis import excitability, single_mode_range
__all__ = ["CrossSection", "Rect", "Dielectric", "rectangular_waveguide", "symmetric_stripline", "microstrip",
           "cpw", "rect_coax", "solve", "ModeSet", "Mode", "excitability", "single_mode_range"]
