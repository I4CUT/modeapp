"""Validation against analytical solutions."""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from scipy.special import ellipk
from .geometry import rectangular_waveguide
from .solver import solve, C0


def validate_rect_waveguide(a=22.86, b=10.16, dx=0.5, n=5):
    ms = solve(rectangular_waveguide(a, b), dx=dx, n_modes=n)
    ana = []
    for m in range(0, 6):
        for k in range(0, 6):
            if m == k == 0:
                continue
            fc = C0 / 2 * np.hypot(m / a, k / b) * 1e3
            ana.append((fc, f"TE{m}{k}"))
            if m >= 1 and k >= 1:
                ana.append((fc, f"TM{m}{k}"))
    ana.sort()
    return [(name, mode.kind, fa / 1e9, mode.fc / 1e9, 100 * (mode.fc - fa) / fa)
            for mode, (fa, name) in zip([m for m in ms.modes if m.kind not in ("TEM", "quasi-TEM")], ana)]


def stripline_z0_thin_strip(width_strip, b, eps_r=1.0):
    """Cohn: thin strip between infinite ground planes spaced b."""
    k = 1.0 / np.cosh(np.pi * width_strip / (2 * b))
    return 30 * np.pi / np.sqrt(eps_r) * ellipk(k ** 2) / ellipk(1 - k ** 2)


# ---- rectangular guide, dielectric slab of thickness t against the side wall x=0, full height ----
def _sk(x, L):
    if abs(x) < 1e-14:
        return L
    return np.sin(np.sqrt(x) * L) / np.sqrt(x) if x > 0 else np.sinh(np.sqrt(-x) * L) / np.sqrt(-x)


def _ck(x, L):
    return np.cos(np.sqrt(x) * L) if x >= 0 else np.cosh(np.sqrt(-x) * L)


def _G(beta2, k0, a, b, t, eps, n, fam):
    ky = n * np.pi / b
    kd2 = eps * k0 ** 2 - beta2 - ky ** 2
    if kd2 <= 0:
        return np.nan
    kd, x, L = np.sqrt(kd2), k0 ** 2 - beta2 - ky ** 2, a - t
    if fam == "LSE":
        return kd * np.cos(kd * t) * _sk(x, L) + _ck(x, L) * np.sin(kd * t)
    return (kd / eps) * np.sin(kd * t) * _ck(x, L) + x * _sk(x, L) * np.cos(kd * t)


def _roots(fun, lo, hi, N=6000):
    xs = np.linspace(lo, hi, N)
    ys = np.array([fun(x) for x in xs])
    out = []
    for i in range(N - 1):
        if np.isfinite(ys[i]) and np.isfinite(ys[i + 1]) and ys[i] * ys[i + 1] < 0:
            out.append(brentq(fun, xs[i], xs[i + 1], xtol=1e-14))
    return out


def loaded_wg_betas(f, a=22.86, b=10.16, t=8.0, eps=4.0, nmax=3):
    """Analytic beta (1/m) of LSE_{mn}/LSM_{mn}; lengths in mm. Returns [(beta, 'LSE m n index'), ...] descending."""
    k0 = 2 * np.pi * f / C0 * 1e-3
    res = []
    for fam in ("LSE", "LSM"):
        for n in range(0 if fam == "LSE" else 1, nmax + 1):
            hi = eps * k0 ** 2 - (n * np.pi / b) ** 2
            if hi <= 0:
                continue
            for k, r in enumerate(_roots(lambda b2: _G(b2, k0, a, b, t, eps, n, fam), 1e-12, hi * 0.9999)):
                res.append((np.sqrt(r) * 1e3, f"{fam}_{k + 1},{n}"))
    return sorted(res, reverse=True)


def loaded_wg_cutoffs(a=22.86, b=10.16, t=8.0, eps=4.0, nmax=3, fmax=40e9):
    kmax = 2 * np.pi * fmax / C0 * 1e-3
    res = []
    for fam in ("LSE", "LSM"):
        for n in range(0 if fam == "LSE" else 1, nmax + 1):
            fn = lambda k0: _G(0.0, k0, a, b, t, eps, n, fam)
            lo = n * np.pi / b / np.sqrt(eps) * 1.0001 + 1e-9
            for k, r in enumerate(_roots(fn, lo, kmax, 20000)):
                res.append((C0 * r * 1e3 / (2 * np.pi), f"{fam}_{k + 1},{n}"))
    return sorted(res)


def microstrip_hammerstad(w_over_h, eps_r):
    u = w_over_h
    if u <= 1:
        ee = (eps_r + 1) / 2 + (eps_r - 1) / 2 * ((1 + 12 / u) ** -0.5 + 0.04 * (1 - u) ** 2)
        z0 = 60 / np.sqrt(ee) * np.log(8 / u + u / 4)
    else:
        ee = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12 / u) ** -0.5
        z0 = 120 * np.pi / (np.sqrt(ee) * (u + 1.393 + 0.667 * np.log(u + 1.444)))
    return ee, z0
