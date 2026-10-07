"""Finite-volume eigenmode solver for the transverse cross-section, inhomogeneous filling supported.

Cutoff (beta = 0) problems are exact decoupled scalar problems:
  H-polarisation (TE-like):  div( (1/eps) grad Hz ) + k0^2 mu Hz = 0   (natural/Neumann at PEC)
  E-polarisation (TM-like):  div( (1/mu)  grad Ez ) + k0^2 eps Ez = 0   (Dirichlet at PEC)
Quasi-TEM (static) modes: Laplace problems with eps and with eps=1 -> capacitance matrices,
  eps_eff, Z0 (single conductor) or modal eps_eff / modal impedance (several conductors).
Frequency-dependent (dispersive) propagation is handled by modeapp.vector.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as sla
from scipy.sparse.csgraph import connected_components
from .geometry import CrossSection
from .grid import Grid, build_grid, grid_symmetry

log = logging.getLogger(__name__)
C0 = 299_792_458.0
ETA0 = 376.730313668


def _acc(B, B0, cells, grp, val, gcol):
    gr = grp < 0
    np.add.at(B0, cells[gr], val[gr])
    sg = ~gr
    np.add.at(B, (cells[sg], gcol[grp[sg]]), val[sg])


@dataclass
class Assembly:
    A: sp.csr_matrix
    B: np.ndarray
    B0: np.ndarray
    n: int
    idx: np.ndarray


def _assemble(g: Grid, p: np.ndarray) -> Assembly:
    """Face conductances of div(p grad .) on free cells; Dirichlet couplings to conductors / outer box."""
    n = int(g.free.sum())
    idx = -np.ones((g.ny, g.nx), int); idx[g.free] = np.arange(n)
    gcol = np.full(max(g.groups + [0]) + 1, -1, int)
    for k, gid in enumerate(g.groups):
        gcol[gid] = k
    B = np.zeros((n, len(g.groups))); B0 = np.zeros(n)
    rows, cols, vals = [], [], []
    own = g.owner

    def pairs(a, b, oa, ob, wall, L, da, db, pa, pb):
        both = (oa == 0) & (ob == 0)
        link = both & (wall == 0)
        G = L / (da / (2 * pa) + db / (2 * pb))
        rows.append(a[link]); cols.append(b[link]); vals.append(G[link])
        m = both & (wall != 0)
        _acc(B, B0, a[m], wall[m], 2 * pa[m] * L[m] / da[m], gcol)
        _acc(B, B0, b[m], wall[m], 2 * pb[m] * L[m] / db[m], gcol)
        m = (oa == 0) & (ob != 0)
        _acc(B, B0, a[m], ob[m], 2 * pa[m] * L[m] / da[m], gcol)
        m = (ob == 0) & (oa != 0)
        _acc(B, B0, b[m], oa[m], 2 * pb[m] * L[m] / db[m], gcol)

    sh = (g.ny, g.nx - 1)
    pairs(idx[:, :-1], idx[:, 1:], own[:, :-1], own[:, 1:], g.wall_v[:, 1:-1],
          np.broadcast_to(g.dyc[:, None], sh), np.broadcast_to(g.dxc[None, :-1], sh),
          np.broadcast_to(g.dxc[None, 1:], sh), p[:, :-1], p[:, 1:])
    sh = (g.ny - 1, g.nx)
    pairs(idx[:-1, :], idx[1:, :], own[:-1, :], own[1:, :], g.wall_h[1:-1, :],
          np.broadcast_to(g.dxc[None, :], sh), np.broadcast_to(g.dyc[:-1, None], sh),
          np.broadcast_to(g.dyc[1:, None], sh), p[:-1, :], p[1:, :])
    for sl, val in ((idx[:, 0], 2 * p[:, 0] * g.dyc / g.dxc[0]), (idx[:, -1], 2 * p[:, -1] * g.dyc / g.dxc[-1]),
                    (idx[0, :], 2 * p[0, :] * g.dxc / g.dyc[0]), (idx[-1, :], 2 * p[-1, :] * g.dxc / g.dyc[-1])):
        ok = sl >= 0
        np.add.at(B0, sl[ok], val[ok])
    r, c, v = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    A = sp.coo_matrix((v, (r, c)), shape=(n, n)).tocsr()
    return Assembly(A + A.T, B, B0, n, idx)


def _smallest(S, k, sigma):
    n = S.shape[0]
    if n <= 1500 or k >= n - 1:
        w, v = np.linalg.eigh(S.toarray())
        return w[:k], v[:, :k]
    w, v = spla.eigsh(S.tocsc(), k=k, sigma=sigma, which="LM")
    o = np.argsort(w)
    return w[o], v[:, o]


def _expand(g, vec, fill=np.nan):
    out = np.full((g.ny, g.nx), fill, float)
    out[g.free] = vec
    return out


def _grad(psi, g):
    p = np.nan_to_num(psi)
    fx = (p[:, 1:] - p[:, :-1]) / (0.5 * (g.dxc[:-1] + g.dxc[1:]))[None, :]
    vx = g.free[:, 1:] & g.free[:, :-1] & (g.wall_v[:, 1:-1] == 0)
    fy = (p[1:, :] - p[:-1, :]) / (0.5 * (g.dyc[:-1] + g.dyc[1:]))[:, None]
    vy = g.free[1:, :] & g.free[:-1, :] & (g.wall_h[1:-1, :] == 0)
    gx, cx, gy, cy = (np.zeros_like(p) for _ in range(4))
    gx[:, :-1] += fx * vx; cx[:, :-1] += vx; gx[:, 1:] += fx * vx; cx[:, 1:] += vx
    gy[:-1, :] += fy * vy; cy[:-1, :] += vy; gy[1:, :] += fy * vy; cy[1:, :] += vy
    return (np.divide(gx, cx, out=np.zeros_like(gx), where=cx > 0),
            np.divide(gy, cy, out=np.zeros_like(gy), where=cy > 0))


def _parity(Ex, Ey, axis, w):
    """Parity of the transverse E vector field under mirror reflection about the centre line."""
    if axis == "x":
        Rx, Ry = -Ex[:, ::-1], Ey[:, ::-1]
    else:
        Rx, Ry = Ex[::-1, :], -Ey[::-1, :]
    den = float(np.sum(w * (Ex ** 2 + Ey ** 2)))
    s = float(np.sum(w * (Ex * Rx + Ey * Ry))) / den if den > 0 else 0.0
    return "even" if s > 0.9 else "odd" if s < -0.9 else "mixed"


def _normalise(Ex, Ey, g):
    s = np.sqrt(np.sum((Ex ** 2 + Ey ** 2) * g.area * g.free))
    return (Ex / s, Ey / s) if s > 0 else (Ex, Ey)


@dataclass
class Mode:
    number: int
    kind: str                 # TEM / quasi-TEM / TE / TM / TE-like / TM-like
    fc: float                 # Hz (0 for TEM / quasi-TEM)
    kc: float                 # 1/m (k0 at cutoff)
    psi: np.ndarray           # potential (TEM) / Ez (TM) / Hz (TE); NaN inside conductors
    Ex: np.ndarray
    Ey: np.ndarray
    parity_x: str
    parity_y: str
    z0: float | None = None
    eps_eff: float | None = None
    confidence: str = "High"

    @property
    def symmetry(self) -> str:
        return f"{self.parity_x}/{self.parity_y}"


@dataclass
class ModeSet:
    grid: Grid
    modes: list
    capacitance: np.ndarray | None     # C with the real filling, units of eps0 per metre
    capacitance_air: np.ndarray | None
    geom_sym: tuple

    @property
    def disc(self):                     # backward compatibility with stage 1
        return self.grid

    def table(self) -> str:
        lines = ["Mode | Type      | fc, MHz   | Symmetry (x/y) | eps_eff | Z0, Ohm | Confidence"]
        for m in self.modes:
            z = f"{m.z0:.2f}" if m.z0 else "-"
            ee = f"{m.eps_eff:.3f}" if m.eps_eff else "-"
            lines.append(f"{m.number:4d} | {m.kind:9s} | {m.fc / 1e6:9.2f} | {m.symmetry:14s} | {ee:>7s} | {z:>7s} | {m.confidence}")
        return "\n".join(lines)


def _static(g, p):
    asm = _assemble(g, p)
    LD = sp.diags(np.asarray(asm.A.sum(axis=1)).ravel() + asm.B.sum(axis=1) + asm.B0) - asm.A
    Phi = spla.splu(LD.tocsc()).solve(asm.B)
    C = np.diag(asm.B.sum(axis=0)) - asm.B.T @ Phi
    return 0.5 * (C + C.T), Phi


def solve(cs: CrossSection, dx: float = 5.0, n_modes: int = 8, edge_dx: float | None = None) -> ModeSet:
    """TEM/quasi-TEM + TE(-like) + TM(-like) modes. dx: max cell size, mm; edge_dx: optional fine size at conductor edges."""
    g = build_grid(cs, dx, edge_dx)
    log.info("grid %dx%d", g.nx, g.ny)
    homog = cs.is_homogeneous()
    area = g.area
    Lmax = max(cs.width, cs.height)
    sigma = -0.01 * (np.pi / Lmax) ** 2
    raw = []
    # H-polarisation (TE-like)
    asm = _assemble(g, 1.0 / g.eps)
    K = sp.diags(np.asarray(asm.A.sum(axis=1)).ravel()) - asm.A
    M = (g.mu * area)[g.free]
    D = sp.diags(1 / np.sqrt(M))
    ncomp = connected_components(asm.A, directed=False)[0]
    w, y = _smallest((D @ K @ D).tocsr(), n_modes + ncomp + 2, sigma)
    thr = 1e-6 * (np.pi / Lmax) ** 2
    for lam, vec in zip(w, y.T):
        if lam > thr and len(raw) < 2 * n_modes:
            raw.append(("TE" if homog else "TE-like", lam, D @ vec))
    # E-polarisation (TM-like)
    asm = _assemble(g, 1.0 / g.mu)
    KD = sp.diags(np.asarray(asm.A.sum(axis=1)).ravel() + asm.B.sum(axis=1) + asm.B0) - asm.A
    M = (g.eps * area)[g.free]
    D = sp.diags(1 / np.sqrt(M))
    w, y = _smallest((D @ KD @ D).tocsr(), n_modes + 1, sigma)
    for lam, vec in zip(w, y.T):
        raw.append(("TM" if homog else "TM-like", lam, D @ vec))
    cut = []
    epsc = np.where(g.free, g.eps, 1.0); muc = np.where(g.free, g.mu, 1.0)
    for kind, lam, vec in sorted(raw, key=lambda t: t[1])[:n_modes]:
        psi = _expand(g, vec)
        gx, gy = _grad(psi, g)
        Ex, Ey = (-gy / epsc, gx / epsc) if kind.startswith("TE") else (-gx / (epsc * muc), -gy / (epsc * muc))
        Ex, Ey = _normalise(Ex * g.free, Ey * g.free, g)
        k0 = np.sqrt(lam) * 1e3
        cut.append((C0 * k0 / (2 * np.pi), kind, k0, psi, Ex, Ey, None, None))
    # quasi-TEM from static capacitance matrices
    tem, C, Cair = [], None, None
    if g.groups:
        C, Phi = _static(g, g.eps)
        Cair, _ = _static(g, np.ones_like(g.eps))
        lam, V = sla.eigh(C, Cair)
        for m in range(len(lam)):
            v = V[:, m] / np.linalg.norm(V[:, m])
            v = v * (1 if v[np.argmax(abs(v))] > 0 else -1)
            phi = _expand(g, Phi @ v)
            full = phi.copy()
            for k, gid in enumerate(g.groups):
                full[g.owner == gid] = v[k]
            full[g.owner == -1] = 0.0
            gx, gy = _grad(np.nan_to_num(full) * 1.0, g)
            Ex, Ey = _normalise(-gx * g.free, -gy * g.free, g)
            ee = float(lam[m])
            z0 = ETA0 * np.sqrt(ee) * (v @ v) / (v @ C @ v)
            tem.append((0.0, "TEM" if homog else "quasi-TEM", 0.0, full, Ex, Ey, z0, ee))
    modes = []
    w2 = area / area.sum()
    sym = grid_symmetry(g)
    for t in tem + cut:
        fc, kind, kc, psi, Ex, Ey, z0, ee = t
        px = _parity(Ex, Ey, "x", w2) if sym[0] else "n/a"     # parity is only defined along symmetric axes
        py = _parity(Ex, Ey, "y", w2) if sym[1] else "n/a"
        modes.append(Mode(len(modes), kind, fc, kc, psi, Ex, Ey, px, py, z0, ee,
                          "High" if "mixed" not in (px, py) else "Medium"))
    return ModeSet(g, modes, C, Cair, sym)
