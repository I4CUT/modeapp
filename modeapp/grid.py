"""Non-uniform tensor-product grid with grid lines on every conductor/dielectric edge.

Cell (j,i) spans [xn[i],xn[i+1]] x [yn[j],yn[j+1]].  Arrays are indexed [row j (y), column i (x)].
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .geometry import CrossSection


def _uniq(vals, tol):
    v = np.sort(np.asarray(vals, float))
    out = [v[0]]
    for x in v[1:]:
        if x - out[-1] > tol:
            out.append(x)
    return np.array(out)


def _subdivide(a, b, hmax, ha=None, hb=None, g=1.4):
    L = b - a
    left, right = [], []
    if ha:
        s, pos = ha, 0.0
        while s < hmax and pos + s < 0.5 * L - 1e-12:
            left.append(s); pos += s; s *= g
    if hb:
        s, pos = hb, 0.0
        while s < hmax and pos + s < 0.5 * L - 1e-12:
            right.append(s); pos += s; s *= g
    mid = L - sum(left) - sum(right)
    nmid = max(1, int(np.ceil(mid / hmax - 1e-9)))
    return left + [mid / nmid] * nmid + right[::-1]


def _axis(coords, edges, hmax, edge_h, grade, tol):
    nodes = [coords[0]]
    for a, b in zip(coords[:-1], coords[1:]):
        ha = edge_h if edge_h and any(abs(a - e) <= tol for e in edges) else None
        hb = edge_h if edge_h and any(abs(b - e) <= tol for e in edges) else None
        parts = _subdivide(a, b, hmax, ha, hb, grade)
        nodes.extend(a + np.cumsum(parts)[:-1])
        nodes.append(b)
    return np.array(nodes)


@dataclass
class Grid:
    cs: CrossSection
    xn: np.ndarray
    yn: np.ndarray
    nx: int
    ny: int
    dxc: np.ndarray
    dyc: np.ndarray
    xc: np.ndarray
    yc: np.ndarray
    dxn: np.ndarray       # dual spacing at nodes (length nx+1)
    dyn: np.ndarray
    eps: np.ndarray       # (ny,nx)
    mu: np.ndarray
    owner: np.ndarray     # (ny,nx) conductor group of blocked cells (-1 ground), 0 free
    wall_h: np.ndarray    # (ny+1,nx) thin walls
    wall_v: np.ndarray    # (ny,nx+1)
    free: np.ndarray
    area: np.ndarray
    pec_x: np.ndarray     # (ny+1,nx) horizontal edges that are PEC
    pec_y: np.ndarray     # (ny,nx+1) vertical edges that are PEC
    pec_node: np.ndarray  # (ny+1,nx+1)
    groups: list          # signal conductor group ids (>0)


def build_grid(cs: CrossSection, dx: float, edge_dx: float | None = None, grade: float = 1.4) -> Grid:
    W, H = cs.width, cs.height
    tol = 1e-9 * max(W, H)
    cl = lambda v, hi: float(np.clip(v, 0.0, hi))
    xs, ys, ex, ey = [0.0, W], [0.0, H], [], []
    for r in cs.conductors:
        for v in (r.x0, r.x1):
            xs.append(cl(v, W)); ex.append(cl(v, W))
        for v in (r.y0, r.y1):
            ys.append(cl(v, H)); ey.append(cl(v, H))
    for d in cs.dielectrics:
        xs += [cl(d.x0, W), cl(d.x1, W)]
        ys += [cl(d.y0, H), cl(d.y1, H)]
    xn = _axis(_uniq(xs, tol), ex, dx, edge_dx, grade, tol)
    yn = _axis(_uniq(ys, tol), ey, dx, edge_dx, grade, tol)
    nx, ny = len(xn) - 1, len(yn) - 1
    dxc, dyc = np.diff(xn), np.diff(yn)
    xc, yc = 0.5 * (xn[1:] + xn[:-1]), 0.5 * (yn[1:] + yn[:-1])
    dxn = np.empty(nx + 1); dxn[1:-1] = 0.5 * (dxc[:-1] + dxc[1:]); dxn[0] = 0.5 * dxc[0]; dxn[-1] = 0.5 * dxc[-1]
    dyn = np.empty(ny + 1); dyn[1:-1] = 0.5 * (dyc[:-1] + dyc[1:]); dyn[0] = 0.5 * dyc[0]; dyn[-1] = 0.5 * dyc[-1]
    eps = np.full((ny, nx), float(cs.eps_r)); mu = np.full((ny, nx), float(cs.mu_r))
    for d in cs.dielectrics:
        m = ((yc > d.y0) & (yc < d.y1))[:, None] & ((xc > d.x0) & (xc < d.x1))[None, :]
        eps[m], mu[m] = d.eps_r, d.mu_r
    owner = np.zeros((ny, nx), int)
    wall_h = np.zeros((ny + 1, nx), int)
    wall_v = np.zeros((ny, nx + 1), int)
    groups = sorted({r.group for r in cs.conductors if not r.ground and r.group > 0})
    for r in cs.conductors:
        gid = -1 if r.ground else r.group
        zx, zy = abs(r.x1 - r.x0) <= tol, abs(r.y1 - r.y0) <= tol
        inx, iny = (xc > r.x0) & (xc < r.x1), (yc > r.y0) & (yc < r.y1)
        if zy and not zx:
            wall_h[int(np.argmin(abs(yn - r.y0))), (xc > r.x0 - tol) & (xc < r.x1 + tol)] = gid
        elif zx and not zy:
            wall_v[(yc > r.y0 - tol) & (yc < r.y1 + tol), int(np.argmin(abs(xn - r.x0)))] = gid
        else:
            owner[np.ix_(iny, inx)] = gid
    free = owner == 0
    blocked = ~free
    pec_x = np.zeros((ny + 1, nx), bool); pec_x[0, :] = pec_x[ny, :] = True
    pec_x |= wall_h != 0; pec_x[1:, :] |= blocked; pec_x[:-1, :] |= blocked
    pec_y = np.zeros((ny, nx + 1), bool); pec_y[:, 0] = pec_y[:, nx] = True
    pec_y |= wall_v != 0; pec_y[:, 1:] |= blocked; pec_y[:, :-1] |= blocked
    pn = np.zeros((ny + 1, nx + 1), bool)
    pn[:, :-1] |= pec_x; pn[:, 1:] |= pec_x; pn[:-1, :] |= pec_y; pn[1:, :] |= pec_y
    return Grid(cs, xn, yn, nx, ny, dxc, dyc, xc, yc, dxn, dyn, eps, mu, owner, wall_h, wall_v, free,
                np.outer(dyc, dxc), pec_x, pec_y, pn, groups)


def grid_symmetry(g: Grid) -> tuple[bool, bool]:
    def same(fx):
        return (np.array_equal(g.owner, fx(g.owner)) and np.allclose(g.eps, fx(g.eps)) and np.allclose(g.mu, fx(g.mu)))
    sx = (same(lambda a: a[:, ::-1]) and np.array_equal(g.wall_h, g.wall_h[:, ::-1])
          and np.array_equal(g.wall_v, g.wall_v[:, ::-1]) and np.allclose(g.dxc, g.dxc[::-1], rtol=1e-9))
    sy = (same(lambda a: a[::-1, :]) and np.array_equal(g.wall_h, g.wall_h[::-1, :])
          and np.array_equal(g.wall_v, g.wall_v[::-1, :]) and np.allclose(g.dyc, g.dyc[::-1], rtol=1e-9))
    return bool(sx), bool(sy)
