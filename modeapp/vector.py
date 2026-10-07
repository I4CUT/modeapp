"""Full-vector frequency-domain mode solver on a non-uniform Yee grid (transverse-H formulation).

Unknowns Hx (on vertical cell edges) and Hy (on horizontal cell edges); eigenvalue = beta^2 at fixed frequency:
    beta^2 Hx = k0^2 eps_y Hx + d/dx(div Ht) - eps_y d/dy( curl_z Ht / eps_z )
    beta^2 Hy = k0^2 eps_x Hy + d/dy(div Ht) + eps_x d/dx( curl_z Ht / eps_z )
PEC: normal H = 0 and tangential E = 0 on outer box / conductors.  Limitation: mu_r = 1 only.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from .geometry import CrossSection
from .grid import Grid, build_grid
from .solver import ETA0, C0


def _edge_eps(g: Grid):
    eps = np.where(g.free, g.eps, 0.0); w = g.free.astype(float)
    def avg(num, den):
        return np.where(den > 0, num / np.maximum(den, 1e-30), 1.0)
    ny, nx = g.ny, g.nx
    nh = np.zeros((ny + 1, nx)); dh = np.zeros((ny + 1, nx))
    nh[1:, :] += eps; dh[1:, :] += w; nh[:-1, :] += eps; dh[:-1, :] += w
    nv = np.zeros((ny, nx + 1)); dv = np.zeros((ny, nx + 1))
    nv[:, 1:] += eps; dv[:, 1:] += w; nv[:, :-1] += eps; dv[:, :-1] += w
    nn = np.zeros((ny + 1, nx + 1)); dn = np.zeros((ny + 1, nx + 1))
    for dj in (0, 1):
        for di in (0, 1):
            nn[dj:dj + ny, di:di + nx] += eps; dn[dj:dj + ny, di:di + nx] += w
    return avg(nh, dh), avg(nv, dv), avg(nn, dn)


class VectorOperator:
    """Frequency-independent part of the eigenproblem for one grid (reuse across a frequency sweep)."""

    def __init__(self, g: Grid):
        if np.any(np.abs(g.mu[g.free] - 1.0) > 1e-12):
            raise ValueError("vector solver supports mu_r = 1 only")
        self.g = g
        ny, nx = g.ny, g.nx
        self.NHx, self.NHy = ny * (nx + 1), (ny + 1) * nx
        N = self.NHx + self.NHy
        self.N = N
        self.epsx, self.epsy, self.epsz = _edge_eps(g)
        NHx = self.NHx
        J, I = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        c = (J * nx + I).ravel(); J, I = J.ravel(), I.ravel()
        r = np.concatenate([c, c, c, c])
        k = np.concatenate([J * (nx + 1) + I + 1, J * (nx + 1) + I, NHx + (J + 1) * nx + I, NHx + J * nx + I])
        v = np.concatenate([1 / g.dxc[I], -1 / g.dxc[I], 1 / g.dyc[J], -1 / g.dyc[J]])
        self.DIV = sp.coo_matrix((v, (r, k)), shape=(ny * nx, N)).tocsr()
        # gradient of div onto Hx / Hy edges
        J, I = np.meshgrid(np.arange(ny), np.arange(1, nx), indexing="ij"); J, I = J.ravel(), I.ravel()
        row = J * (nx + 1) + I
        Gx = sp.coo_matrix((np.concatenate([1 / g.dxn[I], -1 / g.dxn[I]]),
                            (np.concatenate([row, row]), np.concatenate([J * nx + I, J * nx + I - 1]))),
                           shape=(NHx, ny * nx))
        J, I = np.meshgrid(np.arange(1, ny), np.arange(nx), indexing="ij"); J, I = J.ravel(), I.ravel()
        row = J * nx + I
        Gy = sp.coo_matrix((np.concatenate([1 / g.dyn[J], -1 / g.dyn[J]]),
                            (np.concatenate([row, row]), np.concatenate([J * nx + I, (J - 1) * nx + I]))),
                           shape=(self.NHy, ny * nx))
        G = sp.vstack([Gx, Gy]).tocsr()
        # curl_z at interior nodes
        J, I = np.meshgrid(np.arange(1, ny), np.arange(1, nx), indexing="ij"); J, I = J.ravel(), I.ravel()
        node = J * (nx + 1) + I
        rr = np.concatenate([node] * 4)
        kk = np.concatenate([NHx + J * nx + I, NHx + J * nx + I - 1, J * (nx + 1) + I, (J - 1) * (nx + 1) + I])
        vv = np.concatenate([1 / g.dxn[I], -1 / g.dxn[I], -1 / g.dyn[J], 1 / g.dyn[J]])
        self.CURL = sp.coo_matrix((vv, (rr, kk)), shape=((ny + 1) * (nx + 1), N)).tocsr()
        wz = np.where(g.pec_node, 0.0, 1.0 / self.epsz).ravel()
        self.wz = wz
        J, I = np.meshgrid(np.arange(ny), np.arange(nx + 1), indexing="ij"); J, I = J.ravel(), I.ravel()
        row = J * (nx + 1) + I
        Nx = sp.coo_matrix((np.concatenate([1 / g.dyc[J], -1 / g.dyc[J]]),
                            (np.concatenate([row, row]), np.concatenate([(J + 1) * (nx + 1) + I, J * (nx + 1) + I]))),
                           shape=(NHx, (ny + 1) * (nx + 1)))
        J, I = np.meshgrid(np.arange(ny + 1), np.arange(nx), indexing="ij"); J, I = J.ravel(), I.ravel()
        row = J * nx + I
        Ny = sp.coo_matrix((np.concatenate([1 / g.dxc[I], -1 / g.dxc[I]]),
                            (np.concatenate([row, row]), np.concatenate([J * (nx + 1) + I + 1, J * (nx + 1) + I]))),
                           shape=(self.NHy, (ny + 1) * (nx + 1)))
        T = sp.vstack([-sp.diags(self.epsy.ravel()) @ Nx, sp.diags(self.epsx.ravel()) @ Ny]) @ sp.diags(wz) @ self.CURL
        P = G @ self.DIV + T
        self.act = np.concatenate([(~g.pec_y).ravel(), (~g.pec_x).ravel()])
        a = np.where(self.act)[0]
        self.P = P.tocsr()[a][:, a].tocsc()
        self.eps_act = np.concatenate([self.epsy.ravel(), self.epsx.ravel()])[self.act]
        self.epsmax = float(g.eps[g.free].max())

    def matrix(self, k0):
        return (k0 ** 2 * sp.diags(self.eps_act) + self.P).tocsc()


@dataclass
class VMode:
    f: float
    beta: float
    neff: float
    kind: str
    long_ratio: float
    Hx: np.ndarray
    Hy: np.ndarray
    Ex: np.ndarray
    Ey: np.ndarray
    Ez: np.ndarray
    Hz: np.ndarray
    power: float
    h_act: np.ndarray = field(repr=False)
    z: dict = field(default_factory=dict)

    @property
    def vp(self):
        return C0 / self.neff


@dataclass
class VectorModeSet:
    op: VectorOperator
    f: float
    modes: list


def _fields(op: VectorOperator, k0: float, beta: float, h_act: np.ndarray):
    g = op.g
    ny, nx = g.ny, g.nx
    h = np.zeros(op.N); h[op.act] = h_act
    Hx = h[:op.NHx].reshape(ny, nx + 1); Hy = h[op.NHx:].reshape(ny + 1, nx)
    div = (op.DIV @ h).reshape(ny, nx)
    curl = (op.CURL @ h).reshape(ny + 1, nx + 1)
    ddy = np.zeros((ny + 1, nx)); ddy[1:-1, :] = (div[1:, :] - div[:-1, :]) / g.dyn[1:-1, None]
    ddx = np.zeros((ny, nx + 1)); ddx[:, 1:-1] = (div[:, 1:] - div[:, :-1]) / g.dxn[None, 1:-1]
    Ex = ETA0 / (k0 * op.epsx) * (beta * Hy - ddy / beta); Ex[g.pec_x] = 0
    Ey = -ETA0 / (k0 * op.epsy) * (beta * Hx - ddx / beta); Ey[g.pec_y] = 0
    Ez = -1j * ETA0 * curl * op.wz.reshape(ny + 1, nx + 1) / k0
    Hz = -1j * div / beta
    areaX = g.dxc[None, :] * g.dyn[:, None]; areaY = g.dxn[None, :] * g.dyc[:, None]
    P = 0.5 * (np.sum(Ex * Hy * areaX) - np.sum(Ey * Hx * areaY)).real
    return Hx, Hy, Ex, Ey, Ez, Hz, P


def _classify(op, Hx, Hy, Ex, Ey, Ez, Hz):
    g = op.g
    c = lambda a, b: 0.5 * (a + b)
    Ex_c, Ey_c = c(Ex[:-1], Ex[1:]), c(Ey[:, :-1], Ey[:, 1:])
    Hx_c, Hy_c = c(Hx[:, :-1], Hx[:, 1:]), c(Hy[:-1], Hy[1:])
    Ez_c = 0.25 * (Ez[:-1, :-1] + Ez[:-1, 1:] + Ez[1:, :-1] + Ez[1:, 1:])
    w = g.area * g.free
    e_t = np.sum(w * g.eps * (Ex_c ** 2 + Ey_c ** 2)); h_t = ETA0 ** 2 * np.sum(w * (Hx_c ** 2 + Hy_c ** 2))
    e_z = np.sum(w * g.eps * np.abs(Ez_c) ** 2); h_z = ETA0 ** 2 * np.sum(w * np.abs(Hz) ** 2)
    ratio = (e_z + h_z) / max(e_t + h_t, 1e-300)
    fe = e_z / max(e_z + h_z, 1e-300)
    if ratio < 0.05:
        kind = "quasi-TEM"
    else:
        kind = "TE-like" if fe < 0.2 else "TM-like" if fe > 0.8 else "hybrid"
    return kind, float(ratio)


def impedances(op: VectorOperator, m: VMode, group: int = 1, ring: int = 1) -> dict:
    """Z by P-I (always), P-V and V-I (voltage path from the conductor straight down to the bottom wall)."""
    g = op.g
    ny, nx = g.ny, g.nx
    nodes = np.zeros((ny + 1, nx + 1), bool)
    cells = g.owner == group
    for dj in (0, 1):
        for di in (0, 1):
            nodes[dj:dj + ny, di:di + nx] |= cells
    nodes[:-1, :-1] |= False
    wh = np.argwhere(g.wall_h == group)
    for j, i in wh:
        nodes[j, i] = nodes[j, i + 1] = True
    wv = np.argwhere(g.wall_v == group)
    for j, i in wv:
        nodes[j, i] = nodes[j + 1, i] = True
    if not nodes.any():
        return {}
    jj, ii = np.where(nodes)
    ja, jb, ia, ib = jj.min(), jj.max(), ii.min(), ii.max()
    out = {}
    I = np.nan
    if ja - ring >= 0 and jb + ring <= ny - 1 and ia - ring >= 1 - 1 and ib + ring <= nx - 1 and ia - ring >= 0:
        bot = np.sum(m.Hx[ja - ring, ia:ib + 1] * g.dxn[ia:ib + 1])
        top = np.sum(m.Hx[jb - 1 + ring, ia:ib + 1] * g.dxn[ia:ib + 1])
        rgt = np.sum(m.Hy[ja:jb + 1, ib + ring - 1 + 1 - 1 + 0] * g.dyn[ja:jb + 1]) if False else np.sum(m.Hy[ja:jb + 1, ib + ring - 1] * g.dyn[ja:jb + 1])
        lft = np.sum(m.Hy[ja:jb + 1, ia - ring] * g.dyn[ja:jb + 1])
        I = float(bot + rgt - top - lft)
    P = m.power
    if np.isfinite(I) and abs(I) > 0:
        out["Z_PI"] = 2 * P / I ** 2
    ic = int(np.argmin(abs(g.xn[ia:ib + 1] - 0.5 * (g.xn[ia] + g.xn[ib])))) + ia
    if ja >= 1 and not g.pec_y[:ja, ic].any():
        V = float(np.sum(m.Ey[:ja, ic] * g.dyc[:ja]))
        out["V"] = abs(V)
        out["Z_PV"] = V ** 2 / (2 * P)
        if np.isfinite(I) and abs(I) > 0:
            out["Z_VI"] = abs(V / I)
    out["I"] = abs(I) if np.isfinite(I) else np.nan
    return out


def solve_vector(cs_or_op, f: float, n_modes: int = 4, dx: float = 5.0, edge_dx: float | None = None,
                 want_z: bool = True) -> VectorModeSet:
    op = cs_or_op if isinstance(cs_or_op, VectorOperator) else VectorOperator(build_grid(cs_or_op, dx, edge_dx))
    k0 = 2 * np.pi * f / C0 * 1e-3          # 1/mm
    A = op.matrix(k0)
    k = min(n_modes + 6, A.shape[0] - 2)
    vals, vecs = spla.eigs(A, k=k, sigma=1.02 * k0 ** 2 * op.epsmax, which="LM")
    order = np.argsort(-vals.real)
    modes = []
    for o in order:
        b2 = vals[o]
        if abs(b2.imag) > 1e-6 * abs(b2.real) or b2.real <= 0:
            continue
        vec = vecs[:, o]
        vec = vec * np.exp(-1j * np.angle(vec[np.argmax(abs(vec))]))
        if np.max(abs(vec.imag)) > 1e-6 * np.max(abs(vec.real)):
            continue
        h = vec.real / np.linalg.norm(vec.real)
        beta = float(np.sqrt(b2.real))          # 1/mm
        Hx, Hy, Ex, Ey, Ez, Hz, P = _fields(op, k0, beta, h)
        if P < 0:
            h = -h; Hx, Hy, Ex, Ey, Ez, Hz, P = _fields(op, k0, beta, h)
        if P <= 0:
            continue
        s = 1.0 / np.sqrt(P)                    # normalise to 1 W
        h = h * s
        Hx, Hy, Ex, Ey, Ez, Hz = (a * s for a in (Hx, Hy, Ex, Ey, Ez, Hz))
        kind, ratio = _classify(op, Hx, Hy, Ex, Ey, Ez, Hz)
        m = VMode(f, beta * 1e3, beta / k0, kind, ratio, Hx, Hy, Ex, Ey, Ez, Hz, 1.0, h)
        if want_z and op.g.groups:
            m.z = impedances(op, m, op.g.groups[0])
        modes.append(m)
        if len(modes) >= n_modes:
            break
    return VectorModeSet(op, f, modes)


def dispersion(cs: CrossSection, freqs, dx: float = 5.0, edge_dx: float | None = None, n_track: int = 1):
    """Track the n_track highest-beta modes found at the first frequency across `freqs` (overlap tracking)."""
    op = VectorOperator(build_grid(cs, dx, edge_dx))
    freqs = np.asarray(freqs, float)
    prev, rows = None, []
    for f in freqs:
        ms = solve_vector(op, f, n_modes=n_track + 3)
        cur = ms.modes
        if prev is None:
            sel = cur[:n_track]
        else:
            sel = []
            for pm in prev:
                ov = [abs(pm.h_act @ m.h_act) / (np.linalg.norm(pm.h_act) * np.linalg.norm(m.h_act)) for m in cur]
                sel.append(cur[int(np.argmax(ov))] if ov and max(ov) > 0.5 else None)
        rows.append(sel)
        prev = [m if m is not None else p for m, p in zip(sel, prev if prev else sel)]
    out = []
    for t in range(n_track):
        beta = np.array([r[t].beta if r[t] else np.nan for r in rows])
        neff = np.array([r[t].neff if r[t] else np.nan for r in rows])
        ok = np.isfinite(beta)
        vg = np.full_like(beta, np.nan)
        if ok.sum() > 2:
            vg[ok] = np.gradient(2 * np.pi * freqs[ok], beta[ok])
        zs = {key: np.array([r[t].z.get(key, np.nan) if r[t] else np.nan for r in rows])
              for key in ("Z_PI", "Z_PV", "Z_VI")}
        out.append({"f": freqs, "beta": beta, "neff": neff, "vp": C0 / neff, "vg": vg, "kind": [r[t].kind if r[t] else None for r in rows], **zs})
    return out
