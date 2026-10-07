"""Full-vector finite-difference mode solver on a 2.5D Yee grid (hybrid modes, any eps(x,y), PEC conductors).

Unknowns: transverse magnetic field (Hx, Hy). For fixed frequency the eigenproblem is  beta^2 H = A(k0) H
(Zhu & Brown type formulation, derived directly from the discrete Yee curl operators, so PEC walls,
thin conductors and dielectric interfaces are handled consistently and no spurious modes appear).
Lengths are metres inside this module; non-magnetic media (mu_r = 1) only.

Yee locations (cell (i,j), node (i,j)):  Ez nodes; Ex,Hy at x-edges; Ey,Hx at y-edges; Hz cell centres.
Ex and Hy (and Ey and Hx) share their location, so the Poynting flux is evaluated edge by edge.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from .geometry import CrossSection
from .solver import C0, ETA0, parity, _even

log = logging.getLogger(__name__)
EPS0 = 8.8541878128e-12


class FullWaveGrid:
    def __init__(self, cs: CrossSection, dx_target: float):
        self.cs = cs
        nx, ny = _even(round(cs.width / dx_target)), _even(round(cs.height / dx_target))
        self.nx, self.ny = nx, ny
        self.dx, self.dy = cs.width / nx * 1e-3, cs.height / ny * 1e-3
        eps, td = cs.eps_cells(nx, ny)
        self.eps_cell, self.tand_cell = eps, td
        code = cs.pec_nodes(nx, ny)
        self.code = code
        same = lambda a, b: (a == b) & (a != 0)
        self.free_ex = ~same(code[:, :-1], code[:, 1:])          # (ny+1, nx)
        self.free_ey = ~same(code[:-1, :], code[1:, :])          # (ny, nx+1)
        self.free_ez = code == 0                                 # (ny+1, nx+1)

        def avg(a):
            p = np.pad(a, 1, mode="edge")
            ax = 0.5 * (p[:-1, 1:-1] + p[1:, 1:-1])
            ay = 0.5 * (p[1:-1, :-1] + p[1:-1, 1:])
            az = 0.25 * (p[:-1, :-1] + p[:-1, 1:] + p[1:, :-1] + p[1:, 1:])
            return ax, ay, az
        self.eps_x, self.eps_y, self.eps_z = avg(eps)
        self.etd_x, self.etd_y, self.etd_z = avg(eps * td)
        dxm, dyd = self.dx, self.dy
        Dx = sp.diags([-np.ones(nx), np.ones(nx)], [0, 1], shape=(nx, nx + 1)) / dxm
        Dy = sp.diags([-np.ones(ny), np.ones(ny)], [0, 1], shape=(ny, ny + 1)) / dyd
        Ix, Iy, Ixn, Iyn = (sp.identity(k) for k in (nx, ny, nx + 1, ny + 1))
        self.Px, self.Py = sp.kron(Iy, Dx).tocsr(), sp.kron(Dy, Ix).tocsr()
        self.Gx, self.Gy = sp.kron(Iy, -Dx.T).tocsr(), sp.kron(-Dy.T, Ix).tocsr()
        self.Dxm, self.Dym = sp.kron(Iyn, -Dx.T).tocsr(), sp.kron(-Dy.T, Ixn).tocsr()
        Dyp, Dxp = sp.kron(Dy, Ixn).tocsr(), sp.kron(Iyn, Dx).tocsr()
        invEz = sp.diags(self.free_ez.ravel() / self.eps_z.ravel())
        Ey_, Ex_ = sp.diags(self.eps_y.ravel()), sp.diags(self.eps_x.ravel())
        Axx = self.Gx @ self.Px + Ey_ @ Dyp @ invEz @ self.Dym
        Axy = self.Gx @ self.Py - Ey_ @ Dyp @ invEz @ self.Dxm
        Ayx = self.Gy @ self.Px - Ex_ @ Dxp @ invEz @ self.Dym
        Ayy = self.Gy @ self.Py + Ex_ @ Dxp @ invEz @ self.Dxm
        self.ix = np.flatnonzero(self.free_ey.ravel())           # free Hx unknowns
        self.iy = np.flatnonzero(self.free_ex.ravel())           # free Hy unknowns
        ix, iy = self.ix, self.iy
        self.A0 = sp.bmat([[Axx[ix][:, ix], Axy[ix][:, iy]], [Ayx[iy][:, ix], Ayy[iy][:, iy]]]).tocsc()
        self.deps = np.concatenate([self.eps_y.ravel()[ix], self.eps_x.ravel()[iy]])
        self.nunk = len(ix) + len(iy)
        self._invEz = invEz
        sx = np.array_equal(code, code[:, ::-1]) and np.allclose(eps, eps[:, ::-1])
        sy = np.array_equal(code, code[::-1, :]) and np.allclose(eps, eps[::-1, :])
        self.geom_sym = (bool(sx), bool(sy))

    # ------------------------------------------------------------------ eigen solve
    def eigs(self, f: float, k: int = 4, v0=None):
        """Highest beta^2 eigenpairs at frequency f (Hz). Returns (beta2 array desc, vectors)."""
        k0 = 2 * np.pi * f / C0
        A = (self.A0 + sp.diags(k0 ** 2 * self.deps)).tocsc()
        sigma = 1.05 * k0 ** 2 * float(self.eps_cell.max()) + 1e-12
        kk = min(k, self.nunk - 2)
        if v0 is None:
            v0 = np.ones(self.nunk)
        w, v = spla.eigs(A, k=kk, sigma=sigma, which="LM", v0=v0)
        w, v = w.real, v.real if np.abs(v.imag).max() < 1e-6 * np.abs(v.real).max() else v
        o = np.argsort(-w)
        return w[o], v[:, o]

    # ------------------------------------------------------------------ fields
    def fields(self, h: np.ndarray, beta: float, f: float):
        """Complex Yee-grid fields for eigenvector h; scaled so that the guided power is 1 W/m."""
        w = 2 * np.pi * f
        nx, ny = self.nx, self.ny
        Hx, Hy = np.zeros(ny * (nx + 1), complex), np.zeros((ny + 1) * nx, complex)
        Hx[self.ix], Hy[self.iy] = h[:len(self.ix)], h[len(self.ix):]
        Hz = (self.Px @ Hx + self.Py @ Hy) / (1j * beta)
        pre = 1.0 / (1j * w * EPS0)
        Ey = pre * self.free_ey.ravel() / self.eps_y.ravel() * (-(self.Gx @ Hz) - 1j * beta * Hx)
        Ex = pre * self.free_ex.ravel() / self.eps_x.ravel() * (self.Gy @ Hz + 1j * beta * Hy)
        Ez = pre * (self._invEz @ (self.Dxm @ Hy - self.Dym @ Hx))
        F = dict(Hx=Hx.reshape(ny, nx + 1), Hy=Hy.reshape(ny + 1, nx), Hz=Hz.reshape(ny, nx),
                 Ex=Ex.reshape(ny + 1, nx), Ey=Ey.reshape(ny, nx + 1), Ez=Ez.reshape(ny + 1, nx + 1))
        P = self.power(F)
        s = 1.0 / np.sqrt(abs(P)) if abs(P) > 0 else 1.0
        return {k: v * s for k, v in F.items()}

    def power(self, F) -> float:
        a = self.dx * self.dy
        return 0.5 * a * float(np.sum((F["Ex"] * np.conj(F["Hy"])).real) - np.sum((F["Ey"] * np.conj(F["Hx"])).real))

    def e_cell(self, F):
        """Transverse E at cell centres, phase-rotated to real, for parity analysis."""
        Ex = 0.5 * (F["Ex"][:-1, :] + F["Ex"][1:, :])
        Ey = 0.5 * (F["Ey"][:, :-1] + F["Ey"][:, 1:])
        big = Ex if np.abs(Ex).max() > np.abs(Ey).max() else Ey
        ph = np.exp(-1j * np.angle(big.flat[np.argmax(np.abs(big))]))
        return (Ex * ph).real, (Ey * ph).real

    def signal_current(self, F, group: int):
        nodes = np.argwhere(self.code == group)
        if len(nodes) == 0:
            return np.nan, None
        ja, ia = nodes.min(axis=0)
        jb, ib = nodes.max(axis=0)
        ra, rb = max(ja - 1, 0), min(jb, self.ny - 1)
        ca, cb = max(ia - 1, 0), min(ib, self.nx - 1)
        Hx, Hy = F["Hx"], F["Hy"]
        I = (Hx[ra, ca + 1:cb + 1].sum() * self.dx + Hy[ra + 1:rb + 1, cb].sum() * self.dy
             - Hx[rb, ca + 1:cb + 1].sum() * self.dx - Hy[ra + 1:rb + 1, ca].sum() * self.dy)
        return abs(I), (ja, jb, ia, ib)

    def voltage(self, F, bbox):
        """Line integral of Ey from the conductor straight down to ground (falls back to going up)."""
        ja, jb, ia, ib = bbox
        ic = (ia + ib) // 2
        col = self.code[:, ic]
        for direction in (-1, 1):
            j = ja if direction < 0 else jb
            path = []
            while 0 <= j + direction <= self.ny:
                path.append(j + (0 if direction > 0 else -1))
                j += direction
                if col[j] == -1:
                    return abs(sum(F["Ey"][p, ic] for p in path) * self.dy)
        return np.nan

    def loss_dielectric(self, F, f: float) -> float:
        a = self.dx * self.dy
        pl = 0.5 * 2 * np.pi * f * EPS0 * a * (np.sum(self.etd_x * np.abs(F["Ex"]) ** 2) + np.sum(self.etd_y * np.abs(F["Ey"]) ** 2)
                                               + np.sum(self.etd_z * np.abs(F["Ez"]) ** 2))
        return pl                              # W/m for P = 1 W/m

    def classify(self, F, homogeneous: bool, has_signal: bool) -> str:
        eE = (np.sum(self.eps_x * np.abs(F["Ex"]) ** 2) + np.sum(self.eps_y * np.abs(F["Ey"]) ** 2))
        eZ = np.sum(self.eps_z * np.abs(F["Ez"]) ** 2)
        hT = np.sum(np.abs(F["Hx"]) ** 2) + np.sum(np.abs(F["Hy"]) ** 2)
        hZ = np.sum(np.abs(F["Hz"]) ** 2)
        fe, fh = eZ / (eE + eZ + 1e-300), hZ / (hT + hZ + 1e-300)
        if has_signal and max(fe, fh) < 0.02:
            return "TEM" if homogeneous else "quasi-TEM"
        if has_signal and max(fe, fh) < 0.15 and fe > 0 and fh >= 0 and np.sum(np.abs(F["Hy"])) > 0 and False:
            return "quasi-TEM"
        if fe < 0.2 * fh:
            return "TE-like"
        if fh < 0.2 * fe:
            return "TM-like"
        return "hybrid"


@dataclass
class Dispersion:
    freqs: np.ndarray
    beta: np.ndarray            # (nf, nm) rad/m, NaN where mode is below cutoff
    neff: np.ndarray
    vp: np.ndarray
    vg: np.ndarray
    lam_g: np.ndarray
    z0_pi: np.ndarray           # 2P/|I|^2
    z0_vi: np.ndarray           # |V|/|I|
    alpha_d: np.ndarray         # Np/m
    kinds: list                 # per track, kind at first appearance
    symmetry: list
    confidence: list
    fields: dict = field(default_factory=dict)   # (freq index, track) -> dict of Yee fields (only if keep_fields)
    grid: FullWaveGrid | None = None

    def table(self, fi: int | None = None) -> str:
        fi = len(self.freqs) // 2 if fi is None else fi
        f = self.freqs[fi]
        out = [f"f = {f / 1e9:.4f} GHz",
               "Track | Type      | Symmetry  | beta, rad/m | neff   | vp/c   | vg/c   | Z0(P,I) | Z0(V,I) | alpha_d, dB/m"]
        for m in range(self.beta.shape[1]):
            b = self.beta[fi, m]
            if np.isnan(b):
                continue
            out.append(f"{m:5d} | {self.kinds[m]:9s} | {self.symmetry[m]:9s} | {b:11.4f} | {self.neff[fi, m]:6.3f} | "
                       f"{self.vp[fi, m] / C0:6.3f} | {self.vg[fi, m] / C0:6.3f} | {self.z0_pi[fi, m]:7.2f} | "
                       f"{self.z0_vi[fi, m]:7.2f} | {8.686 * self.alpha_d[fi, m]:.4f}")
        return "\n".join(out)


def dispersion(cs: CrossSection, freqs, dx: float = 1.0, n_modes: int = 4, keep_fields_at=(), grid=None) -> Dispersion:
    """beta(f), neff, vp, vg, lambda_g, Z0(f), dielectric attenuation for the n_modes highest-beta modes.
    Modes are tracked across frequency by eigenvector overlap. keep_fields_at: indices of freqs to store Yee fields."""
    g = grid or FullWaveGrid(cs, dx)
    freqs = np.asarray(freqs, float)
    nf = len(freqs)
    signal = sorted({c.group for c in cs.conductors if c.group > 0})
    sig = signal[0] if signal else None
    tracks = []                 # each: dict(last=vector, data={fi: ...})
    for fi, f in enumerate(freqs):
        w, V = g.eigs(f, k=n_modes + 2)
        sel = [i for i in range(len(w)) if w[i] > 1e-9 * (2 * np.pi * f / C0) ** 2]
        sel = sel[:n_modes + 2]
        used = set()
        # match to existing tracks by overlap
        cand = []
        for ti, tr in enumerate(tracks):
            if tr["last_f"] != fi - 1:
                continue
            for i in sel:
                v = V[:, i] / np.linalg.norm(V[:, i])
                cand.append((abs(float(tr["last"] @ v)), ti, i))
        cand.sort(reverse=True)
        assign, taken_t = {}, set()
        for ov, ti, i in cand:
            if ov < 0.5 or ti in taken_t or i in assign:
                continue
            assign[i] = (ti, ov); taken_t.add(ti)
        for i in sel:
            v = V[:, i] / np.linalg.norm(V[:, i])
            if i in assign:
                ti, ov = assign[i]
            else:
                tracks.append(dict(last=v, last_f=-2, data={}, first=fi, ov_min=1.0)); ti, ov = len(tracks) - 1, 1.0
            tr = tracks[ti]
            beta = float(np.sqrt(w[i]))
            F = g.fields(V[:, i], beta, f)
            I, bbox = g.signal_current(F, sig) if sig is not None else (np.nan, None)
            P = g.power(F)
            z_pi = 2 * P / I ** 2 if sig is not None and I > 0 else np.nan
            Vv = g.voltage(F, bbox) if bbox is not None else np.nan
            z_vi = Vv / I if sig is not None and I > 0 else np.nan
            ex, ey = g.e_cell(F)
            kind = g.classify(F, cs.homogeneous, sig is not None)
            tr["data"][fi] = dict(beta=beta, z_pi=z_pi, z_vi=z_vi, ad=g.loss_dielectric(F, f) / (2 * P),
                                  kind=kind, sym=f"{parity(ex, ey, 'x')}/{parity(ex, ey, 'y')}")
            tr["ov_min"] = min(tr["ov_min"], ov)
            tr["last"], tr["last_f"] = v, fi
            if fi in keep_fields_at:
                tr.setdefault("fields", {})[fi] = F
    tracks.sort(key=lambda t: (t["first"], -t["data"][t["first"]]["beta"]))
    nm = len(tracks)
    arr = lambda: np.full((nf, nm), np.nan)
    beta, z_pi, z_vi, ad = arr(), arr(), arr(), arr()
    kinds, syms, conf, flds = [], [], [], {}
    for m, tr in enumerate(tracks):
        for fi, d in tr["data"].items():
            beta[fi, m], z_pi[fi, m], z_vi[fi, m], ad[fi, m] = d["beta"], d["z_pi"], d["z_vi"], d["ad"]
        d0 = tr["data"][tr["first"]]
        kinds.append(d0["kind"]); syms.append(d0["sym"])
        conf.append("High" if tr["ov_min"] > 0.95 else "Medium" if tr["ov_min"] > 0.8 else "Low")
        for fi, F in tr.get("fields", {}).items():
            flds[(fi, m)] = F
    w = 2 * np.pi * freqs
    vp, neff = w[:, None] / beta, C0 * beta / w[:, None]
    vg = np.full_like(beta, np.nan)
    for m in range(nm):
        ok = ~np.isnan(beta[:, m])
        if ok.sum() >= 3:
            vg[ok, m] = 1.0 / np.gradient(beta[ok, m], w[ok])
    return Dispersion(freqs, beta, neff, vp, vg, 2 * np.pi / beta, z_pi, z_vi, ad, kinds, syms, conf, flds, g)
