"""STL (and other mesh) -> 2D CrossSection recognition.

Pipeline: load bodies -> detect propagation axis -> slice every body with the plane normal to it ->
assign roles (ground/shield, signal conductor, dielectric) -> rasterise on a grid that contains all
polygon vertex lines -> merge cells into rectangles -> CrossSection for modeapp.solver / modeapp.vector.

Conventions: lengths in mm. In-plane axes (u,v) are the cyclic ones: axis x -> (y,z), y -> (z,x), z -> (x,y).
The domain is a PEC box; a body with role 'ground' (the shield/enclosure) is PEC, tied to the box, and
everything outside the shield's outer outline is filled with ground as well.
"""
from __future__ import annotations
import fnmatch
import logging
import os
import re
from dataclasses import dataclass, field
import numpy as np
import trimesh
from shapely import contains_xy
from shapely.geometry import Polygon, MultiPolygon, box as sbox
from shapely.ops import unary_union
from .geometry import CrossSection, Rect, Dielectric
from .grid import _uniq, _axis

log = logging.getLogger(__name__)
UNITS = {"mm": 1.0, "m": 1e3, "cm": 10.0, "um": 1e-3, "in": 25.4}
AXES = "xyz"
_DIEL_HINT = re.compile(r"(sub|diel|fr-?4|ro\d{4}|teflon|ptfe|ceram|alumina|quartz|glass|foam|insul)", re.I)


@dataclass
class Body:
    name: str
    mesh: trimesh.Trimesh
    role: str | None = None          # ground | conductor | dielectric | ignore
    eps_r: float = 1.0
    mu_r: float = 1.0
    group: int | None = None
    reason: str = ""


@dataclass
class ImportedStructure:
    cs: CrossSection
    axis: int
    position: float
    length: float
    domain: tuple                    # (u0, v0, u1, v1) in model coordinates, mm
    bodies: list
    fidelity: dict
    uniformity: dict
    warnings: list = field(default_factory=list)
    raster_cells: tuple = (0, 0)

    def summary(self) -> str:
        u, v = _plane_axes(self.axis)
        L = ["Recognised structure",
             f"  propagation axis : {AXES[self.axis]} (length {self.length:.3f} mm), cross-section plane at {AXES[self.axis]} = {self.position:.3f} mm",
             f"  in-plane axes    : u={AXES[u]}, v={AXES[v]}; domain {self.domain[2] - self.domain[0]:.4g} x {self.domain[3] - self.domain[1]:.4g} mm "
             f"(PEC box), raster {self.raster_cells[0]}x{self.raster_cells[1]} cells",
             "  bodies:"]
        for b in self.bodies:
            extra = f", eps_r={b.eps_r:g}" if b.role == "dielectric" else (f", group {b.group}" if b.role == "conductor" else "")
            fid = self.fidelity.get(b.name)
            L.append(f"    {b.name:24s} -> {b.role}{extra}   [{b.reason}]" + (f"  raster area error {100 * fid:.2f} %" if fid is not None else ""))
        un = self.uniformity
        if un:
            L.append(f"  translation invariance along {AXES[self.axis]}: max section change {100 * un['max_change']:.2f} % "
                     f"({'uniform' if un['uniform'] else 'NOT uniform'})")
        for w in self.warnings:
            L.append(f"  WARNING: {w}")
        return "\n".join(L)


def _plane_axes(axis: int):
    return {0: (1, 2), 1: (2, 0), 2: (0, 1)}[axis]


# ------------------------------------------------------------------ loading
def load_bodies(sources, units: str = "mm", split: bool = True) -> list:
    """sources: path / list of paths / directory / Trimesh / {name: Trimesh}. One body per file; a multi-component
    file is split into bodies if `split` (touching bodies sharing vertices cannot be separated: use one file per body)."""
    if isinstance(sources, (str, os.PathLike)):
        sources = [sources]
    if isinstance(sources, trimesh.Trimesh):
        sources = {"body": sources}
    items = []
    if isinstance(sources, dict):
        items = list(sources.items())
    else:
        paths = []
        for s in sources:
            s = str(s)
            if os.path.isdir(s):
                paths += sorted(os.path.join(s, f) for f in os.listdir(s) if f.lower().endswith((".stl", ".obj", ".ply", ".3mf", ".glb")))
            else:
                paths.append(s)
        for p in paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            obj = trimesh.load(p)
            if isinstance(obj, trimesh.Scene):
                for name, g in obj.geometry.items():
                    if isinstance(g, trimesh.Trimesh):
                        items.append((f"{stem}:{name}", g))
            elif isinstance(obj, trimesh.Trimesh):
                items.append((stem, obj))
    bodies = []
    sc = UNITS[units]
    for name, m in items:
        m = m.copy()
        m.apply_scale(sc)
        parts = m.split(only_watertight=False) if split and m.body_count > 1 else [m]
        for k, part in enumerate(parts):
            bodies.append(Body(name if len(parts) == 1 else f"{name}_{k + 1}", part))
    if not bodies:
        raise ValueError("no meshes found")
    return bodies


# ------------------------------------------------------------------ geometry
def _stability(bodies, ax, lo, hi, fracs=(0.2, 0.5, 0.8)):
    """Max relative change of the body sections along axis `ax` (0 = perfectly prismatic along it)."""
    ref, worst = None, 0.0
    for fr in (fracs[1], fracs[0], fracs[2]):
        p = float(lo[ax] + fr * (hi[ax] - lo[ax]))
        cur = [slice_body(b.mesh, ax, p) for b in bodies]
        if ref is None:
            ref = cur
            if any(g is None for g in cur):
                return 1.0
            continue
        for g0, g1 in zip(ref, cur):
            worst = max(worst, 1.0 if g1 is None else g0.symmetric_difference(g1).area / max(g0.area, 1e-12))
    return worst


def detect_axis(bodies, tol=0.01):
    """Propagation axis = axis along which every body is prismatic; ties are resolved by the longest extent."""
    lo = np.min([b.mesh.bounds[0] for b in bodies], axis=0)
    hi = np.max([b.mesh.bounds[1] for b in bodies], axis=0)
    ext = hi - lo
    order = [int(a) for a in np.argsort(ext)[::-1]]
    scores = {a: _stability(bodies, a, lo, hi) for a in order}
    uniform = [a for a in order if scores[a] <= tol]
    axis = uniform[0] if uniform else min(order, key=lambda a: scores[a])
    conf = "High" if len(uniform) == 1 else ("Low" if len(uniform) > 1 else "None")
    return axis, conf, scores, lo, hi


def slice_body(mesh: trimesh.Trimesh, axis: int, pos: float):
    """Cross-section of one body as a shapely geometry in (u,v) coordinates, or None if the plane misses it."""
    u, v = _plane_axes(axis)
    origin = np.zeros(3); origin[axis] = pos
    normal = np.zeros(3); normal[axis] = 1.0
    sec = mesh.section(plane_origin=origin, plane_normal=normal)
    if sec is None:
        return None
    M = np.zeros((4, 4)); M[0, u] = 1; M[1, v] = 1; M[2, axis] = 1; M[3, 3] = 1
    planar, _ = sec.to_2D(to_2D=M)
    polys = [p.buffer(0) for p in planar.polygons_full if p.area > 1e-12]
    if not polys:
        return None
    return unary_union(polys)


def _polys(g):
    return [] if g is None else list(g.geoms) if isinstance(g, MultiPolygon) else [g]


def _rectilinear(poly: Polygon, tol=1e-6) -> bool:
    for ring in [poly.exterior, *poly.interiors]:
        c = np.asarray(ring.coords)
        d = np.abs(np.diff(c, axis=0))
        if np.any((d[:, 0] > tol) & (d[:, 1] > tol)):
            return False
    return True


def _runs(row):
    out, i, n = [], 0, len(row)
    while i < n:
        if row[i]:
            j = i
            while j < n and row[j]:
                j += 1
            out.append((i, j)); i = j
        else:
            i += 1
    return out


def mask_to_rects(mask, xn, yn):
    """Greedy decomposition of a cell mask into rectangles (x0,y0,x1,y1)."""
    open_runs, out = {}, []
    for j in range(mask.shape[0]):
        new = {}
        for run in _runs(mask[j]):
            new[run] = open_runs.pop(run) if run in open_runs else j
        for run, j0 in open_runs.items():
            out.append((xn[run[0]], yn[j0], xn[run[1]], yn[j]))
        open_runs = new
    for run, j0 in open_runs.items():
        out.append((xn[run[0]], yn[j0], xn[run[1]], yn[mask.shape[0]]))
    return out


# ------------------------------------------------------------------ roles
def _parse_role(spec):
    """'ground' | 'conductor' | 'conductor:2' | 'dielectric:9.8' | ('dielectric', 9.8) | 'ignore'"""
    if isinstance(spec, (tuple, list)):
        kind, val = spec[0], spec[1] if len(spec) > 1 else None
    else:
        kind, _, val = str(spec).partition(":")
        val = val or None
    kind = kind.strip().lower()
    if kind in ("shield", "enclosure", "gnd"):
        kind = "ground"
    if kind not in ("ground", "conductor", "dielectric", "ignore"):
        raise ValueError(f"unknown role '{spec}'")
    return kind, (float(val) if val is not None else None)


def assign_roles(bodies, sections, roles=None):
    roles = roles or {}
    for b in bodies:
        spec = next((roles[k] for k in roles if k == b.name or fnmatch.fnmatch(b.name, k)), None)
        if spec is not None:
            kind, val = _parse_role(spec)
            b.role, b.reason = kind, "user"
            if kind == "dielectric":
                if val is None:
                    raise ValueError(f"body '{b.name}': dielectric needs eps_r, e.g. {b.name}=dielectric:4.4")
                b.eps_r = val
            if kind == "conductor" and val is not None:
                b.group = int(val)
    for b in bodies:
        if b.role:
            continue
        g = sections.get(b.name)
        others = [sections[o.name] for o in bodies if o is not b and sections.get(o.name) is not None]
        shield = False
        if g is not None:
            for p in _polys(g):
                for ring in p.interiors:
                    hole = Polygon(ring)
                    if all(hole.buffer(1e-6).contains(o) for o in others):
                        shield = True
        if shield:
            b.role, b.reason = "ground", "auto: closed outline enclosing all other bodies"
        elif _DIEL_HINT.search(b.name):
            b.role, b.reason = "dielectric", "auto: name hint (eps_r must be given!)"
        else:
            b.role, b.reason = "conductor", "auto: default"
    for b in bodies:
        if b.role == "dielectric" and "auto" in b.reason:
            raise ValueError(f"body '{b.name}' looks like a dielectric but eps_r is unknown: pass roles={{'{b.name}': 'dielectric:<eps_r>'}}")
    gid = 0
    for b in bodies:
        if b.role == "conductor" and b.group is None:
            gid += 1
            while any(o.group == gid for o in bodies if o is not b):
                gid += 1
            b.group = gid


# ------------------------------------------------------------------ main entry
def import_structure(sources, roles: dict | None = None, units: str = "mm", axis: str | int = "auto",
                     position: float = 0.5, raster_dx: float | None = None, pad: float | None = None,
                     domain: tuple | None = None, split: bool = True, uniform_tol: float = 0.01) -> ImportedStructure:
    """position: fraction (0..1) of the structure length along the axis if <=1 and axis is auto-sized; use
    `position_mm` semantics by passing a value > 1 (absolute mm in model coordinates)."""
    bodies = load_bodies(sources, units, split)
    warns = []
    for b in bodies:
        if not b.mesh.is_watertight:
            warns.append(f"mesh '{b.name}' is not watertight; sections may be incomplete")
    ax, conf, scores, lo, hi = detect_axis(bodies)
    if axis != "auto":
        ax = AXES.index(axis) if isinstance(axis, str) else int(axis)
    elif conf == "Low":
        cands = [AXES[a] for a, sc in scores.items() if sc <= 0.01]
        warns.append(f"several axes are prismatic ({', '.join(cands)}); chose {AXES[ax]} (longest) - set axis= explicitly if wrong")
    elif conf == "None":
        warns.append(f"no axis along which all bodies are prismatic; using {AXES[ax]} (smallest change {100 * scores[ax]:.1f} %)")
    length = float(hi[ax] - lo[ax])
    pos = float(lo[ax] + position * length) if position <= 1.0 else float(position)
    u, v = _plane_axes(ax)
    sections = {b.name: slice_body(b.mesh, ax, pos) for b in bodies}
    for b in bodies:
        if sections[b.name] is None:
            warns.append(f"body '{b.name}' does not intersect the plane at {AXES[ax]}={pos:.3f} mm; ignored")
            b.role, b.reason = "ignore", "no section"
    bodies = [b for b in bodies if b.role != "ignore"]
    active = {b.name: sections[b.name] for b in bodies}
    assign_roles(bodies, sections, roles)
    bodies = [b for b in bodies if b.role != "ignore"]
    # uniformity along the axis
    un = {"max_change": 0.0, "uniform": True, "positions": []}
    for fr in (0.1, 0.3, 0.7, 0.9):
        p2 = float(lo[ax] + fr * length)
        worst = 0.0
        for b in bodies:
            g1, g0 = slice_body(b.mesh, ax, p2), active[b.name]
            if g1 is None:
                worst = max(worst, 1.0); continue
            worst = max(worst, g0.symmetric_difference(g1).area / max(g0.area, 1e-12))
        un["positions"].append((fr, worst)); un["max_change"] = max(un["max_change"], worst)
    un["uniform"] = un["max_change"] <= uniform_tol
    if not un["uniform"]:
        warns.append("cross-section changes along the axis: a 2D mode analysis is valid only for the chosen plane")
    # domain
    shield = [b for b in bodies if b.role == "ground"]
    allg = unary_union([active[b.name] for b in bodies])
    if domain is not None:
        d0, d1, d2, d3 = domain
    elif shield:
        d0, d1, d2, d3 = unary_union([active[b.name] for b in shield]).bounds
    else:
        d0, d1, d2, d3 = allg.bounds
        p = pad if pad is not None else 0.5 * max(d2 - d0, d3 - d1)
        d0, d1, d2, d3 = d0 - p, d1 - p, d2 + p, d3 + p
        warns.append(f"no enclosure body in the model: added a PEC box with {p:.3g} mm padding around the structure")
    W, H = d2 - d0, d3 - d1
    # raster grid: all vertex lines of rectilinear polygons + bbox lines of curved ones + uniform fill
    rdx = raster_dx or max(W, H) / 200.0
    xs, ys = [d0, d2], [d1, d3]
    for b in bodies:
        for p in _polys(active[b.name]):
            if _rectilinear(p):
                for ring in [p.exterior, *p.interiors]:
                    c = np.asarray(ring.coords); xs += list(c[:, 0]); ys += list(c[:, 1])
            else:
                bx0, by0, bx1, by1 = p.bounds
                xs += [bx0, bx1]; ys += [by0, by1]
    tol = 1e-7 * max(W, H)
    clip = lambda a, lo_, hi_: [float(min(max(t, lo_), hi_)) for t in a]
    xn = _axis(_uniq(clip(xs, d0, d2), tol), [], rdx, None, 1.4, tol)
    yn = _axis(_uniq(clip(ys, d1, d3), tol), [], rdx, None, 1.4, tol)
    xc, yc = 0.5 * (xn[1:] + xn[:-1]), 0.5 * (yn[1:] + yn[:-1])
    XC, YC = np.meshgrid(xc, yc)
    cell_area = np.outer(np.diff(yn), np.diff(xn))
    nyr, nxr = len(yc), len(xc)
    eps = np.ones((nyr, nxr)); mu = np.ones((nyr, nxr))
    kind = np.zeros((nyr, nxr), int)      # 0 none, -1 ground, >0 conductor group
    fid = {}
    for b in sorted([b for b in bodies if b.role == "dielectric"], key=lambda b: -active[b.name].area):
        m = contains_xy(active[b.name], XC, YC)
        eps[m], mu[m] = b.eps_r, b.mu_r
        fid[b.name] = abs(cell_area[m].sum() - active[b.name].area) / active[b.name].area
    for b in shield:
        g = active[b.name]
        filled = unary_union([Polygon(p.exterior) for p in _polys(g)])
        m = contains_xy(g, XC, YC) | ~contains_xy(filled, XC, YC)
        kind[m] = -1
        fid[b.name] = abs(cell_area[contains_xy(g, XC, YC)].sum() - g.area) / g.area
    overlap = 0
    for b in [b for b in bodies if b.role == "conductor"]:
        m = contains_xy(active[b.name], XC, YC)
        overlap += int(np.sum(m & (kind != 0) & (kind != b.group)))
        kind[m] = b.group
        fid[b.name] = abs(cell_area[m].sum() - active[b.name].area) / active[b.name].area
    if overlap:
        warns.append(f"{overlap} raster cells belong to overlapping conductors/shield (later body wins)")
    worst = max(fid.values()) if fid else 0.0
    if worst > 0.02:
        warns.append(f"raster area error up to {100 * worst:.1f} % (curved/oblique outlines): reduce raster_dx")
    # to CrossSection (local coordinates, origin at the domain corner)
    sx, sy = xn - d0, yn - d1
    conductors, diel = [], []
    for x0, y0, x1, y1 in mask_to_rects(kind == -1, sx, sy):
        conductors.append(Rect(x0, y0, x1, y1, 0, True))
    for gid in sorted({b.group for b in bodies if b.role == "conductor"}):
        for x0, y0, x1, y1 in mask_to_rects(kind == gid, sx, sy):
            conductors.append(Rect(x0, y0, x1, y1, gid, False))
    free = kind == 0
    for ev, mv in sorted({(round(e, 9), round(m_, 9)) for e, m_ in zip(eps[free], mu[free])}):
        if abs(ev - 1) < 1e-12 and abs(mv - 1) < 1e-12:
            continue
        for x0, y0, x1, y1 in mask_to_rects((np.abs(eps - ev) < 1e-9) & (np.abs(mu - mv) < 1e-9) & free, sx, sy):
            diel.append(Dielectric(x0, y0, x1, y1, ev, mv))
    cs = CrossSection(W, H, conductors, 1.0, "imported", diel)
    return ImportedStructure(cs, ax, pos, length, (d0, d1, d2, d3), bodies, fid, un, warns, (nxr, nyr))
