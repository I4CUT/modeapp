"""Generate test STL structures (one STL per body) for the import pipeline. Units: mm, propagation along z."""
from __future__ import annotations
import os
import numpy as np
import trimesh
from shapely.geometry import Polygon, Point, box as sbox


def _extrude(poly, length):
    m = trimesh.creation.extrude_polygon(poly, length)      # z in [0, length]
    m.merge_vertices()
    return m


def rect_tube(cx, cy, inner_w, inner_h, wall, length):
    outer = sbox(cx - inner_w / 2 - wall, cy - inner_h / 2 - wall, cx + inner_w / 2 + wall, cy + inner_h / 2 + wall)
    inner = sbox(cx - inner_w / 2, cy - inner_h / 2, cx + inner_w / 2, cy + inner_h / 2)
    return _extrude(outer.difference(inner), length)


def circ_tube(cx, cy, r_in, wall, length, res=256):
    outer = Point(cx, cy).buffer(r_in + wall, quad_segs=res // 4)
    inner = Point(cx, cy).buffer(r_in, quad_segs=res // 4)
    return _extrude(outer.difference(inner), length)


def rect_bar(x0, y0, x1, y1, length):
    return _extrude(sbox(x0, y0, x1, y1), length)


def cylinder(cx, cy, r, length, res=256):
    return _extrude(Point(cx, cy).buffer(r, quad_segs=res // 4), length)


def make_all(outdir: str) -> dict:
    """Write STL sets; returns {set_name: [paths]}."""
    os.makedirs(outdir, exist_ok=True)
    sets = {}

    def save(setname, bodies):
        paths = []
        for name, m in bodies.items():
            p = os.path.join(outdir, f"{setname}_{name}.stl")
            m.export(p)
            paths.append(p)
        sets[setname] = paths

    # 1) benchmark stripline: cavity 500 x 205, strip 280 x 1 centred, length 1000
    save("stripline", {"shield": rect_tube(0, 0, 500, 205, 5, 1000),
                       "strip": rect_bar(-140, -0.5, 140, 0.5, 1000)})
    # 2) microstrip: cavity 40 x 24, substrate 1 mm eps 9.8 on the bottom, strip 1 x 0.035
    save("microstrip", {"shield": rect_tube(0, 0, 40, 24, 2, 50),
                        "substrate": rect_bar(-20, -12, 20, -11, 50),
                        "strip": rect_bar(-0.5, -11, 0.5, -10.965, 50)})
    # 3) circular coax: r_in(outer cavity)=7.5, inner conductor r=3
    save("coax_round", {"shield": circ_tube(0, 0, 7.5, 1.5, 60), "inner": cylinder(0, 0, 3.0, 60)})
    # 4) circular waveguide r=10
    save("circ_waveguide", {"shield": circ_tube(0, 0, 10.0, 1.5, 60)})
    return sets
