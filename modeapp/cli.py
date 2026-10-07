"""Command line: STL(s) -> recognised cross-section -> modes.

  python -m modeapp.cli shield.stl strip.stl --role "*shield*=ground" --role "*strip*=conductor" --dx 5 --edge-dx 0.5
  python -m modeapp.cli shield.stl sub.stl strip.stl --role sub=dielectric:9.8 --vector 5e9
"""
from __future__ import annotations
import argparse
import warnings
from .stl_import import import_structure
from .solver import solve
from .vector import solve_vector
from .analysis import excitability, single_mode_range


def analyse(sources, roles=None, dx=None, edge_dx=None, n_modes=8, vector_f=None, report=True, **kw):
    st = import_structure(sources, roles=roles, **kw)
    if report:
        print(st.summary())
    dx = dx or max(st.cs.width, st.cs.height) / 100.0
    ms = solve(st.cs, dx=dx, n_modes=n_modes, edge_dx=edge_dx)
    if report:
        print(f"\nSolver grid {ms.grid.nx}x{ms.grid.ny}, geometry symmetry (x,y) = {ms.geom_sym}\n")
        print(ms.table())
        r = single_mode_range(ms)
        if r:
            print(f"\nFirst higher mode {r['first_higher_fc'] / 1e6:.2f} MHz; "
                  + (f"first excitable by a symmetric source {r['first_excitable_fc'] / 1e6:.2f} MHz" if r.get("first_excitable_fc") else "none excitable by a symmetric source")
                  + f"; single-mode range up to {r['range_conservative'][1] / 1e6:.2f} MHz (with 20% margin {r['range_conservative_margin'][1] / 1e6:.2f} MHz)")
    vm = None
    if vector_f:
        try:
            vm = solve_vector(st.cs, vector_f, n_modes=4, dx=dx, edge_dx=edge_dx)
            if report:
                print(f"\nFull-vector modes at {vector_f / 1e9:.4g} GHz:")
                for i, m in enumerate(vm.modes):
                    z = f", Z_PI={m.z['Z_PI']:.2f} Ohm" if m.z.get("Z_PI") else ""
                    print(f"  {i}: {m.kind:9s} neff={m.neff:.4f} vp/c={1 / m.neff:.4f}{z}")
        except ValueError as e:
            print("vector solver skipped:", e)
    return st, ms, vm


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="STL/OBJ/... files or directories (one body per file)")
    ap.add_argument("--role", action="append", default=[], help="NAME=ground|conductor[:group]|dielectric:EPS|ignore (glob allowed)")
    ap.add_argument("--units", default="mm", choices=["mm", "m", "cm", "um", "in"])
    ap.add_argument("--axis", default="auto", choices=["auto", "x", "y", "z"])
    ap.add_argument("--pos", type=float, default=0.5, help="plane position: fraction (<=1) of the length or absolute mm (>1)")
    ap.add_argument("--raster-dx", type=float, default=None, help="rasterisation cell size, mm (curved outlines need it small)")
    ap.add_argument("--pad", type=float, default=None, help="padding of the PEC box if the model has no enclosure, mm")
    ap.add_argument("--dx", type=float, default=None, help="solver max cell size, mm")
    ap.add_argument("--edge-dx", type=float, default=None, help="solver cell size at conductor edges, mm")
    ap.add_argument("--modes", type=int, default=8)
    ap.add_argument("--vector", type=float, default=None, help="also run the full-vector solver at this frequency, Hz")
    a = ap.parse_args(argv)
    roles = dict(r.split("=", 1) for r in a.role)
    warnings.simplefilter("ignore")
    analyse(a.files, roles or None, a.dx, a.edge_dx, a.modes, a.vector, units=a.units, axis=a.axis,
            position=a.pos, raster_dx=a.raster_dx, pad=a.pad)


if __name__ == "__main__":
    main()
