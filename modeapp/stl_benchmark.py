"""Validation of the STL pipeline: python -m modeapp.stl_benchmark"""
import os
import time
import warnings
import numpy as np
import trimesh
from scipy.special import jn_zeros, jnp_zeros
from .testmeshes import make_all, circ_tube, cylinder
from .stl_import import import_structure
from .geometry import symmetric_stripline, microstrip
from .solver import solve, C0, ETA0


def main(outdir="results"):
    warnings.simplefilter("ignore")
    meshdir = os.path.join(outdir, "test_meshes")
    sets = make_all(meshdir)
    print("test STL files written to", meshdir)

    print("\n=== A. STL stripline vs parametric stripline (same solver settings) ===")
    st = import_structure(sets["stripline"])
    print(st.summary())
    a = solve(st.cs, dx=2.5, n_modes=6, edge_dx=0.5)
    b = solve(symmetric_stripline(), dx=2.5, n_modes=6, edge_dx=0.5)
    print(f"Z0: STL {a.modes[0].z0:.3f} Ohm, parametric {b.modes[0].z0:.3f} Ohm")
    print("fc, MHz  STL: " + ", ".join(f"{m.fc / 1e6:.2f}" for m in a.modes[1:6]))
    print("fc, MHz  par: " + ", ".join(f"{m.fc / 1e6:.2f}" for m in b.modes[1:6]))

    print("\n=== B. STL microstrip (substrate eps_r=9.8 given by the user) vs parametric ===")
    st = import_structure(sets["microstrip"], roles={"*substrate": "dielectric:9.8"})
    print(st.summary())
    a = solve(st.cs, dx=0.5, n_modes=4, edge_dx=0.05)
    b = solve(microstrip(1.0, 1.0, 0.035, 9.8, 40, 24), dx=0.5, n_modes=4, edge_dx=0.05)
    print(f"quasi-TEM: STL eps_eff={a.modes[0].eps_eff:.4f}, Z0={a.modes[0].z0:.3f} | parametric eps_eff={b.modes[0].eps_eff:.4f}, Z0={b.modes[0].z0:.3f}")
    print("fc, GHz  STL: " + ", ".join(f"{m.fc / 1e9:.3f}" for m in a.modes[1:4]) + "   par: " + ", ".join(f"{m.fc / 1e9:.3f}" for m in b.modes[1:4]))

    print("\n=== C. Round coax (outer r=7.5, inner r=3 mm): curved outlines, analytic Z0 ===")
    z_an = ETA0 / (2 * np.pi) * np.log(7.5 / 3.0)
    print(f"analytic Z0 = {z_an:.3f} Ohm")
    for rd in (0.2, 0.1, 0.05):
        t0 = time.time()
        st = import_structure(sets["coax_round"], raster_dx=rd)
        ms = solve(st.cs, dx=rd, n_modes=3, edge_dx=None)
        err = max(st.fidelity.values())
        print(f"  raster_dx={rd}: grid {ms.grid.nx}x{ms.grid.ny}, Z0={ms.modes[0].z0:.3f} Ohm ({100 * (ms.modes[0].z0 - z_an) / z_an:+.2f} %), "
              f"max raster area error {100 * err:.3f} %, {time.time() - t0:.1f} s")

    print("\n=== D. Round waveguide r=10 mm: cutoffs vs Bessel zeros ===")
    a_r = 10.0
    ana = sorted([(jnp_zeros(1, 1)[0], "TE11"), (jnp_zeros(1, 1)[0], "TE11"), (jn_zeros(0, 1)[0], "TM01"),
                  (jnp_zeros(2, 1)[0], "TE21"), (jnp_zeros(2, 1)[0], "TE21"), (jnp_zeros(0, 1)[0], "TE01")])
    ana = [(C0 * x / (2 * np.pi * a_r) * 1e3 / 1e9, n) for x, n in ana]
    for rd in (0.2, 0.1, 0.05):
        st = import_structure(sets["circ_waveguide"], raster_dx=rd)
        ms = solve(st.cs, dx=rd, n_modes=6)
        fc = [m.fc / 1e9 for m in ms.modes]
        print(f"  raster_dx={rd}: " + ", ".join(f"{n} {f:.4f} ({100 * (f - fa) / fa:+.2f}%)" for f, (fa, n) in zip(fc, ana)))

    print("\n=== E. Recognition checks ===")
    sub = {"strip_only": trimesh.load(sets["stripline"][1])}
    st = import_structure(sub)
    print("no enclosure ->", [w for w in st.warnings if "enclosure" in w][0])
    try:
        import_structure(sets["microstrip"])
    except ValueError as e:
        print("dielectric without eps_r ->", e)
    tube = trimesh.load(sets["stripline"][0])
    cone = trimesh.creation.cone(radius=40.0, height=900.0, sections=64)
    cone.apply_translation([0, 0, 50.0])
    st = import_structure({"shield": tube, "cone": cone})
    print("tapered body ->", [w for w in st.warnings if "changes along" in w][0], f"(max change {100 * st.uniformity['max_change']:.0f} %)")
    st = import_structure(sets["stripline"], axis="x")
    print("forced wrong axis -> axis", "xyz"[st.axis], "|", "; ".join(w[:70] for w in st.warnings))


if __name__ == "__main__":
    main()
