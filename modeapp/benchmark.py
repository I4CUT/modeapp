"""Validation suite: python -m modeapp.benchmark  (results are written to ./results)"""
import logging
import os
import numpy as np
from .geometry import (symmetric_stripline, microstrip, cpw, rect_coax, CrossSection, Dielectric)
from .solver import solve, C0
from .vector import solve_vector, dispersion
from .analysis import excitability, single_mode_range
from .validation import (validate_rect_waveguide, stripline_z0_thin_strip, loaded_wg_betas, loaded_wg_cutoffs,
                         microstrip_hammerstad)
from .atlas import mode_atlas, save_csv, plot_atlas


def main(outdir="results"):
    logging.basicConfig(level=logging.WARNING)
    os.makedirs(outdir, exist_ok=True)
    print("=== 1. Rectangular waveguide 22.86x10.16 mm: cutoff numerical vs analytical ===")
    for name, kind, fa, fn, err in validate_rect_waveguide():
        print(f"{name}: analytic {fa:.4f} GHz, numeric {fn:.4f} GHz, error {err:+.3f} %")

    print("\n=== 2. Symmetric stripline (W=500, H=205, strip 280 x 1 mm), real strip thickness ===")
    for dx, e in ((5.0, None), (5.0, 0.5), (2.5, 0.5), (2.5, 0.25)):
        ms = solve(symmetric_stripline(), dx=dx, n_modes=8, edge_dx=e)
        print(f"dx={dx}, edge_dx={e}, grid {ms.grid.nx}x{ms.grid.ny}: Z0={ms.modes[0].z0:.3f} Ohm; fc, MHz: "
              + ", ".join(f"{m.fc / 1e6:.1f}({m.symmetry})" for m in ms.modes[1:5]))
    ms = solve(symmetric_stripline(), dx=2.5, n_modes=8, edge_dx=0.5)
    print("\n" + ms.table())
    for m in ms.modes:
        st, why = excitability(m, ms)
        print(f"  mode {m.number} {m.kind:8s} {m.fc / 1e6:8.1f} MHz -> {st}: {why}")
    r = single_mode_range(ms)
    print(f"First higher mode {r['first_higher_fc'] / 1e6:.1f} MHz; first excitable {r['first_excitable_fc'] / 1e6:.1f} MHz; "
          f"single-mode range 10-{r['range_conservative'][1] / 1e6:.1f} MHz, with 20% margin 10-{r['range_conservative_margin'][1] / 1e6:.1f} MHz")
    print(f"Cohn thin strip, no side walls: {stripline_z0_thin_strip(280, 205):.2f} Ohm")
    zc = stripline_z0_thin_strip(280, 205)
    zw = solve(symmetric_stripline(width=2000, thickness=0.0), dx=5.0, n_modes=1, edge_dx=0.5).modes[0].z0
    print(f"Exact check (wide box 2000 mm, zero-thickness strip) Z0={zw:.3f} Ohm vs Cohn {zc:.3f} Ohm ({100 * (zw - zc) / zc:+.2f} %)")
    mv = solve_vector(symmetric_stripline(), 100e6, n_modes=1, dx=5.0, edge_dx=0.5).modes[0]
    print(f"Full-vector @100 MHz: neff={mv.neff:.5f} (TEM => 1), Z_PI={mv.z['Z_PI']:.2f}, Z_PV={mv.z['Z_PV']:.2f}, Z_VI={mv.z['Z_VI']:.2f} Ohm")

    print("\n=== 3. Rectangular guide 22.86x10.16 loaded with eps_r=4 slab (t=8 mm, side wall): analytic LSE/LSM ===")
    a, b, t, eps = 22.86, 10.16, 8.0, 4.0
    cs = CrossSection(a, b, [], 1.0, "loaded", [Dielectric(0, 0, t, b, eps)])
    sc = solve(cs, dx=0.125, n_modes=4)
    ana = loaded_wg_cutoffs(a, b, t, eps)
    print("cutoffs, GHz   numeric (scalar FV) vs analytic:")
    for m, (fa, nm) in zip(sc.modes, ana):
        print(f"  {nm}: analytic {fa / 1e9:.4f}, numeric {m.fc / 1e9:.4f} ({m.kind}), error {100 * (m.fc - fa) / fa:+.3f} %")
    ba = loaded_wg_betas(10e9, a, b, t, eps)
    print("beta @10 GHz, full-vector error vs analytic, by mesh:")
    for dx in (0.5, 0.25, 0.125):
        ms = solve_vector(cs, 10e9, n_modes=4, dx=dx)
        print(f"  dx={dx}: " + ", ".join(f"{n} {100 * (m.beta - x) / x:+.2f}%" for m, (x, n) in zip(ms.modes, ba)))

    print("\n=== 4. Microstrip (h=1 mm, eps_r=9.8, t=35 um): static solver vs Hammerstad-Jensen (open, t=0) ===")
    for w in (0.5, 1.0, 2.0):
        ee, z = microstrip_hammerstad(w, 9.8)
        for bw, bh in ((20, 12), (40, 24), (80, 48)):
            m = solve(microstrip(1.0, w, 0.035, 9.8, bw, bh), dx=0.5, n_modes=2, edge_dx=0.05).modes[0]
            print(f"  w/h={w}, box {bw}x{bh} mm: eps_eff={m.eps_eff:.3f} (H-J {ee:.3f}), Z0={m.z0:.2f} Ohm (H-J {z:.2f})")
    cs = microstrip(1.0, 1.0, 0.035, 9.8, 20, 12)
    print("static vs full-vector @1 GHz, mesh refinement (edge singularity -> slow convergence):")
    for dx, e in ((0.5, 0.1), (0.25, 0.05), (0.2, 0.025)):
        st = solve(cs, dx=dx, n_modes=2, edge_dx=e).modes[0]
        m = solve_vector(cs, 1e9, n_modes=1, dx=dx, edge_dx=e).modes[0]
        print(f"  dx={dx}, edge={e}: static eps_eff={st.eps_eff:.3f}, Z0={st.z0:.2f} | vector neff^2={m.neff ** 2:.3f}, "
              f"Z_PI={m.z['Z_PI']:.2f}, Z_PV={m.z['Z_PV']:.2f}, Z_VI={m.z['Z_VI']:.2f}")
    print("static edge-refinement trend, box 40x24:")
    for dx, e in ((0.5, 0.05), (0.5, 0.02), (0.5, 0.01), (0.25, 0.005)):
        m = solve(microstrip(1.0, 1.0, 0.035, 9.8, 40, 24), dx=dx, n_modes=1, edge_dx=e).modes[0]
        print(f"  dx={dx}, edge={e}: eps_eff={m.eps_eff:.4f}, Z0={m.z0:.3f} Ohm")
    fr = np.array([0.5, 1, 2, 4, 6, 8, 10, 12, 14]) * 1e9
    d = dispersion(cs, fr, dx=0.25, edge_dx=0.05, n_track=1)[0]
    print("dispersion of the fundamental mode:")
    print("  f, GHz | neff^2 | vp/c  | vg/c  | Z_PI  | Z_PV  | Z_VI")
    for i, f in enumerate(fr):
        print(f"  {f / 1e9:6.1f} | {d['neff'][i] ** 2:6.3f} | {1 / d['neff'][i]:.3f} | {d['vg'][i] / C0:.3f} | "
              f"{d['Z_PI'][i]:5.2f} | {d['Z_PV'][i]:5.2f} | {d['Z_VI'][i]:5.2f}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(fr / 1e9, d["neff"] ** 2, "o-"); ax[0].set_xlabel("f, GHz"); ax[0].set_ylabel("eps_eff(f) = neff^2"); ax[0].grid(alpha=.3)
    for k in ("Z_PI", "Z_PV", "Z_VI"):
        ax[1].plot(fr / 1e9, d[k], "o-", label=k)
    ax[1].set_xlabel("f, GHz"); ax[1].set_ylabel("Z, Ohm"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.suptitle("Microstrip w/h=1, eps_r=9.8: full-vector dispersion"); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "microstrip_dispersion.png"), dpi=130); plt.close(fig)

    print("\n=== 5. Coax scaling check and CPW demo ===")
    z1 = solve(rect_coax(eps_r=1.0), dx=0.25, n_modes=2).modes[0]
    z2 = solve(rect_coax(eps_r=2.0), dx=0.25, n_modes=2).modes[0]
    print(f"square coax 20x20 / 8x8 mm: Z0(air)={z1.z0:.2f}, Z0(eps_r=2)={z2.z0:.2f}, ratio={z1.z0 / z2.z0:.4f} (sqrt2={np.sqrt(2):.4f})")
    ms = solve(cpw(1.0, 0.5, 1.0, 0.035, 9.8, 20, 12), dx=0.5, n_modes=4, edge_dx=0.05)
    print(ms.table())

    print("\n=== 6. Mode atlases ===")
    rows = mode_atlas(symmetric_stripline, "strip_width", np.arange(100, 401, 25.0), dx=5.0, n_modes=4, edge_dx=0.5)
    save_csv(rows, os.path.join(outdir, "atlas_strip_width.csv"), "strip_width_mm")
    plot_atlas(rows, os.path.join(outdir, "atlas_strip_width.png"), "stripline strip width, mm")
    rows = mode_atlas(lambda w: microstrip(1.0, w, 0.035, 9.8, 20, 12), "w", np.arange(0.4, 3.01, 0.2), dx=0.5, n_modes=3, edge_dx=0.05)
    save_csv(rows, os.path.join(outdir, "atlas_microstrip_w.csv"), "w_mm")
    plot_atlas(rows, os.path.join(outdir, "atlas_microstrip_w.png"), "microstrip width w, mm (h=1 mm)")
    print("saved atlas CSV/PNG to", outdir)


if __name__ == "__main__":
    main()
