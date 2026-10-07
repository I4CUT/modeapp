"""Mode atlas: sweep one geometric/material parameter and track fc_n(p), Z0(p)."""
from __future__ import annotations
import csv
import numpy as np
from .solver import solve
from .analysis import single_mode_range


def mode_atlas(factory, param: str, values, dx: float = 5.0, n_modes: int = 6, edge_dx: float | None = None, **fixed):
    rows = []
    for v in values:
        ms = solve(factory(**{param: v}, **fixed), dx=dx, n_modes=n_modes + 1, edge_dx=edge_dx)
        hi = [m for m in ms.modes if m.fc > 0][:n_modes]
        rng = single_mode_range(ms)
        rows.append({"value": v, "fc": [m.fc for m in hi], "kinds": [m.kind for m in hi],
                     "z0": ms.modes[0].z0, "first_higher": rng.get("first_higher_fc"),
                     "first_excitable": rng.get("first_excitable_fc")})
    return rows


def save_csv(rows, path, param="param"):
    n = max(len(r["fc"]) for r in rows)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([param] + [f"fc{i + 1}_GHz" for i in range(n)] + ["Z0_Ohm", "first_excitable_GHz"])
        for r in rows:
            fe = r["first_excitable"]
            w.writerow([r["value"]] + [f"{f / 1e9:.6f}" for f in r["fc"]] + [f"{r['z0']:.4f}" if r["z0"] else "",
                       f"{fe / 1e9:.6f}" if fe else ""])


def plot_atlas(rows, path, xlabel="parameter"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [r["value"] for r in rows]
    n = max(len(r["fc"]) for r in rows)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    for i in range(n):
        a1.plot(x, [r["fc"][i] / 1e9 if i < len(r["fc"]) else np.nan for r in rows], "o-", ms=3, label=f"mode {i + 1}")
    a1.set_ylabel("fc, GHz"); a1.legend(); a1.grid(alpha=.3)
    a2.plot(x, [r["z0"] for r in rows], "s-", ms=3, color="C3")
    a2.set_ylabel("Z0, Ohm"); a2.set_xlabel(xlabel); a2.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
