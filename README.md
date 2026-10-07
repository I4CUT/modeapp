# modeapp - 2D cross-section mode analysis (stages 1-3: solvers + STL pipeline)

    pip install -r requirements.txt
    python -m modeapp.benchmark        # validation, dispersion, mode atlases -> ./results

## Contents
- geometry.py   CrossSection = PEC box + conductor rectangles (signal / ground, thick or zero-thickness) + dielectric
                rectangles. Factories: rectangular_waveguide, symmetric_stripline, microstrip, cpw, rect_coax.
- grid.py       non-uniform tensor grid, lines on every conductor/dielectric edge, optional graded refinement at
                conductor edges (edge_dx). Strip thickness is resolved exactly (no snapping).
- solver.py     solve(): cutoffs of TE-like (Hz) and TM-like (Ez) modes (exact for beta=0 in inhomogeneous filling),
                quasi-TEM modes from static capacitance matrices (eps_eff, Z0, modal eps_eff for several conductors),
                transverse E fields, mirror parity, confidence.
- vector.py     solve_vector(): full-vector Yee-grid solver, beta(f), neff, vp, vg (dispersion()), 1 W-normalised
                fields, Z by P-I, P-V, V-I.
- analysis.py   excitability by source symmetry (+ reason), single-mode range with margin.
- atlas.py      parameter sweeps -> CSV / PNG.   validation.py   analytical references.

## STL pipeline (stage 3)
    python -m modeapp.cli shield.stl sub.stl strip.stl --role sub=dielectric:9.8 --dx 0.5 --edge-dx 0.05 --vector 5e9
    python -m modeapp.stl_benchmark      # generates test STLs in results/test_meshes and validates the pipeline
- stl_import.py  load bodies (one STL per body) -> propagation axis (the axis along which all bodies are prismatic)
                 -> section at a chosen position -> roles (ground/shield, conductor[:group], dielectric:eps_r) ->
                 raster -> rectangles -> CrossSection. Reports raster area error and translation invariance.
- cli.py         recognition report + modes (+ optional full-vector solve).  testmeshes.py: test STL generators.
- Roles: an enclosure (closed outline around everything) is detected automatically; materials are NOT in an STL,
  so dielectrics need eps_r from the user (--role NAME=dielectric:EPS); anything else defaults to a signal conductor.

## Usage
    from modeapp import microstrip, solve
    from modeapp.vector import solve_vector, dispersion
    cs = microstrip(h=1.0, w=1.0, t=0.035, eps_r=9.8, box_w=40, box_h=24)
    ms = solve(cs, dx=0.5, edge_dx=0.02); print(ms.table())
    vm = solve_vector(cs, 10e9, n_modes=3, dx=0.25, edge_dx=0.05); print([(m.kind, m.neff) for m in vm.modes])

## Known limitations
- Vector solver: mu_r = 1 only (the scalar cutoff solver supports mu_r).
- Mode type of hybrid modes is classified by Ez/Hz energy (heuristic thresholds); modes are not strictly TE/TM.
- Field singularity at conductor edges: eps_eff/Z0 converge slowly with refinement (see benchmark section 4).
- Vector beta of modes within ~10 % of cutoff needs fine meshes (benchmark section 3).
- Current for Z by P-I is a tight Ampere loop (1 cell ring) and includes a small displacement-current part.
- Parity is only defined along axes in which the geometry is symmetric (otherwise reported as n/a).
- Lossless, PEC walls only. No conductor/dielectric loss yet.
- STL: bodies that touch and share vertices merge into one body when loaded from one file: use one file per body.
- STL: one cross-section plane only (non-prismatic structures are flagged, not analysed along the length).
- STL: curved outlines are staircased (accuracy follows raster_dx, see stl_benchmark C/D); conductors thinner than the
  raster cell may vanish for curved outlines (rectilinear outlines are resolved exactly).
