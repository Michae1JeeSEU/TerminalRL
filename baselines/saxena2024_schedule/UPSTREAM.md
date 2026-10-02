# Upstream record

This port records the official repository snapshot used as its source:

- repository: `https://github.com/rrsaxena92/Vertiport_schedule`
- default branch at acquisition: `main`
- immutable commit: `87a8e5ba7393d622768a7695770a4fa4e09c26bd`
- source entrypoint: `upstream/test_10_vert_schedule_ArrDep.m`
- topology: `upstream/topo_1_arr_dep_dir_4.m`
- source validator: `upstream/validateOptSol_AD.m`

The upstream snapshot contains MATLAB scripts and plotting helpers. The
repository API reported no license and the snapshot contains no `LICENSE`
file; no license is inferred here. The raw snapshot is ignored by the parent
`.gitignore` and is kept locally for audit and exact source comparison.

`model.py` is a new Python transcription of the function-local MILP. It uses
SciPy `optimize.milp` with HiGHS and emits each generated row by constraint
family. `run.py` writes the effective request times, node-time CSV, solver
status, and residual checks. The port is evidence from a newly run Python
solver; native MATLAB has not been executed and cross-solver numerical parity
is not claimed.

The source `wakeConstr` loops only over `f1 < f2`, while the paper's Eq. (12)
states the ordered pair condition. The default `--formulation paper` emits
both ordered wake rows; `--formulation upstream-legacy` preserves the source
one-sided loop. Other source rows, including the arrival-to-departure-only
`TLOFpadSingle` relation, are retained without inferred safety upgrades.

The paper selector is read against the accessible preprint
[`arXiv:2408.01152v1`](https://arxiv.org/html/2408.01152v1),
“Vertiport Terminal Scheduling with Multiple Surface Directions along with
Throughput Analysis”, specifically C10 / Eq. (12). The related final journal
record is Saxena, Kuri, Prabhakar, and Yadav, “Vertiport Terminal Scheduling
and Throughput Analysis for Multiple Surface Directions”, *The Aeronautical
Journal* 130(1350), 2434–2460,
[DOI 10.1017/aer.2026.10173](https://doi.org/10.1017/aer.2026.10173).
The final journal model was not independently verified here; the port targets
the fixed MATLAB snapshot above.

Constraint-family map in the port:

| Python family | Source helper / section |
| --- | --- |
| `ARAPR_i`, `gateOutC1`, `minspeed`, `maxspeed`, `TAT` | request, `SpeedConstr`, and turnaround rows |
| `ytime`, `y7`, `Overtake`, `Collision` | `YtimeConstr`, `y7Constr`, `overtakeConstr`, `collisonConstr` |
| `taxiSeparation1`, `fixSeparation1` | `taxiseparationConstr`, `fixseparationConstr` |
| `TLOFexitArr`, `TLOFClearArr`, `TLOFenterDep`, `TLOFenter2Dep` | corresponding TLOF helper rows |
| `wake` | `wakeConstr`; paper ordered-pair selector or legacy one-sided selector |
| `TLOFpadSingle` | `TLOFpadSingleConstr` |
| `GateQ`, `GateQ_ygenij`, `GateQ_ygexij`, `GateCapacity1`, `GateCapacity2` | turnaround continuity and gate-capacity helpers |

No MATLAB or Octave executable was available in the local or server PATH
during the bounded availability check, so native MATLAB execution remains
unverified.
