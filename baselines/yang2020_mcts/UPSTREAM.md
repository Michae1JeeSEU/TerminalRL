# Upstream provenance

The unmodified source snapshot in `upstream/` is the Yang--Wei repository:

- repository: <https://github.com/xuxiyang1993/Multi_MCTS_Guidance_Separation_Assurance>
- paper: Xuxi Yang and Peng Wei, “Scalable Multi-Agent Computational Guidance with Separation Assurance for Autonomous Urban Air Mobility”, *Journal of Guidance, Control, and Dynamics*, 43(8), 1473–1486 (2020)
- DOI: <https://doi.org/10.2514/1.G005000>
- acquired commit: `2c83aeab807daee6bab0561199116210d44839da`

`bootstrap.ps1` downloads the exact commit archive and extracts the source
without creating a nested Git repository. The wrapper does not edit the files
under `upstream/`.

At import time, `runtime.py` makes one in-memory compatibility transform in
each node module. The source condition
`if self.state.init_action == 'random':` becomes a string-type guarded
condition so NumPy action arrays are not compared to a string. The transformed
module is compiled in the worker process and is not written to disk. No
algorithmic branch, action cadence, threshold, reward, or simulator constant
is changed.

The wrapper executes the three source entry-point loops headlessly:

| wrapper case | source entry point | planning counter | distance threshold |
|---|---|---:|---:|
| `crossing` | `Agent_vertiport.py` | starts at 1, period 5 | `3 × minimum_separation` |
| `hex` | `Agent_vertiHexSecGatePlus.py` | starts at 0, period 5 | `2 × minimum_separation` |
| `hex_two_stage` | `Agent_vertiHexSecGatePlusTwoStage.py` | starts at 0, period 5 | `5 × minimum_separation` for high and low priority |

The default source configuration is read from the acquired `Config` classes:
100 full rollouts, 30 lite rollouts, search depths 3 and 2, and
`simulate_frame=10` for these cases. A CLI budget or `--cohort-limit` changes
the workload and is recorded in `metadata.json`; bounded/truncated runs do
not reproduce the paper's reported performance experiment.
