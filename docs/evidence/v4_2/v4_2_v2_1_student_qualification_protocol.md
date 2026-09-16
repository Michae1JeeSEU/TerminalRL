# Student-only I3 qualification V1

Preregistered 2026-09-17 after DAgger R1 PASS, before any I3 episode.
Authority: `terminal_marl/configs/low_v4_2/i3_v2_1_qualification.yaml`.

R1 retained 976 new semantic events, combined once with 7,576 R4 rows
(D1=8,552 rows), trained 35 fixed epochs, and passed all collection,
retention and fresh short-validation gates. All 16 tapes replayed and
requested likelihoods matched; all safety counts were zero.
R1 manifest SHA256:
`3be18287f3956ded27a376ac53b9cf5b285d77df553fb1738db1d6c1eae0e079`.
Frozen student SHA256:
`a97e0284b07b18404f60cd666c72806cbf5e261bf807c7f9f3a1d897c69b1584`.

I3 uses six fresh seeds660001–660006. In order: warm-start N3, natural-arrival
N3, natural-arrival N6, repeated once. The first three use deterministic
student requests, the last three sample the student distribution. Natural
arrivals start at 10 s and repeat every 60 s. Horizon is 900 s; stop earlier
only when all arrivals, active UAVs, queue and service are exhausted.
Teacher is disabled entirely. Fixed geometry and safety remain enabled.
No optimization, Teacher labels, Q2 access or checkpoint selection occurs.

Every episode must complete all requested UAVs before the deadline, be
terminal with a drained queue, have zero collision/NMAC/hard violation/
uncertified execution/SafetyInvariantError, and exactly replay actions/states
and requested likelihoods. Shield interventions on actor decisions must be
<=5%; both WAIT and RELEASE must occur across the complete suite. Report
completion, throughput per hour, queue trace and shield fraction per seed.
Thus initial N3/N6 throughput must be at least 12/24 completions per hour
respectively within this bounded suite; completion cannot be replaced by a
short-horizon zero-safety observation. No failed episode is discarded.

This is admission evidence for the initial fixed-geometry N3–N6 curriculum,
not full static-geometry robustness qualification, saturation capacity,
dynamic-airspace readiness or hardware safety certification. Larger loads
and new geometry require subsequent curriculum qualification.

Create an exclusive qualification-consumed marker before first reset.
An interrupted attempt is not automatically repeated. Smoke seed669001
runs one warm3 step, verifies the output/replay path and cannot qualify I3.
After I3 PASS, a separate admission review and training-entry validation are
still required; this runner always records PPO_ALLOWED=false.
