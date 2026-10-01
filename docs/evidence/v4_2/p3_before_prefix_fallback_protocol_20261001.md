# P3 BEFORE-prefix fallback panel protocol

## Status and purpose

This is a pre-registered frozen panel diagnostic. It adds only the four `BEFORE-prefix` arms needed to complete the original four-cell comparison. It is not a training run, PPO admission, qualification, default promotion, or claim that the P3 regression is solved.

The existing fixed panel is:

- native `BEFORE` legacy: existing `80 normal / 16 emergency`;
- native `AFTER` legacy: existing `74 normal / 22 emergency`;
- `AFTER` prefix fallback: existing `88 normal / 8 emergency`;
- `BEFORE` prefix fallback: the only four new arms; results are unknown until run.

Only the four `BEFORE-prefix` arms are new. There is no new population, profile, policy seed, or cap expansion. Each arm keeps the original 24-request scope and the original 1800-step cap.

## Frozen inputs and controller

The candidate source is `BEFORE=3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab`. The immutable native `BEFORE` legacy baseline is guarded by the existing full artifact SHA `3f65446f49474452d31c3e1498c8019f04f4250cad3e8594c75107a123525919` and its four before arms. The completed `AFTER-prefix` full is an immutable comparison with SHA `020312813fc2c90a445bd3fe30e94e6739797151fe46efb65f760809a9361c50`; this run does not rerun `AFTER` or old V1.

Warmup `a82` is historical input metadata only; no warmup weights or warmup critic are injected, while normal checkpoint critic loading is unchanged. The controller reuses the existing public K=3 prefix and max25 fallback semantics, the same native source forward and native initial policy seed, singleton passthrough, and the original safety, retry, and capacity guards. The native Torch sampling sequence is retained; this is not an exact keyed-CRN sampler. Expanding the all-energy exhaustion check from three to up to 25 candidates can change induced emergency behavior, which must be measured rather than assumed identical.

Fixed cells are `875101..875104`; their fixed sampling seeds are the existing `policy_seed` values, while `875201..875204` are cell labels rather than substitute seeds. The same initial policy seeds are used for the candidate and native comparison, but the native Torch sampling sequence is not an exact CRN after trajectories diverge. A complete pilot for cell 1 requires root review before FULL may reuse it for cells 2–4. No automatic launch, seed resampling, new population, or budget expansion is allowed.

## Preflight, execution, and retention

Preflight binds the candidate source, frozen configuration, model and caller-RNG identities, runtime base, helper and wrapper source, this protocol, and the guarded native baseline hashes. It must validate input and JSON serialization before any pilot execution. Malformed controller inputs fail closed. Technical, model, environment, RNG, source, replay, or safety failure stops the phase. Business failures, censored runs, errors, and unresolved outcomes remain retained; unknown metrics remain unknown and are never filled with zero. No seed or failure is selected away.

The pilot and FULL phases preserve the native 24-request identities and report per-cell natural/censored/error status, technical and hard-safety status, new failed/diverted/unresolved arrival IDs, normal/emergency/failed/diverted/unresolved counts, wait, request elapsed, energy, raw reward, rejection penalty, and controller cost fields. The development screen requires all four cells to be natural with `censored=0` and `error=0`, technical and hard-safety checks clear, exact 24-request identity, and replay/ledger/energy/wait checks passing. Every cell must have no new failed/diverted/unresolved arrival IDs. All per-cell regressions remain visible.

## Development screen and claim boundary

The overall `BEFORE-prefix` screen uses the existing development rule: aggregate normal and successful completion must be at least native `BEFORE`, strict business gain must be a normal increase or failed decrease, and aggregate wait, request elapsed, and energy must be no higher than native `BEFORE` plus `1e-8`. The report must also show the original native `BEFORE` versus `AFTER` business deltas and the repaired `BEFORE-prefix` versus `AFTER-prefix` deltas. These panel differences are not exact CRN difference-in-differences and cannot establish a population-wide training cause. Cost fields that are not exposed remain explicit unknowns.

The candidate loads only `BEFORE`; warmup is historical input metadata and the completed `AFTER-prefix` is an immutable comparison artifact. There is no new performance population. The protocol does not use PPO, teacher, optimizer, held-out data, or default promotion. It does not treat `AFTER-prefix > BEFORE legacy` as proof that training degradation is solved; the missing same-controller `BEFORE-prefix` arm is required for the panel comparison.
