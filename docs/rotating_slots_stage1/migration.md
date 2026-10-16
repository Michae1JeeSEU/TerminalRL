# Migration from archived fixed-terminal research

FT-RS-S1-1.0 is a separate namespace and configuration loader. Old training
checkpoints, policies, replay formats and optimizer assumptions are never
automatically adapted or imported as new-contract evidence.

| Archived model | Frozen rotating-slot contract |
|---|---|
| Ring radii 12/20/28 m; heights 6/10/14 m | Inner/middle/outer radii 300/450/600 m; heights 30/70/110 m AGL |
| Angular speed 0.12 rad/s; 1 s updates | Common clockwise 0.035 rad/s; 0.2 s execution and exact events |
| 3 m separation plus two 0.5 m tracking margins | Exactly 80 m Euclidean centre distance, no hidden extra buffer or tolerance undershoot |
| Constant phase SOC drains 0.0008/0.0012/0.0009/0.0014; reserve 0.06 | Kinematics-dependent literature power surrogate; 252 Ah at equivalent 85 V; normal touchdown reserve 0.25 |
| Normal pad at (0,0,0), independent emergency pad at (-38,0,0) | Six shared pads on radius 100 m, service 60/120/240 s; emergency has no outside pad |
| Historical route-level waiting/scheduling assumptions | Arbitrary initiation phase, immutable current maneuver, editable unstarted soft dependencies, source protection and pad service calendars |

The old `terminal_marl/core/fixed_terminal/` package and staged results remain
archived to preserve provenance. Their numerical success does not qualify this
new geometry, dynamics, safety, battery model or UI. No PPO training, later-stage
optimizer comparison, historical action set or old energy approximation is
enabled by the stage 1 entry points. Legacy viewer interpolation remains for
legacy users; the rotating-slot replay calls the authoritative trajectory at
the requested time and bypasses that interpolation path entirely.

Configuration, manifest and trajectory hashes are checked on replay load. A
bundle with legacy schema or a plan tied to a different profile/configuration
is rejected rather than silently migrated.
