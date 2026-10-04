# Crucible — the scoreboard layer for agent-society simulations

MiroFish (75k stars) built the theater: thousands of LLM personas rehearse a
future in a simulated world. Nobody built the referee. Crucible is the referee:
resolvable claims, ensemble runs, and scoring that tells you whether the
simulation actually knows anything.

**Clean-room build.** No AGPL code (MiroFish) was read into this codebase.
Permissive components only:

| Layer | Component | License |
|---|---|---|
| Simulation engine | [camel-ai/oasis](https://github.com/camel-ai/oasis) | Apache-2.0 |
| Memory / knowledge graph | [getzep/graphiti](https://github.com/getzep/graphiti) (self-hosted) | Apache-2.0 |
| Ground truth feeds | Metaculus / Manifold / Polymarket APIs | permissive APIs |
| Crucible itself | this repo | MIT |

## The loop (what MiroFish is missing)

```
claim ──► sim world (OASIS personas) ──► ensemble of N runs ──► distribution
                                   │                                  │
ground truth (feed or human) ──────┴──► resolve ──► score (Brier/log) ─┴─► calibration report
```

1. **Claim**: a falsifiable statement with a resolution date and source.
2. **Ensemble**: never one run — N runs, each a different persona seed; the
   output is a *distribution*, not a trajectory.
3. **Resolve**: outcome arrives from a ground-truth feed (prediction markets,
   Metaculus) or a human ruling.
4. **Score**: Brier + log score per run and per ensemble.
5. **Calibrate**: reliability curve across all resolved claims. This is the
   product: "our sim's 70% calls come true 71% of the time" is the thing
   MiroFish cannot say and a decision-maker will pay for.

## Layout

- `crucible/calibration.py` — pure-Python scoring core, zero dependencies.
- `tests/test_calibration.py` — Murphy-decomposition cases + the ensemble claim.

## Laws

- MIT forever. No AGPL-derived lines, ever.
- Every simulation spend goes through a ledger (nybls law applies here too).
- A prediction without a resolution source is not a prediction; the resolver
  refuses claims that cannot be scored.
