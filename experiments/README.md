# experiments/

Sandbox for trying ideas **on top of** the `herald` codebase without changing it. The core
library evolves only *after* an experiment is validated.

## Rules

1. **`herald` develops incrementally.** Core code (`herald/`, `services/`, `utils/`) is changed
   only through deliberate, validated increments — never mid-experiment.
2. **One folder per experiment**, named `stage-name-NN` (e.g. `area-cluster-01`): the pipeline
   *stage*, a short *name*, and a two-digit *index*.
3. **Experiment code lives entirely in its folder** and *imports* `herald` as a dependency. It
   must not edit anything under `herald/`, `services/`, or `utils/`.
4. **If the core data structures / logic don't fit, wrap — don't fork.** Write thin adapters or
   wrappers around the core logic inside the experiment folder to test the idea. The wrapper is
   throwaway; the point is to learn what the core *should* expose.
5. **Every experiment has an `idea.md`** documenting concisely: the hypothesis, the approach, how
   to run it, and the result/verdict. Keep it short.
6. **Promotion:** when an experiment works and is validated, its idea graduates into the core
   codebase as a proper incremental change (rule 1) — and the experiment folder stays as the
   record of how it was reached.

## Folder shape

```
experiments/stage-name-NN/
├── idea.md          # hypothesis · approach · how-to-run · result (rule 5)
├── run.py           # experiment entry (imports herald; no core edits)
├── <wrappers>.py    # optional throwaway adapters around core logic (rule 4)
├── *.slurm          # optional SLURM wrapper for GPU runs
└── out/             # outputs (rrd/npz/png/logs) — gitignored, local only
```

Track `idea.md` + code; **outputs go in `out/` and are gitignored** (see repo `.gitignore`).
Run from the repo root, e.g. `uv run python experiments/area-cluster-01/run.py ...`.
