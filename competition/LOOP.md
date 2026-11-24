# Champion loop (user decision 2026-11-08)

Rules of the competition: FRAME.md. This file: how agents are run.

## Loop

```
champion = best baseline on dev M1 (ndmi_drop / low_peak_evi2)
no_win = 0, round = 1
while no_win < 4 and round <= 10:
    two agents in parallel:  A = improver (starts from the champion's code)
                             B = challenger (an idea not yet in attempts.md)
    harness scores both: python -m competition.score dev <name>
    harness: python -m competition.score compare <name> <champion> dev
    new champion = the candidate that beats the champion; if both do, the one with higher dev M1
    beat = paired bootstrap on dev fields, share of resamples with M1(candidate) > M1(champion) >= 0.95
    beaten: no_win = 0, else no_win += 1
    attempts.md: one entry per candidate, numbers only
end: final champion, runner-up and baselines scored once on the lockbox (python -m competition.score lockbox)
```

Beat rule and caps: proposed by Claude, user asked to start (2026-11-08); change here before round 1 if wanted.

## Agent brief (filled per agent: ROLE, NAME, CHAMPION)

You build one damage method for the competition in `competition/FRAME.md`. Read FRAME.md, this file and
`competition/attempts.md` first.

- Your role: ROLE.
  - improver: start from the current champion `methods/CHAMPION/` (copy it to `methods/NAME/`), and make it better.
  - challenger: build a method whose main idea is not in attempts.md yet.
- Write the method to `methods/NAME/` (method.py with `predict(field)` and optional `fit(fields, labels)`,
  settings.yaml, README.md), per FRAME.md section 5. Data API: `competition/field.py`.
- Score yourself only on development: `python -m competition.score dev NAME`, and compare:
  `python -m competition.score compare NAME CHAMPION dev`. Use the frostmap environment
  (`/opt/homebrew/Caskroom/miniforge/base/envs/frostmap/bin/python`).
- Allowed reading: competition/FRAME.md, LOOP.md, attempts.md, field.py, score.py, baselines/, competition/data/
  (dev labels; lockbox fields' imagery only as unlabeled data, never for tuning), methods/ (any method's code).
- Not allowed: competition/_harness/, cache/brazil_psr/, validation/brazil_psr/ data or caches, PSR CSVs,
  ERA5 / any weather, `score lockbox`, editing score.py, field.py, harness.py, baselines or other agents' folders.
- Leave-one-year-out is the dev score; do not tune on single folds you have looked at many times. Dev has
  ~280 fields: a gain inside the bootstrap interval is noise.
- Budget: hand in when you are done or after about 3 hours of work. Hand-in message: method name, idea in 2-3
  lines, final dev metrics (the score output), compare output, what is learned from labels.
- Write numbers only in README results: no causal verdicts about why something did or did not work.
