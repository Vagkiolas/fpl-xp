# fpl-xp

Expected-points (xP) projections for Fantasy Premier League, built from the public FPL API.

The output CSV plugs straight into [open-fpl-solver](https://github.com/solioanalytics/open-fpl-solver),
a MILP (HiGHS) optimizer that turns projections into transfers, starting XI and captain picks
for your own team.

```
FPL API ──> make_projections.py ──> myxp.csv ──> open-fpl-solver ──> transfers / XI / captain
            (this repo: the model)                (the optimizer)
```

## Quick start

```bash
uv run --with-requirements requirements.txt python make_projections.py --horizon 8 --out ../open-fpl-solver/data/myxp.csv --save-raw raw
```

(or `pip install -r requirements.txt` and plain `python ...`)

Then in `open-fpl-solver/data/user_settings.json`:

```json
"datasource": "myxp",
"team_data": "id",
"team_id": 1234567,
"horizon": 8
```

Your team ID is the number in the URL of your Points page:
`fantasy.premierleague.com/entry/1234567/event/...`.
The solver's `horizon` must not be longer than the projection horizon.

Run the solver:

```bash
cd ../open-fpl-solver/run
uv run python solve.py
```

## Model (v0)

For player *i* and gameweek *gw*:

```
xP[i,gw] = Σ_fixtures  rate90[i] · xMins[i,f] / 90 · (1 + α·(3 − FDR_f))
```

| Term | How it is computed |
|---|---|
| `rate90` | FPL points per 90 min, shrunk towards the position average: `(pts + k·prior) / (mins/90 + k)` with `k = 4` |
| `xMins` | minutes per team match so far × probability of being available (`chance_of_playing_next_round`, recovering linearly over 3 GWs) |
| `FDR` | official fixture difficulty (1–5), `α = 0.10` |

Double gameweeks sum both fixtures; blank gameweeks give 0.

## Known limitations (roadmap)

- [ ] Points are treated as linear in minutes (appearance points are not)
- [ ] Season-long averages: no recency weighting / form
- [ ] FDR is coarse → replace with a Poisson team-goals model (attack/defence strengths, home advantage)
- [ ] Split points into components: goals (xG), assists (xA), clean sheets, defensive contributions, bonus
- [ ] Better minutes model (rotation, returning from injury)
- [x] Weekly snapshots + evaluation against actual points and FPL's `ep_next`
- [ ] Backtest on past seasons and tune the parameters
- [ ] Sensitivity tools: exact tolerance ranges per player, "what if X is out" analysis

## Options

```
--horizon N        number of gameweeks to project (default 8)
--out PATH         output CSV (default projections.csv)
--save-raw DIR     also save a snapshot of the API to DIR/<season>/gw<N>/ (N = upcoming GW)
--offline DIR      rebuild from a saved snapshot folder, no internet needed
```

## Snapshots and evaluation

Run with `--save-raw raw` every week before the deadline. Each snapshot keeps what was known
*before* the deadline (prices, injury flags, fixtures) - information the API later overwrites.
Re-running in the same week overwrites that week's snapshot with fresher data.

```
raw/2026-27/gw10/   bootstrap.json  fixtures.json  meta.json
raw/2026-27/gw11/   ...
```

With two consecutive snapshots you can score the model:

```bash
python evaluate.py raw/2026-27            # MAE / RMSE per GW + whole period
python evaluate.py raw/2026-27 --gw 10    # one GW: by position, biggest misses
python evaluate.py raw/2026-27 --csv eval.csv
```

- actual points of GW N = `total_points` in snapshot N+1 minus snapshot N
- predictions are recomputed with the *current* model from snapshot N only, so after changing the
  model, re-running `evaluate.py` is an honest backtest (no look-ahead)
- FPL's own `ep_next` is scored on the same players as a benchmark
- only players predicted to play or who actually played are counted
