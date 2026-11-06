"""
evaluate.py - how wrong were the projections?

For gameweek N it needs two snapshots saved with `make_projections.py --save-raw raw`:
    raw/<season>/gw<N>/     taken before the GW N deadline   -> what the model knew
    raw/<season>/gw<N+1>/   taken after GW N finished        -> what actually happened

    actual points  = total_points(gw N+1 snapshot) - total_points(gw N snapshot)
    actual minutes = minutes(gw N+1 snapshot)      - minutes(gw N snapshot)

The predictions are recomputed with the CURRENT model in make_projections.py, using only the
gw N snapshot. So after you improve the model, re-running this is an honest backtest: the model
only sees what was known before each deadline.

As a benchmark it also scores FPL's own prediction (`ep_next`) from the same snapshot.

Usage:
    python evaluate.py raw/2026-27            # every GW that has both snapshots + season summary
    python evaluate.py raw/2026-27 --gw 10    # one GW, with the biggest misses
    python evaluate.py raw/2026-27 --csv eval.csv   # also write player-level results
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from make_projections import POS, load_snapshot, next_gameweek, project


def snapshots(season_dir):
    """{gw: folder} for every raw/<season>/gw<N> folder."""
    out = {}
    for d in Path(season_dir).iterdir():
        m = re.fullmatch(r"gw(\d+)", d.name)
        if d.is_dir() and m and (d / "bootstrap.json").exists():
            out[int(m.group(1))] = d
    return dict(sorted(out.items()))


def gw_finished(bootstrap, gw):
    return any(e["id"] == gw and e.get("finished") for e in bootstrap["events"])


def evaluate_gw(before_dir, after_dir, gw):
    b0, f0 = load_snapshot(before_dir)
    b1, _ = load_snapshot(after_dir)
    if next_gameweek(b0) != gw:
        raise ValueError(f"{before_dir} is not a pre-deadline snapshot of GW{gw}")
    if not gw_finished(b1, gw):
        raise ValueError(f"GW{gw} was not finished when {after_dir} was saved")

    table, _ = project(b0, f0, horizon=1)
    pred = table[["ID", "Name", "Pos", "Team", f"{gw}_Pts", f"{gw}_xMins"]].rename(
        columns={f"{gw}_Pts": "xP", f"{gw}_xMins": "xMins"})

    cols = ["id", "total_points", "minutes", "ep_next"]
    e0 = pd.DataFrame(b0["elements"])[cols].rename(columns={"id": "ID"})
    e1 = pd.DataFrame(b1["elements"])[["id", "total_points", "minutes"]].rename(columns={"id": "ID"})
    df = pred.merge(e0, on="ID").merge(e1, on="ID", suffixes=("_0", "_1"))

    df["actual"] = df["total_points_1"] - df["total_points_0"]
    df["mins"] = df["minutes_1"] - df["minutes_0"]
    df["fpl_ep"] = pd.to_numeric(df["ep_next"], errors="coerce").fillna(0.0)
    df["gw"] = gw
    return df[["gw", "ID", "Name", "Pos", "Team", "xP", "fpl_ep", "actual", "xMins", "mins"]]


def scores(df):
    """Errors on the players that matter: predicted to play or actually played."""
    rel = df[(df["xMins"] > 0) | (df["mins"] > 0)]
    row = {"players": len(rel)}
    for name, col in [("ours", "xP"), ("FPL", "fpl_ep")]:
        err = rel[col] - rel["actual"]
        row[f"MAE {name}"] = err.abs().mean()
        row[f"RMSE {name}"] = np.sqrt((err ** 2).mean())
    row["bias ours"] = (rel["xP"] - rel["actual"]).mean()  # >0: we over-predict
    return row


def print_gw_report(df, gw, top=10):
    print(f"\n=== GW{gw} ===")
    print(pd.DataFrame([scores(df)]).round(3).to_string(index=False))

    by_pos = pd.DataFrame({p: scores(df[df["Pos"] == p]) for p in POS.values()}).T
    print("\nBy position:")
    print(by_pos[["players", "MAE ours", "MAE FPL", "bias ours"]].round(3).to_string())

    top_pred = df.nlargest(20, "xP")
    print(f"\nTop-20 by our xP: predicted {top_pred['xP'].sum():.1f}, actual {top_pred['actual'].sum():.0f} points")

    df = df.assign(error=df["xP"] - df["actual"])
    show = ["Name", "Team", "Pos", "xP", "fpl_ep", "actual", "xMins", "mins"]
    print(f"\nBiggest under-predictions (scored more than expected):")
    print(df.nsmallest(top, "error")[show].round(2).to_string(index=False))
    print(f"\nBiggest over-predictions (scored less than expected):")
    print(df.nlargest(top, "error")[show].round(2).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description="Score projections against what actually happened")
    ap.add_argument("season_dir", help="folder with gw<N> snapshots, e.g. raw/2026-27")
    ap.add_argument("--gw", type=int, help="evaluate a single gameweek (shows the biggest misses)")
    ap.add_argument("--csv", help="write player-level results to this CSV")
    args = ap.parse_args()

    snaps = snapshots(args.season_dir)
    gws = [args.gw] if args.gw else [g for g in snaps if g + 1 in snaps]
    if not gws or any(g not in snaps or g + 1 not in snaps for g in gws):
        have = ", ".join(f"gw{g}" for g in snaps) or "none"
        raise SystemExit(f"Need snapshots gw<N> and gw<N+1>. Found: {have}")

    results, summary = [], []
    for gw in gws:
        try:
            df = evaluate_gw(snaps[gw], snaps[gw + 1], gw)
        except ValueError as e:
            print(f"Skipping GW{gw}: {e}")
            continue
        results.append(df)
        summary.append({"GW": gw, **scores(df)})
        if args.gw:
            print_gw_report(df, gw)

    if not results:
        raise SystemExit("Nothing to evaluate.")
    if not args.gw:
        s = pd.DataFrame(summary)
        print("\nPer gameweek (lower is better):")
        print(s.round(3).to_string(index=False))
        all_df = pd.concat(results)
        print("\nWhole period:")
        print(pd.DataFrame([scores(all_df)]).round(3).to_string(index=False))
        print("\nRun with --gw N to see the biggest misses of one gameweek.")
    if args.csv:
        pd.concat(results).to_csv(args.csv, index=False)
        print(f"\nPlayer-level results written to {args.csv}")


if __name__ == "__main__":
    main()
