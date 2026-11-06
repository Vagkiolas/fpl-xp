"""
make_projections.py - v0 expected-points (xP) projections for Fantasy Premier League.

Reads the public FPL API (no login needed) and writes a CSV in the format
read by open-fpl-solver (https://github.com/solioanalytics/open-fpl-solver):

    ID, Name, Pos, Team, Value, {gw}_Pts, {gw}_xMins, ...

Model (deliberately simple, to be replaced step by step):
    xP[i, gw] = sum over fixtures f of player i's team in gw:
                rate90[i] * xMins[i, f] / 90 * fixture_factor[f]

    rate90  : FPL points per 90 minutes, shrunk towards the position average
              (Bayesian shrinkage: players with few minutes get pulled to the mean)
    xMins   : minutes per team match so far, times probability of being available
    factor  : 1 + alpha * (3 - FDR), FDR = official fixture difficulty (1..5)

Usage:
    python make_projections.py                       # next 8 GWs -> projections.csv
    python make_projections.py --horizon 6 --out ../open-fpl-solver/data/myxp.csv
    python make_projections.py --save-raw raw/       # also keep the API snapshots
    python make_projections.py --offline raw/        # rebuild from saved snapshots
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

API = "https://fantasy.premierleague.com/api"
POS = {1: "G", 2: "D", 3: "M", 4: "F"}

# Model parameters (v0) - tune these, or better: fit them on past seasons.
PRIOR_MATCHES = 4.0       # strength of shrinkage, in "90-minute matches" of pseudo-data
PRIOR_MIN_MINUTES = 450   # players used to compute the position average per 90
FDR_ALPHA = 0.10          # FDR 2 -> x1.10, FDR 5 -> x0.80
RECOVERY_GWS = 3          # doubtful/injured players return to normal over this many GWs
UNAVAILABLE = {"u", "n"}  # u = left the club / unavailable, n = not eligible (e.g. loan)


# --------------------------------------------------------------------------- data
def fetch(endpoint):
    r = requests.get(f"{API}/{endpoint}/", timeout=30, headers={"User-Agent": "fpl-xp/0.1"})
    r.raise_for_status()
    return r.json()


def load_data(offline_dir=None, save_dir=None):
    if offline_dir:
        d = Path(offline_dir)
        return json.loads((d / "bootstrap.json").read_text()), json.loads((d / "fixtures.json").read_text())
    bootstrap, fixtures = fetch("bootstrap-static"), fetch("fixtures")
    if save_dir:
        d = Path(save_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "bootstrap.json").write_text(json.dumps(bootstrap))
        (d / "fixtures.json").write_text(json.dumps(fixtures))
    return bootstrap, fixtures


# --------------------------------------------------------------------------- model
def next_gameweek(bootstrap):
    for e in bootstrap["events"]:
        if e["is_next"]:
            return e["id"]
    raise RuntimeError("No upcoming gameweek found (season over?)")


def team_matches_played(fixtures):
    played = defaultdict(int)
    for f in fixtures:
        if f.get("finished"):
            played[f["team_h"]] += 1
            played[f["team_a"]] += 1
    return played


def players_table(bootstrap, fixtures):
    teams = {t["id"]: t["short_name"] for t in bootstrap["teams"]}
    played = team_matches_played(fixtures)
    df = pd.DataFrame(bootstrap["elements"])
    df = df[df["element_type"].isin(POS)].copy()  # skip non-player types if any appear

    df["Pos"] = df["element_type"].map(POS)
    df["Team"] = df["team"].map(teams)
    df["Value"] = df["now_cost"] / 10
    df["team_played"] = df["team"].map(played).fillna(0)

    # --- points per 90, shrunk to the position average
    regulars = df[df["minutes"] >= PRIOR_MIN_MINUTES]
    prior = (regulars.groupby("Pos")["total_points"].sum() / regulars.groupby("Pos")["minutes"].sum() * 90)
    df["prior90"] = df["Pos"].map(prior).fillna(prior.mean() if len(prior) else 2.0)
    nineties = df["minutes"] / 90
    df["rate90"] = (df["total_points"] + PRIOR_MATCHES * df["prior90"]) / (nineties + PRIOR_MATCHES)

    # --- expected minutes per match when fit
    df["mins_per_match"] = (df["minutes"] / df["team_played"].where(df["team_played"] > 0)).fillna(0).clip(upper=90)

    # --- probability of being available next GW
    chance = df["chance_of_playing_next_round"]
    df["p_next"] = (chance.where(chance.notna(), 100) / 100).astype(float)
    df.loc[df["status"].isin(UNAVAILABLE), "p_next"] = 0.0
    return df


def availability(p_next, status, k):
    """Probability of availability k GWs after the next one (k = 0 is the next GW)."""
    if status in UNAVAILABLE:
        return 0.0
    return min(1.0, p_next + (1 - p_next) * k / RECOVERY_GWS)


def fixtures_by_gw(fixtures, gws):
    """{(team_id, gw): [difficulty, ...]} - a list so double GWs sum and blanks are empty."""
    out = defaultdict(list)
    for f in fixtures:
        if f.get("event") in gws:
            out[(f["team_h"], f["event"])].append(f["team_h_difficulty"])
            out[(f["team_a"], f["event"])].append(f["team_a_difficulty"])
    return out


def project(bootstrap, fixtures, horizon):
    gw0 = next_gameweek(bootstrap)
    gws = list(range(gw0, min(38, gw0 + horizon - 1) + 1))
    df = players_table(bootstrap, fixtures)
    fx = fixtures_by_gw(fixtures, set(gws))

    cols = {}
    for k, gw in enumerate(gws):
        pts, mins = [], []
        for row in df.itertuples():
            p = availability(row.p_next, row.status, k)
            xp = xm = 0.0
            for fdr in fx.get((row.team, gw), []):
                m = row.mins_per_match * p
                xm += m
                xp += row.rate90 * m / 90 * (1 + FDR_ALPHA * (3 - fdr))
            pts.append(round(xp, 2))
            mins.append(round(xm, 1))
        cols[f"{gw}_Pts"] = pts
        cols[f"{gw}_xMins"] = mins

    out = pd.concat([df[["id", "web_name", "Pos", "Team", "Value"]].reset_index(drop=True),
                     pd.DataFrame(cols)], axis=1)
    out = out.rename(columns={"id": "ID", "web_name": "Name"})
    pts_cols = [c for c in out.columns if c.endswith("_Pts")]
    return out.sort_values(by=pts_cols[0], ascending=False), gws


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="v0 FPL projections -> CSV for open-fpl-solver")
    ap.add_argument("--horizon", type=int, default=8, help="number of gameweeks to project (default 8)")
    ap.add_argument("--out", default="projections.csv", help="output CSV path")
    ap.add_argument("--save-raw", metavar="DIR", help="also save the API responses to DIR")
    ap.add_argument("--offline", metavar="DIR", help="read saved API responses from DIR instead of the internet")
    args = ap.parse_args()

    bootstrap, fixtures = load_data(args.offline, args.save_raw)
    table, gws = project(bootstrap, fixtures, args.horizon)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False, encoding="utf-8")

    first = f"{gws[0]}_Pts"
    print(f"Wrote {len(table)} players, GW{gws[0]}-GW{gws[-1]} -> {args.out}")
    print(f"\nTop 10 for GW{gws[0]}:")
    print(table[["Name", "Team", "Pos", "Value", first, f"{gws[0]}_xMins"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
