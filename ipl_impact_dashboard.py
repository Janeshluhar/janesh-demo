"""
+------------------------------------------------------------------------------+
║        IPL PLAYER IMPACT METRIC  ║  STREAMLIT DASHBOARD                    ║
║  Run:  streamlit run ipl_impact_dashboard.py                                ║
+------------------------------------------------------------------------------+

This dashboard:
  1. Loads & processes ALL IPL ball-by-ball CSVs from ipl_male_csv2 (1)/
  2. Computes the multi-component Impact Metric (IM) for every player
  3. Renders interactive visualisations:
       · Impact Gauge          (0–100 meter)
       · Innings Trend         (last 10 innings + pressure colours)
       · Team Impact Simulator  (simulate match between two custom XI teams)
       · Leaderboard           (top N players, filterable by role / season)
       · Head-to-Head           (compare two players)
"""

import os
import glob
import warnings
import math
import base64

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.patches as mpatches

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# --- Page config --------------------------------------------------------------
st.set_page_config(
    page_title  ="IPL Impact Metric",
    page_icon   ="🏏",
    layout      ="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "ipl_male_csv2 (1)")

# -- Load sidebar logo as base64 data URI -------
_logo_path = os.path.join(os.path.dirname(__file__), "logo.jpg")
if os.path.isfile(_logo_path):
    with open(_logo_path, "rb") as _f:
        _LOGO_B64 = base64.b64encode(_f.read()).decode()
    _LOGO_URI = f"data:image/jpeg;base64,{_LOGO_B64}"
else:
    _LOGO_URI = ""

# ------------------------------------------------------------------------------
# DATA LOADING & PROCESSING  (cached so it runs only once)
# ------------------------------------------------------------------------------

@st.cache_data(show_spinner="⏳ Loading & processing all IPL data …")
def load_all_data():
    # -- 1. Parse info CSVs -------------------------------------------------
    def parse_info(filepath):
        match_id = int(os.path.basename(filepath).replace("_info.csv", ""))
        info = {"match_id": match_id}
        teams = []
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",", 2)
                if len(parts) < 2 or parts[0] != "info":
                    continue
                key = parts[1].strip()
                val = parts[2].strip().strip('"') if len(parts) > 2 else ""
                if key == "team":
                    teams.append(val)
                elif key in ("winner", "toss_winner", "toss_decision",
                             "venue", "city", "season", "event"):
                    info[key] = val
                elif key == "winner_runs":
                    info["winner_runs"] = int(val) if val else None
                elif key == "winner_wickets":
                    info["winner_wickets"] = int(val) if val else None
                elif key == "date" and "date" not in info:
                    info["date"] = val
        info["team1"] = teams[0] if len(teams) > 0 else None
        info["team2"] = teams[1] if len(teams) > 1 else None
        return info

    info_files  = sorted(glob.glob(os.path.join(DATA_DIR, "*_info.csv")))
    matches_df  = pd.DataFrame([parse_info(f) for f in info_files])
    matches_df["date"] = pd.to_datetime(matches_df["date"], errors="coerce")

    # -- NEW: Match Importance Weight --------------------------------------
    # Parse event name to detect stage; higher-stake games get more weight
    def _match_importance(event_str):
        e = str(event_str).lower()
        if "final" in e and "semi" not in e and "quarter" not in e:
            return 1.5
        if "semi" in e or "semi final" in e:
            return 1.3
        if "qualifier" in e or "eliminator" in e or "playoff" in e:
            return 1.2
        return 1.0
    matches_df["match_importance"] = matches_df.get("event", pd.Series(
        [""] * len(matches_df), index=matches_df.index
    )).fillna("").apply(_match_importance)

    # -- 2. Load ball-by-ball CSVs ------------------------------------------
    ball_files = sorted([
        f for f in glob.glob(os.path.join(DATA_DIR, "*.csv"))
        if "_info" not in f
    ])
    dfs = []
    for f in ball_files:
        try:
            dfs.append(pd.read_csv(f, low_memory=False))
        except Exception:
            pass
    balls = pd.concat(dfs, ignore_index=True)

    for col in ["runs_off_bat","extras","wides","noballs","byes","legbyes","penalty"]:
        balls[col] = pd.to_numeric(balls[col], errors="coerce").fillna(0)
    balls["ball"]    = pd.to_numeric(balls["ball"],    errors="coerce")
    balls["innings"] = pd.to_numeric(balls["innings"], errors="coerce")
    balls["total_runs"] = balls["runs_off_bat"] + balls["extras"]
    balls["is_wicket"]  = balls["wicket_type"].notna() & (balls["wicket_type"] != "")
    balls["is_legal"]   = (balls["wides"] == 0) & (balls["noballs"] == 0)
    balls["over"]       = balls["ball"].apply(lambda x: int(x) if pd.notna(x) else np.nan)

    def stage(o):
        if pd.isna(o): return "middle"
        return "powerplay" if o <= 5 else ("death" if o >= 15 else "middle")

    balls["match_stage"] = balls["over"].apply(stage)
    balls = balls.sort_values(["match_id","innings","ball"]).reset_index(drop=True)
    balls["cum_runs"]    = balls.groupby(["match_id","innings"])["total_runs"].cumsum()
    balls["cum_wickets"] = balls.groupby(["match_id","innings"])["is_wicket"].cumsum()

    first_inn = (balls[balls["innings"]==1]
                 .groupby("match_id")["total_runs"].sum()
                 .rename("first_innings_total").reset_index())
    balls = balls.merge(first_inn, on="match_id", how="left")
    balls["first_innings_total"] = balls["first_innings_total"].fillna(0)
    balls = balls.merge(
        matches_df[["match_id","winner","toss_winner","date","season","venue","match_importance"]],
        on="match_id", how="left"
    )
    balls["batting_team_won"] = (balls["batting_team"] == balls["winner"]).astype(float)
    balls["match_importance"]  = balls["match_importance"].fillna(1.0)

    # -- 3. Batting aggregation ---------------------------------------------
    bat = balls.groupby(["match_id","innings","striker","batting_team"]).agg(
        runs_scored    = ("runs_off_bat","sum"),
        balls_faced    = ("is_legal","sum"),
        fours          = ("runs_off_bat", lambda x:(x==4).sum()),
        sixes          = ("runs_off_bat", lambda x:(x==6).sum()),
        match_importance = ("match_importance","first"),
    ).reset_index()
    bat["strike_rate"]   = np.where(bat["balls_faced"]>0,
                                    bat["runs_scored"]/bat["balls_faced"]*100, 0)
    bat["boundary_runs"] = bat["fours"]*4 + bat["sixes"]*6

    phase = (balls.groupby(["match_id","innings","striker","match_stage"])["runs_off_bat"]
             .sum().unstack(fill_value=0).reset_index()
             .rename(columns={"powerplay":"pp_runs","middle":"middle_runs","death":"death_runs"}))
    for c in ["pp_runs","middle_runs","death_runs"]:
        if c not in phase.columns: phase[c] = 0
    bat = bat.merge(phase[["match_id","innings","striker","pp_runs","middle_runs","death_runs"]],
                    on=["match_id","innings","striker"], how="left").fillna(0)

    # -- 4. Pressure index -------------------------------------------------
    last_del = (balls[balls["is_legal"]].sort_values("ball")
                .groupby(["match_id","innings","striker"]).last().reset_index()
                [["match_id","innings","striker","over","cum_runs",
                  "cum_wickets","first_innings_total","match_stage"]])

    def calc_pressure(row):
        o  = row["over"] if pd.notna(row["over"]) else 10
        wk = min(row["cum_wickets"], 9)
        cr = row["cum_runs"]
        tgt= row["first_innings_total"] + 1
        sm = {"powerplay":0.5,"middle":0.7,"death":1.0}.get(row["match_stage"],0.7)
        wp = wk / 10.0
        if row["innings"] == 2 and tgt > 0:
            ol  = max(20 - o, 0.5)
            rn  = max(tgt - cr, 0)
            rrr = rn / ol
            rp  = min(rrr / 12.0, 1.0)
            return float(np.clip(0.5*rp + 0.3*wp + 0.2*sm, 0, 1))
        return float(np.clip(0.4*wp + 0.6*sm, 0, 1))

    last_del["pressure_index"] = last_del.apply(calc_pressure, axis=1)
    last_del["pressure_level"] = last_del["pressure_index"].apply(
        lambda p: "Low" if p<0.25 else ("Medium" if p<0.50 else ("High" if p<0.75 else "Extreme"))
    )
    bat = bat.merge(last_del[["match_id","innings","striker","pressure_index","pressure_level"]],
                    on=["match_id","innings","striker"], how="left")
    bat["pressure_index"] = bat["pressure_index"].fillna(0.3)
    bat["pressure_level"] = bat["pressure_level"].fillna("Low")

    # -- 5. Batting performance score · Phase-weighted ---------------------
    # Phase weights: powerplay?0.9, middle?1.0, death?1.2
    # Phase contribution = weighted blend of each phase's run share · phase multiplier
    bat["total_runs_bat"] = bat["pp_runs"] + bat["middle_runs"] + bat["death_runs"]
    bat["total_runs_bat"] = bat["total_runs_bat"].replace(0, np.nan)
    bat["phase_multiplier"] = (
        (bat["pp_runs"]     / bat["total_runs_bat"].fillna(1)) * 0.9 +
        (bat["middle_runs"] / bat["total_runs_bat"].fillna(1)) * 1.0 +
        (bat["death_runs"]  / bat["total_runs_bat"].fillna(1)) * 1.2
    ).fillna(1.0).clip(0.9, 1.2)
    bat["total_runs_bat"] = bat["total_runs_bat"].fillna(0)

    bat["run_component"] = np.clip(bat["runs_scored"]/50, 0, 1)
    bat["sr_component"]  = np.clip(bat["strike_rate"]/150, 0, 1)
    bat["bnd_component"] = np.where(bat["runs_scored"]>0,
                                    np.clip(bat["boundary_runs"]/bat["runs_scored"],0,1), 0)
    bat["batting_perf_raw"] = (
        (0.40*bat["run_component"]
       + 0.30*bat["sr_component"]
       + 0.20*bat["bnd_component"]
       + 0.10*bat["pressure_index"]*0.15)
        * bat["phase_multiplier"]
    )

    # -- 6. Bowling aggregation --------------------------------------------
    bowl = balls.groupby(["match_id","innings","bowler","bowling_team"]).agg(
        wickets          = ("is_wicket","sum"),
        runs_conceded    = ("total_runs","sum"),
        legal_balls      = ("is_legal","sum"),
        dot_balls_bowl   = ("is_legal",
                            lambda x: (balls.loc[x.index,"runs_off_bat"]==0).sum()),
        match_importance = ("match_importance","first"),
    ).reset_index()
    bowl["overs_bowled"]    = bowl["legal_balls"] / 6
    bowl["economy_rate"]    = np.where(bowl["overs_bowled"]>0,
                                       bowl["runs_conceded"]/bowl["overs_bowled"], 0)
    bowl["dot_pct"]         = np.where(bowl["legal_balls"]>0,
                                       bowl["dot_balls_bowl"]/bowl["legal_balls"], 0)
    bowl["wkt_component"]   = np.clip(bowl["wickets"]/3, 0, 1)
    bowl["eco_component"]   = np.clip(1 - bowl["economy_rate"]/12.0, 0, 1)
    bowl["bowling_perf_raw"]= 0.50*bowl["wkt_component"]+0.30*bowl["eco_component"]+0.20*bowl["dot_pct"]

    # -- NEW: Opposition Strength ------------------------------------------
    # For batters: career economy rate of each bowler they faced → low eco = strong attack
    # For bowlers: career strike rate of each batter they bowled at → high SR = strong batting
    bowler_career_eco = (
        balls[balls["is_legal"] & (balls["overs_bowled"] if "overs_bowled" in balls.columns else pd.Series([True]*len(balls)))]
        .pipe(lambda d: d.groupby("bowler").agg(
            _rc=("total_runs","sum"), _lb=("is_legal","sum")
        ))
        .assign(career_eco=lambda d: d["_rc"] / (d["_lb"]/6).clip(lower=0.1))
        .reset_index()[["bowler","career_eco"]]
    )
    batter_career_sr = (
        balls[balls["is_legal"]]
        .groupby("striker").agg(_rs=("runs_off_bat","sum"), _bf=("is_legal","sum"))
        .assign(career_sr=lambda d: np.where(d["_bf"]>0, d["_rs"]/d["_bf"]*100, 0))
        .reset_index()[["striker","career_sr"]]
    )

    # Batter opposition strength = avg economy of bowlers faced per innings
    bat_opp = (
        balls[["match_id","innings","striker","bowler"]]
        .merge(bowler_career_eco, on="bowler", how="left")
        .groupby(["match_id","innings","striker"])["career_eco"]
        .mean().reset_index()
        .rename(columns={"career_eco":"opp_avg_eco"})
    )
    # Lower eco = stronger opposition → invert: opp_strength = 1 - eco/12
    bat_opp["bat_opp_strength"] = np.clip(1 - bat_opp["opp_avg_eco"]/12.0, 0, 1)
    # Normalize 0–1 across all innings
    _mn, _mx = bat_opp["bat_opp_strength"].min(), bat_opp["bat_opp_strength"].max()
    bat_opp["bat_opp_strength"] = (bat_opp["bat_opp_strength"] - _mn) / max(_mx - _mn, 1e-6)

    # Bowler opposition strength = avg SR of batters faced per innings
    bowl_opp = (
        balls[["match_id","innings","bowler","striker"]]
        .merge(batter_career_sr, on="striker", how="left")
        .groupby(["match_id","innings","bowler"])["career_sr"]
        .mean().reset_index()
        .rename(columns={"career_sr":"opp_avg_sr"})
    )
    # Higher SR opponents = stronger batting lineup
    bowl_opp["bowl_opp_strength"] = np.clip(bowl_opp["opp_avg_sr"]/150.0, 0, 1)
    _mn2, _mx2 = bowl_opp["bowl_opp_strength"].min(), bowl_opp["bowl_opp_strength"].max()
    bowl_opp["bowl_opp_strength"] = (bowl_opp["bowl_opp_strength"] - _mn2) / max(_mx2 - _mn2, 1e-6)

    bat  = bat.merge(bat_opp[["match_id","innings","striker","bat_opp_strength"]],
                     on=["match_id","innings","striker"], how="left")
    bat["bat_opp_strength"] = bat["bat_opp_strength"].fillna(0.5)

    bowl = bowl.merge(bowl_opp[["match_id","innings","bowler","bowl_opp_strength"]],
                      on=["match_id","innings","bowler"], how="left")
    bowl["bowl_opp_strength"] = bowl["bowl_opp_strength"].fillna(0.5)

    # -- 7. Win Probability model ------------------------------------------
    wp = balls.copy()
    wp["overs_left"]   = np.clip(20 - wp["over"], 0.1, 20)
    wp["runs_needed"]  = np.where(wp["innings"]==2,
                                  np.maximum(wp["first_innings_total"]+1-wp["cum_runs"],0), 0)
    wp["rrr"]          = wp["runs_needed"] / wp["overs_left"]
    wp["wickets_left"] = 10 - wp["cum_wickets"]
    WP_FEATS = ["innings","over","cum_runs","cum_wickets","wickets_left","rrr","overs_left"]
    wp_clean = wp[WP_FEATS+["batting_team_won"]].dropna()
    wp_clean = wp_clean[wp_clean["batting_team_won"].isin([0,1])]
    X_ = wp_clean[WP_FEATS].astype(float)
    y_ = wp_clean["batting_team_won"].astype(int)
    Xtr, _, ytr, _ = train_test_split(X_, y_, test_size=0.85, random_state=42, stratify=y_)
    sc_wp = StandardScaler(); Xtr_s = sc_wp.fit_transform(Xtr)
    wp_model = LogisticRegression(max_iter=500, C=1.0, random_state=42)
    wp_model.fit(Xtr_s, ytr)

    wp_feat   = wp[WP_FEATS].fillna(0).astype(float)
    wp["win_prob"]   = wp_model.predict_proba(sc_wp.transform(wp_feat))[:,1]
    wp["delta_wp"]   = wp.groupby(["match_id","innings"])["win_prob"].diff().fillna(0)

    def sig(x, s=3): return 1/(1+np.exp(-x*s))

    bat_wpa = (wp.groupby(["match_id","innings","striker"])["delta_wp"]
               .sum().reset_index().rename(columns={"striker":"player","delta_wp":"wpa"}))
    bat_wpa["wpa_norm"] = sig(bat_wpa["wpa"])

    bowl_wpa = (wp.groupby(["match_id","innings","bowler"])["delta_wp"]
                .sum().apply(lambda x:-x).reset_index()
                .rename(columns={"bowler":"player","delta_wp":"wpa"}))
    bowl_wpa["wpa_norm"] = sig(bowl_wpa["wpa"])

    # -- 8. Clutch factor --------------------------------------------------
    oa  = bat.groupby("striker")["batting_perf_raw"].mean().rename("overall_avg").reset_index()
    ha  = (bat[bat["pressure_level"].isin(["High","Extreme"])]
           .groupby("striker")["batting_perf_raw"].mean().rename("high_avg").reset_index())
    cl  = oa.merge(ha, on="striker", how="left")
    cl["high_avg"]     = cl["high_avg"].fillna(cl["overall_avg"])
    cl["clutch_ratio"] = np.clip(cl["high_avg"] / (cl["overall_avg"]+1e-4), 0, 3)
    cl["clutch_score"] = np.clip(cl["clutch_ratio"] / 3.0, 0, 1)

    # -- 9. Per-innings IM · Updated Formula --------------------------------
    # IM = 0.30×Perf + 0.20×Pressure + 0.20×WPA + 0.10×Clutch
    #     + 0.10×OppStrength + 0.05×MatchImportance + 0.05×FormRecovery
    # Match importance maps 1.0?0.40, 1.2?0.55, 1.3?0.65, 1.5?0.80
    def _imp_to_norm(v):
        return float(np.clip((v - 1.0) / 0.5 * 0.8 + 0.4, 0.4, 0.9))

    bat_im = bat.copy()
    bat_im = bat_im.merge(bat_wpa[["match_id","innings","player","wpa_norm"]],
                          left_on=["match_id","innings","striker"],
                          right_on=["match_id","innings","player"], how="left").drop(columns=["player"])
    bat_im["wpa_norm"] = bat_im["wpa_norm"].fillna(0.5)
    bat_im = bat_im.merge(cl[["striker","clutch_score"]], on="striker", how="left")
    bat_im["clutch_score"]    = bat_im["clutch_score"].fillna(0.33)
    bat_im["match_imp_norm"]  = bat_im["match_importance"].apply(_imp_to_norm)
    bat_im = bat_im.merge(matches_df[["match_id","date"]], on="match_id", how="left")

    # Form Recovery · rolling 5-innings deviation (computed after sort)
    bat_im = bat_im.sort_values(["striker","date","innings"])
    bat_im["_roll5_mean"] = (bat_im.groupby("striker")["batting_perf_raw"]
                             .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
                             .fillna(bat_im["batting_perf_raw"]))
    bat_im["_roll5_std"]  = (bat_im.groupby("striker")["batting_perf_raw"]
                             .transform(lambda s: s.shift(1).rolling(5, min_periods=1).std())
                             .fillna(0.05) + 0.01)
    bat_im["form_recovery"] = np.clip(
        (bat_im["batting_perf_raw"] - bat_im["_roll5_mean"]) / bat_im["_roll5_std"] / 4 + 0.5,
        0, 1
    )

    bat_im["innings_im_raw"] = (
        0.30 * bat_im["batting_perf_raw"]
      + 0.20 * bat_im["pressure_index"]
      + 0.20 * bat_im["wpa_norm"]
      + 0.10 * bat_im["clutch_score"]
      + 0.10 * bat_im["bat_opp_strength"]
      + 0.05 * bat_im["match_imp_norm"]
      + 0.05 * bat_im["form_recovery"]
    )

    bowl_im = bowl.copy()
    bowl_im = bowl_im.merge(bowl_wpa[["match_id","innings","player","wpa_norm"]],
                            left_on=["match_id","innings","bowler"],
                            right_on=["match_id","innings","player"], how="left").drop(columns=["player"])
    bowl_im["wpa_norm"]       = bowl_im["wpa_norm"].fillna(0.5)
    bowl_im["pressure_index"] = 0.35
    bowl_im["clutch_score"]   = 0.33
    bowl_im["match_imp_norm"] = bowl_im["match_importance"].apply(_imp_to_norm)
    bowl_im = bowl_im.merge(matches_df[["match_id","date"]], on="match_id", how="left")

    bowl_im = bowl_im.sort_values(["bowler","date","innings"])
    bowl_im["_roll5_mean"] = (bowl_im.groupby("bowler")["bowling_perf_raw"]
                              .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
                              .fillna(bowl_im["bowling_perf_raw"]))
    bowl_im["_roll5_std"]  = (bowl_im.groupby("bowler")["bowling_perf_raw"]
                              .transform(lambda s: s.shift(1).rolling(5, min_periods=1).std())
                              .fillna(0.05) + 0.01)
    bowl_im["form_recovery"] = np.clip(
        (bowl_im["bowling_perf_raw"] - bowl_im["_roll5_mean"]) / bowl_im["_roll5_std"] / 4 + 0.5,
        0, 1
    )

    bowl_im["innings_im_raw"] = (
        0.30 * bowl_im["bowling_perf_raw"]
      + 0.20 * bowl_im["pressure_index"]
      + 0.20 * bowl_im["wpa_norm"]
      + 0.10 * bowl_im["clutch_score"]
      + 0.10 * bowl_im["bowl_opp_strength"]
      + 0.05 * bowl_im["match_imp_norm"]
      + 0.05 * bowl_im["form_recovery"]
    )

    # -- 10. Recency-weighted rolling IM -----------------------------------
    # Exponential decay: most recent innings gets ~38%, second ~23%, third ~14%, etc.
    # This ensures recent form dominates the Impact Metric calculation.
    _decay = 0.6   # decay factor — lower = steeper recency bias
    WEIGHTS = np.array([_decay**i for i in range(10)])  # [1.0, 0.6, 0.36, 0.216, ...]
    WEIGHTS = WEIGHTS / WEIGHTS.sum()

    def wt_avg(s, w=WEIGHTS, n=10):
        v = s.tail(n).values[::-1]; w2 = w[:len(v)]; w2=w2/w2.sum()
        return float(np.dot(v, w2))

    bat_roll = (bat_im.sort_values(["striker","date","innings"])
                .groupby("striker")["innings_im_raw"].apply(wt_avg)
                .reset_index().rename(columns={"innings_im_raw":"raw","striker":"player"}))
    bowl_roll= (bowl_im.sort_values(["bowler","date","innings"])
                .groupby("bowler")["innings_im_raw"].apply(wt_avg)
                .reset_index().rename(columns={"innings_im_raw":"raw","bowler":"player"}))

    # -- Consistency Score (1/std of last-10 innings IM, normalised 0–1) ---
    bat_cons = (bat_im.sort_values(["striker","date","innings"])
                .groupby("striker")["innings_im_raw"]
                .apply(lambda s: s.tail(10).std())
                .reset_index().rename(columns={"striker":"player","innings_im_raw":"bat_std"}))
    bat_cons["consistency_score"] = np.clip(1 / (bat_cons["bat_std"].fillna(0) + 0.01), 0, 200)
    _cmx_b = bat_cons["consistency_score"].quantile(0.95) or 1.0
    bat_cons["consistency_score"] = np.clip(bat_cons["consistency_score"] / _cmx_b, 0, 1).round(4)

    bowl_cons = (bowl_im.sort_values(["bowler","date","innings"])
                 .groupby("bowler")["innings_im_raw"]
                 .apply(lambda s: s.tail(10).std())
                 .reset_index().rename(columns={"bowler":"player","innings_im_raw":"bowl_std"}))
    bowl_cons["consistency_score"] = np.clip(1 / (bowl_cons["bowl_std"].fillna(0) + 0.01), 0, 200)
    _cmx_w = bowl_cons["consistency_score"].quantile(0.95) or 1.0
    bowl_cons["consistency_score"] = np.clip(bowl_cons["consistency_score"] / _cmx_w, 0, 1).round(4)

    # -- Match Impact Timeline (per-player per-match per-over batting WPA) -
    timeline_df = (wp.groupby(["match_id","innings","striker","over"]).agg(
        delta_wp   = ("delta_wp","sum"),
        runs       = ("runs_off_bat","sum"),
        boundaries = ("runs_off_bat", lambda x: int((x >= 4).sum())),
        wickets    = ("is_wicket","sum"),
    ).reset_index().rename(columns={"striker":"player"}))
    timeline_df["cum_wp"] = (timeline_df
                             .sort_values(["match_id","innings","over"])
                             .groupby(["match_id","innings","player"])["delta_wp"]
                             .cumsum())

    # Component averages per player for the analytics page
    bat_comp = bat_im.groupby("striker").agg(
        avg_perf       = ("batting_perf_raw","mean"),
        avg_pressure   = ("pressure_index","mean"),
        avg_wpa        = ("wpa_norm","mean"),
        avg_clutch     = ("clutch_score","mean"),
        avg_opp_str    = ("bat_opp_strength","mean"),
        avg_match_imp  = ("match_imp_norm","mean"),
        avg_form_rec   = ("form_recovery","mean"),
    ).reset_index().rename(columns={"striker":"player"})

    bowl_comp = bowl_im.groupby("bowler").agg(
        avg_perf       = ("bowling_perf_raw","mean"),
        avg_pressure   = ("pressure_index","mean"),
        avg_wpa        = ("wpa_norm","mean"),
        avg_clutch     = ("clutch_score","mean"),
        avg_opp_str    = ("bowl_opp_strength","mean"),
        avg_match_imp  = ("match_imp_norm","mean"),
        avg_form_rec   = ("form_recovery","mean"),
    ).reset_index().rename(columns={"bowler":"player"})

    def to_0_100(series):
        mu = series.mean(); sig_ = series.std()+1e-8
        z  = (series - mu)/sig_
        return (1/(1+np.exp(-z*1.5)))*100

    bat_roll["impact_metric"]  = to_0_100(bat_roll["raw"]).round(2)
    bowl_roll["impact_metric"] = to_0_100(bowl_roll["raw"]).round(2)

    # -- 11. Combine into final table --------------------------------------
    rows = []
    for player in set(bat_roll["player"]) | set(bowl_roll["player"]):
        br  = bat_roll[bat_roll["player"]==player]
        bwr = bowl_roll[bowl_roll["player"]==player]
        bv  = br["impact_metric"].values[0]  if len(br)>0  else None
        bwv = bwr["impact_metric"].values[0] if len(bwr)>0 else None
        bi  = len(bat[bat["striker"]==player])
        bwi = len(bowl[bowl["bowler"]==player])

        # Role classification using participation ratios
        # total_matches = distinct match appearances (bat or bowl)
        _bat_matches  = set(bat.loc[bat["striker"]==player, "match_id"])
        _bowl_matches = set(bowl.loc[bowl["bowler"]==player,  "match_id"])
        total_matches = max(len(_bat_matches | _bowl_matches), 1)   # avoid /0

        bat_ratio  = bi  / total_matches
        bowl_ratio = bwi / total_matches

        if bi >= 5 and bwi >= 5 and bat_ratio >= 0.35 and bowl_ratio >= 0.35:
            # Genuine all-rounder: meaningful contribution in both disciplines
            role = "All-Rounder"; comb = 0.55 * (bv or 50) + 0.45 * (bwv or 50)
        elif bowl_ratio >= 0.65:
            # Predominantly a bowler
            role = "Bowler";  comb = bwv or 50
        elif bat_ratio >= 0.65:
            # Predominantly a batter
            role = "Batter";  comb = bv  or 50
        else:
            # Fallback
            role = "All-Rounder"; comb = 0.55 * (bv or 50) + 0.45 * (bwv or 50)
        ti=bi+bwi
        if ti<5:
            sh=(5-ti)/5*0.3; comb=comb*(1-sh)+50*sh

        # Gather component averages
        bc = bat_comp[bat_comp["player"]==player]
        wc = bowl_comp[bowl_comp["player"]==player]
        _bperf = bc["avg_perf"].values[0]   if not bc.empty else 0.3
        _wperf = wc["avg_perf"].values[0]   if not wc.empty else 0.3
        _opp   = (bc["avg_opp_str"].values[0] if not bc.empty else
                  wc["avg_opp_str"].values[0] if not wc.empty else 0.5)
        _imp   = (bc["avg_match_imp"].values[0] if not bc.empty else
                  wc["avg_match_imp"].values[0] if not wc.empty else 0.4)
        _frec  = (bc["avg_form_rec"].values[0] if not bc.empty else
                  wc["avg_form_rec"].values[0] if not wc.empty else 0.5)

        _cbc = bat_cons[bat_cons["player"]==player]
        _cwc = bowl_cons[bowl_cons["player"]==player]
        _cscore = (float(_cbc["consistency_score"].values[0]) if not _cbc.empty else
                   float(_cwc["consistency_score"].values[0]) if not _cwc.empty else 0.5)

        rows.append(dict(
            player=player, role=role,
            bat_im=round(bv,2)   if bv  is not None else None,
            bowl_im=round(bwv,2) if bwv is not None else None,
            bat_innings=bi, bowl_innings=bwi,
            impact_metric=round(comb,2),
            avg_perf=round(float(_bperf if bi>bwi else _wperf),4),
            opp_strength=round(float(_opp),4),
            match_importance=round(float(_imp),4),
            form_recovery=round(float(_frec),4),
            consistency_score=round(_cscore, 4),
        ))
    final = (pd.DataFrame(rows)
             .sort_values("impact_metric",ascending=False)
             .reset_index(drop=True))
    final["rank"] = final.index+1

    return matches_df, balls, bat_im, bowl_im, bat, bowl, final, timeline_df

# ------------------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------------------
matches_df, balls_df, bat_im, bowl_im, bat_agg, bowl_agg, final_df, timeline_df = load_all_data()

# ------------------------------------------------------------------------------
# PREMIUM CSS · Dark Sports Analytics Theme
# ------------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=Orbitron:wght@700;900&display=swap');

/* -- Custom SVG Cricket Icon System -- */
.ic { display: inline-block; width: 1em; height: 1em; vertical-align: -0.12em; background-size: contain; background-repeat: no-repeat; background-position: center; }
.ic-lg { width: 1.6em; height: 1.6em; vertical-align: -0.25em; }
.ic-xl { width: 2em; height: 2em; vertical-align: -0.35em; }
.ic-2x { width: 2.4em; height: 2.4em; vertical-align: -0.4em; }

/* Cricket Bat — batting icon */
.ic-bat { background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect x='24' y='6' width='12' height='34' rx='3' ry='3' fill='%2300e5ff' opacity='0.13' stroke='%2300e5ff' stroke-width='2'/%3E%3Crect x='28' y='40' width='4' height='12' rx='1' fill='none' stroke='%2300e5ff' stroke-width='2'/%3E%3Crect x='26' y='52' width='8' height='4' rx='2' fill='none' stroke='%2300e5ff' stroke-width='2'/%3E%3Cline x1='28' y1='14' x2='32' y2='14' stroke='%2300e5ff' stroke-width='1' opacity='0.35'/%3E%3Cline x1='28' y1='20' x2='32' y2='20' stroke='%2300e5ff' stroke-width='1' opacity='0.35'/%3E%3Cline x1='28' y1='26' x2='32' y2='26' stroke='%2300e5ff' stroke-width='1' opacity='0.35'/%3E%3C/svg%3E"); }

/* Cricket Ball — bowling icon */
.ic-ball { background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Ccircle cx='32' cy='32' r='22' fill='none' stroke='%2339ff14' stroke-width='2.5'/%3E%3Cpath d='M18 18c4 6 4 16 0 28M46 18c-4 6-4 16 0 28' fill='none' stroke='%2339ff14' stroke-width='1.5' stroke-dasharray='3 2'/%3E%3C/svg%3E"); }

/* Stumps — wicket icon */
.ic-stumps { background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Cline x1='22' y1='12' x2='22' y2='54' stroke='%23ffd700' stroke-width='2.5' stroke-linecap='round'/%3E%3Cline x1='32' y1='12' x2='32' y2='54' stroke='%23ffd700' stroke-width='2.5' stroke-linecap='round'/%3E%3Cline x1='42' y1='12' x2='42' y2='54' stroke='%23ffd700' stroke-width='2.5' stroke-linecap='round'/%3E%3Cpath d='M18 14h10M28 12h8M38 14h10' stroke='%23ffd700' stroke-width='2' stroke-linecap='round'/%3E%3C/svg%3E"); }

/* Trophy — champion/leaderboard icon */
.ic-trophy { background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Cpath d='M20 10h24v18c0 8-5 14-12 14s-12-6-12-14z' fill='none' stroke='%23ffd700' stroke-width='2.5' stroke-linejoin='round'/%3E%3Cpath d='M20 16h-6c-2 0-4 2-4 4v2c0 4 3 8 8 8h2M44 16h6c2 0 4 2 4 4v2c0 4-3 8-8 8h-2' fill='none' stroke='%23ffd700' stroke-width='2' stroke-linejoin='round'/%3E%3Crect x='26' y='42' width='12' height='4' rx='1' fill='%23ffd700' opacity='0.5'/%3E%3Crect x='22' y='46' width='20' height='6' rx='2' fill='none' stroke='%23ffd700' stroke-width='2'/%3E%3C/svg%3E"); }

/* Crosshair/Target — deep-dive icon */
.ic-target { background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Ccircle cx='32' cy='32' r='22' fill='none' stroke='%23ff6b9d' stroke-width='2'/%3E%3Ccircle cx='32' cy='32' r='14' fill='none' stroke='%23ff6b9d' stroke-width='1.5'/%3E%3Ccircle cx='32' cy='32' r='6' fill='none' stroke='%23ff6b9d' stroke-width='2'/%3E%3Ccircle cx='32' cy='32' r='2' fill='%23ff6b9d'/%3E%3Cline x1='32' y1='6' x2='32' y2='16' stroke='%23ff6b9d' stroke-width='1.5'/%3E%3Cline x1='32' y1='48' x2='32' y2='58' stroke='%23ff6b9d' stroke-width='1.5'/%3E%3Cline x1='6' y1='32' x2='16' y2='32' stroke='%23ff6b9d' stroke-width='1.5'/%3E%3Cline x1='48' y1='32' x2='58' y2='32' stroke='%23ff6b9d' stroke-width='1.5'/%3E%3C/svg%3E"); }

/* Pitch/Field — pressure icon */
.ic-pitch { background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Cellipse cx='32' cy='32' rx='26' ry='18' fill='none' stroke='%2300e5ff' stroke-width='2'/%3E%3Crect x='24' y='22' width='16' height='20' rx='2' fill='none' stroke='%2300e5ff' stroke-width='1.5' stroke-dasharray='3 2'/%3E%3Cline x1='32' y1='14' x2='32' y2='50' stroke='%2300e5ff' stroke-width='1' opacity='0.4'/%3E%3Ccircle cx='32' cy='32' r='2' fill='%2300e5ff'/%3E%3C/svg%3E"); }

/* Two Bats Crossed — head-to-head icon */
.ic-versus { background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Cpath d='M14 50L38 10' stroke='%23ff9f1c' stroke-width='3' stroke-linecap='round'/%3E%3Cpath d='M10 46l8 8' stroke='%23ff9f1c' stroke-width='2.5' stroke-linecap='round'/%3E%3Cpath d='M50 50L26 10' stroke='%23a855f7' stroke-width='3' stroke-linecap='round'/%3E%3Cpath d='M46 46l8 8' stroke='%23a855f7' stroke-width='2.5' stroke-linecap='round'/%3E%3Ccircle cx='32' cy='30' r='4' fill='none' stroke='%2300e5ff' stroke-width='1.5'/%3E%3C/svg%3E"); }

/* Brain/AI — smart predictor icon */
.ic-brain { background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Cpath d='M32 8c-10 0-18 6-18 16 0 6 3 10 8 14v8c0 2 2 4 4 4h12c2 0 4-2 4-4v-8c5-4 8-8 8-14 0-10-8-16-18-16z' fill='none' stroke='%2300e5ff' stroke-width='2.5' stroke-linejoin='round'/%3E%3Cpath d='M26 50h12M28 54h8' stroke='%2300e5ff' stroke-width='2' stroke-linecap='round'/%3E%3Ccircle cx='26' cy='24' r='2' fill='%2300e5ff'/%3E%3Ccircle cx='38' cy='24' r='2' fill='%2300e5ff'/%3E%3Cpath d='M26 24l6 4 6-4' fill='none' stroke='%2300e5ff' stroke-width='1.5'/%3E%3C/svg%3E"); }

/* Microscope/Analytics — advanced analytics icon */
.ic-analytics { background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Ccircle cx='28' cy='24' r='14' fill='none' stroke='%23a855f7' stroke-width='2.5'/%3E%3Cline x1='38' y1='34' x2='52' y2='48' stroke='%23a855f7' stroke-width='3' stroke-linecap='round'/%3E%3Cpath d='M20 22h16M22 28h12' stroke='%23a855f7' stroke-width='1.5' stroke-linecap='round' opacity='0.6'/%3E%3Cpath d='M48 52l4-4' stroke='%23a855f7' stroke-width='3' stroke-linecap='round'/%3E%3C/svg%3E"); }

/* Shield — defence/manipulation-resistant icon */
.ic-shield { background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Cpath d='M32 6L10 18v16c0 14 10 22 22 26 12-4 22-12 22-26V18z' fill='none' stroke='%2339ff14' stroke-width='2.5' stroke-linejoin='round'/%3E%3Cpath d='M24 32l6 6 12-14' fill='none' stroke='%2339ff14' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E"); }

/* Player/Helmet — player icon */
.ic-player { background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Cpath d='M20 28c0-8 5-16 12-16s12 8 12 16' fill='none' stroke='%2300e5ff' stroke-width='2.5' stroke-linecap='round'/%3E%3Crect x='16' y='28' width='32' height='10' rx='5' fill='none' stroke='%2300e5ff' stroke-width='2'/%3E%3Crect x='38' y='30' width='10' height='6' rx='2' fill='none' stroke='%2300e5ff' stroke-width='1.5'/%3E%3Cpath d='M22 38v6c0 4 4 8 10 8s10-4 10-8v-6' fill='none' stroke='%2300e5ff' stroke-width='1.5' opacity='0.5'/%3E%3C/svg%3E"); }

/* Star — all-rounder icon */
.ic-star { background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Cpath d='M32 8l7 14 16 2-12 11 3 16-14-7-14 7 3-16L9 24l16-2z' fill='none' stroke='%23ffd700' stroke-width='2.5' stroke-linejoin='round'/%3E%3C/svg%3E"); }

/* Chart — leaderboard icon */
.ic-chart { background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect x='8' y='36' width='10' height='20' rx='2' fill='none' stroke='%2300e5ff' stroke-width='2'/%3E%3Crect x='27' y='16' width='10' height='40' rx='2' fill='none' stroke='%2300e5ff' stroke-width='2'/%3E%3Crect x='46' y='26' width='10' height='30' rx='2' fill='none' stroke='%2300e5ff' stroke-width='2'/%3E%3Cpath d='M13 32l14-12 14 6' fill='none' stroke='%23ffd700' stroke-width='1.5' stroke-linecap='round'/%3E%3Ccircle cx='13' cy='32' r='2' fill='%23ffd700'/%3E%3Ccircle cx='27' cy='20' r='2' fill='%23ffd700'/%3E%3Ccircle cx='41' cy='26' r='2' fill='%23ffd700'/%3E%3C/svg%3E"); }

/* Lightning — power/live icon */
.ic-bolt { background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Cpath d='M36 6L16 34h12L24 58 48 28H36z' fill='none' stroke='%2300e5ff' stroke-width='2.5' stroke-linejoin='round'/%3E%3C/svg%3E"); }

/* Six/Boundary — boundary icon */
.ic-six { background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Ccircle cx='32' cy='20' r='6' fill='none' stroke='%23ffd700' stroke-width='2'/%3E%3Cpath d='M32 26v20M26 34l6-6 6 6M22 48h20' fill='none' stroke='%23ffd700' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3Cpath d='M14 10c6 2 10 8 10 14M50 10c-6 2-10 8-10 14' fill='none' stroke='%23ffd700' stroke-width='1.5' opacity='0.4' stroke-linecap='round'/%3E%3C/svg%3E"); }

/* Medal icons */
.ic-gold { background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Ccircle cx='32' cy='36' r='16' fill='none' stroke='%23ffd700' stroke-width='2.5'/%3E%3Cpath d='M24 6l8 20M40 6l-8 20' stroke='%23ffd700' stroke-width='2' stroke-linecap='round'/%3E%3Ctext x='32' y='42' text-anchor='middle' font-family='Orbitron' font-weight='900' font-size='16' fill='%23ffd700'>1%3C/text%3E%3C/svg%3E"); }
.ic-silver { background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Ccircle cx='32' cy='36' r='16' fill='none' stroke='%23c0c0c0' stroke-width='2.5'/%3E%3Cpath d='M24 6l8 20M40 6l-8 20' stroke='%23c0c0c0' stroke-width='2' stroke-linecap='round'/%3E%3Ctext x='32' y='42' text-anchor='middle' font-family='Orbitron' font-weight='900' font-size='16' fill='%23c0c0c0'>2%3C/text%3E%3C/svg%3E"); }
.ic-bronze { background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Ccircle cx='32' cy='36' r='16' fill='none' stroke='%23cd7f32' stroke-width='2.5'/%3E%3Cpath d='M24 6l8 20M40 6l-8 20' stroke='%23cd7f32' stroke-width='2' stroke-linecap='round'/%3E%3Ctext x='32' y='42' text-anchor='middle' font-family='Orbitron' font-weight='900' font-size='16' fill='%23cd7f32'>3%3C/text%3E%3C/svg%3E"); }

/* Clipboard/Table icon */
.ic-table { background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect x='12' y='8' width='40' height='48' rx='4' fill='none' stroke='%2300e5ff' stroke-width='2'/%3E%3Cline x1='20' y1='22' x2='44' y2='22' stroke='%2300e5ff' stroke-width='1.5'/%3E%3Cline x1='20' y1='32' x2='44' y2='32' stroke='%2300e5ff' stroke-width='1.5'/%3E%3Cline x1='20' y1='42' x2='36' y2='42' stroke='%2300e5ff' stroke-width='1.5'/%3E%3Crect x='24' y='4' width='16' height='8' rx='2' fill='none' stroke='%2300e5ff' stroke-width='2'/%3E%3C/svg%3E"); }

/* Warning icon */
.ic-warn { background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Cpath d='M32 8L6 54h52z' fill='none' stroke='%23ffd700' stroke-width='2.5' stroke-linejoin='round'/%3E%3Cline x1='32' y1='24' x2='32' y2='38' stroke='%23ffd700' stroke-width='2.5' stroke-linecap='round'/%3E%3Ccircle cx='32' cy='46' r='2' fill='%23ffd700'/%3E%3C/svg%3E"); }

/* Wicket/Out — cross icon */
.ic-out { background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Ccircle cx='32' cy='32' r='20' fill='none' stroke='%23ff3b3b' stroke-width='2'/%3E%3Cline x1='22' y1='22' x2='42' y2='42' stroke='%23ff3b3b' stroke-width='2.5' stroke-linecap='round'/%3E%3Cline x1='42' y1='22' x2='22' y2='42' stroke='%23ff3b3b' stroke-width='2.5' stroke-linecap='round'/%3E%3C/svg%3E"); }

/* Flame — fire/death overs icon */
.ic-flame { background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Cpath d='M32 6c0 12-14 18-14 30 0 10 6 18 14 20 8-2 14-10 14-20 0-12-14-18-14-30z' fill='none' stroke='%23ff6b9d' stroke-width='2.5' stroke-linejoin='round'/%3E%3Cpath d='M32 28c0 6-6 10-6 16s3 8 6 10c3-2 6-4 6-10s-6-10-6-16z' fill='none' stroke='%23ff6b9d' stroke-width='1.5' opacity='0.6'/%3E%3C/svg%3E"); }

/* Cycle — form recovery / cycle icon */
.ic-cycle { background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Cpath d='M40 16a18 18 0 11-20 4' fill='none' stroke='%2339ff14' stroke-width='2.5' stroke-linecap='round'/%3E%3Cpath d='M16 18l4 4 4-4' fill='none' stroke='%2339ff14' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3Cpath d='M24 48a18 18 0 1120-4' fill='none' stroke='%2339ff14' stroke-width='2.5' stroke-linecap='round'/%3E%3Cpath d='M48 46l-4-4-4 4' fill='none' stroke='%2339ff14' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E"); }

/* Scales — balanced/contested icon */
.ic-scales { background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Cline x1='32' y1='8' x2='32' y2='52' stroke='%23ffd700' stroke-width='2.5' stroke-linecap='round'/%3E%3Cline x1='12' y1='20' x2='52' y2='20' stroke='%23ffd700' stroke-width='2' stroke-linecap='round'/%3E%3Cpath d='M12 20l6 16h-12z' fill='none' stroke='%23ffd700' stroke-width='2' stroke-linejoin='round'/%3E%3Cpath d='M52 20l6 16h-12z' fill='none' stroke='%23ffd700' stroke-width='2' stroke-linejoin='round'/%3E%3Crect x='24' y='52' width='16' height='4' rx='2' fill='%23ffd700' opacity='0.4'/%3E%3C/svg%3E"); }

/* Lightbulb — insight icon */
.ic-bulb { background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Cpath d='M32 8c-10 0-18 8-18 18 0 6 3 11 8 14v6c0 2 2 4 4 4h12c2 0 4-2 4-4v-6c5-3 8-8 8-14 0-10-8-18-18-18z' fill='none' stroke='%2339ff14' stroke-width='2.5'/%3E%3Cpath d='M26 50h12M28 54h8M28 58h8' stroke='%2339ff14' stroke-width='2' stroke-linecap='round'/%3E%3C/svg%3E"); }

/* -- Global reset -- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #050b18 !important;
    color: #e0e6f0 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0.6rem !important; max-width: 1400px !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #08112a; }
::-webkit-scrollbar-thumb { background: #00e5ff55; border-radius: 3px; }

/* -- Sidebar -- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,
        rgba(4,10,24,0.92) 0%,
        rgba(7,15,38,0.88) 40%,
        rgba(4,10,24,0.92) 100%) !important;
    backdrop-filter: blur(24px) saturate(1.4) !important;
    -webkit-backdrop-filter: blur(24px) saturate(1.4) !important;
    border-right: 1px solid rgba(0,229,255,0.15) !important;
    box-shadow: 4px 0 32px rgba(0,0,0,0.45), inset -1px 0 0 rgba(0,229,255,0.06) !important;
}
[data-testid="stSidebar"] * { color: #c5d8f5 !important; }

/* -- Hide only the inner close/collapse arrow inside sidebar -- */
[data-testid="stSidebarCollapseButton"] { display: none !important; }

/* -- Always keep sidebar visible at a fixed width -- */
[data-testid="stSidebar"] {
    min-width: 270px !important;
    max-width: 290px !important;
    transform: none !important;
    visibility: visible !important;
    opacity: 1 !important;
}

/* -- Make the collapsed-control (re-open arrow) neon-styled and always visible -- */
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    background: rgba(0,229,255,0.12) !important;
    border: 1px solid rgba(0,229,255,0.35) !important;
    border-radius: 0 8px 8px 0 !important;
}
[data-testid="collapsedControl"] svg { fill: #00e5ff !important; }

/* -- Hide copy / clipboard button on ALL sidebar buttons -- */
[data-testid="stSidebar"] .stButton > button > div[data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] .stButton > button .copy-btn,
[data-testid="stSidebar"] .stButton > button [data-testid="stActionButtonIcon"],
[data-testid="stSidebar"] .stCopyButton,
[data-testid="stSidebar"] button[kind="icon"],
[data-testid="stSidebar"] .stButton > button svg {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
}

/* -- Sidebar nav buttons — Glassmorphism -- */
[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    background: rgba(255,255,255,0.03) !important;
    backdrop-filter: blur(8px) !important;
    -webkit-backdrop-filter: blur(8px) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    color: #7a9cc4 !important;
    font-size: 0.80rem !important;
    font-weight: 500 !important;
    text-align: left !important;
    padding: 11px 14px !important;
    margin-bottom: 4px !important;
    transition: all 0.25s cubic-bezier(0.4,0,0.2,1) !important;
    cursor: pointer !important;
    letter-spacing: 0.3px !important;
    white-space: normal !important;
    word-break: keep-all !important;
    overflow-wrap: break-word !important;
    line-height: 1.35 !important;
    position: relative !important;
    overflow: hidden !important;
}
[data-testid="stSidebar"] .stButton > button::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important; left: -100% !important;
    width: 100% !important; height: 100% !important;
    background: linear-gradient(90deg, transparent, rgba(0,229,255,0.06), transparent) !important;
    transition: left 0.5s ease !important;
}
[data-testid="stSidebar"] .stButton > button:hover::before {
    left: 100% !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(0,229,255,0.07) !important;
    border-color: rgba(0,229,255,0.28) !important;
    color: #00e5ff !important;
    transform: translateX(4px) !important;
    box-shadow: 0 2px 16px rgba(0,229,255,0.10), inset 0 0 0 1px rgba(0,229,255,0.05) !important;
}
/* Active nav button — Glassy active state */
[data-testid="stSidebar"] .nav-active .stButton > button,
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg,
        rgba(0,229,255,0.12) 0%,
        rgba(0,85,255,0.08) 100%) !important;
    border-color: rgba(0,229,255,0.40) !important;
    color: #00e5ff !important;
    font-weight: 700 !important;
    box-shadow: 0 0 20px rgba(0,229,255,0.12),
               0 4px 12px rgba(0,0,0,0.3),
               inset 0 1px 0 rgba(255,255,255,0.06) !important;
    transform: translateX(4px) !important;
    text-shadow: 0 0 12px rgba(0,229,255,0.35) !important;
}

/* -- Hero banner -- */
.hero-banner {
    background: linear-gradient(135deg, #060c1e 0%, #0d1b3e 50%, #091428 100%);
    border: 1px solid rgba(0,229,255,0.2);
    border-radius: 16px;
    padding: 30px 38px 24px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute; top: -80px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(0,229,255,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute; bottom: -50px; left: 30px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(255,215,0,0.05) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.1rem;
    font-weight: 900;
    background: linear-gradient(90deg, #00e5ff 0%, #ffffff 50%, #ffd700 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 6px 0;
    letter-spacing: 2px;
    line-height: 1.2;
}
.hero-sub {
    color: #5a7fa8;
    font-size: 0.85rem;
    font-weight: 400;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,229,255,0.08);
    border: 1px solid rgba(0,229,255,0.3);
    color: #00e5ff;
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

/* -- Stat cards -- */
.stat-card {
    background: linear-gradient(135deg, #0d1b3e 0%, #0a1428 100%);
    border: 1px solid rgba(0,229,255,0.15);
    border-radius: 14px;
    padding: 20px 18px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s, transform 0.2s;
}
.stat-card:hover { border-color: rgba(0,229,255,0.45); transform: translateY(-3px); }
.stat-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #00e5ff, transparent);
}
.stat-card.gold::before   { background: linear-gradient(90deg, transparent, #ffd700, transparent); }
.stat-card.blue::before   { background: linear-gradient(90deg, transparent, #00aaff, transparent); }
.stat-card.green::before  { background: linear-gradient(90deg, transparent, #39ff14, transparent); }
.stat-card.pink::before   { background: linear-gradient(90deg, transparent, #ff6b9d, transparent); }
.stat-icon  { font-size: 1.6rem; margin-bottom: 6px; display: block; }
.stat-number {
    font-family: 'Orbitron', monospace;
    font-size: 2rem; font-weight: 900; color: #fff;
    display: block; line-height: 1;
}
.stat-card.gold  .stat-number { color: #ffd700; }
.stat-card.blue  .stat-number { color: #00aaff; }
.stat-card.green .stat-number { color: #39ff14; }
.stat-card.pink  .stat-number { color: #ff6b9d; }
.stat-label {
    font-size: 0.68rem; color: #3a5a82;
    text-transform: uppercase; letter-spacing: 2px;
    margin-top: 6px; display: block;
}

/* -- Section header -- */
.section-header {
    display: flex; align-items: center; gap: 10px;
    margin: 26px 0 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(0,229,255,0.1);
}
.section-header h2 {
    font-family: 'Orbitron', monospace; font-size: 0.95rem;
    font-weight: 700; color: #c5d8f5; margin: 0;
    letter-spacing: 2px; text-transform: uppercase;
}
.section-accent {
    width: 4px; height: 20px;
    background: linear-gradient(180deg, #00e5ff, #0055ff);
    border-radius: 2px;
}
.neon-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,229,255,0.25), transparent);
    margin: 22px 0;
}

/* -- Player rank cards -- */
.player-card {
    background: linear-gradient(135deg, #0c1a38 0%, #091428 100%);
    border: 1px solid rgba(0,229,255,0.12);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 8px;
    display: flex; align-items: center; gap: 14px;
    transition: border-color 0.2s, background 0.2s;
}
.player-card:hover {
    border-color: rgba(0,229,255,0.38);
    background: linear-gradient(135deg, #0f2050 0%, #0c1a38 100%);
}
.rank-chip {
    display: inline-flex; align-items: center; justify-content: center;
    min-width: 30px; height: 30px; border-radius: 50%;
    font-weight: 700; font-size: 0.72rem; flex-shrink: 0;
}
.rank-1  { background: linear-gradient(135deg,#ffd700,#ffaa00); color: #000; }
.rank-2  { background: linear-gradient(135deg,#c0c0c0,#909090); color: #000; }
.rank-3  { background: linear-gradient(135deg,#cd7f32,#a05a20); color: #000; }
.rank-n  { background: rgba(255,255,255,0.06); color: #4a6a8a; border:1px solid rgba(255,255,255,0.1); }
.player-name { font-weight: 600; font-size: 0.88rem; color: #e0e6f0; }
.player-role { font-size: 0.69rem; color: #3a5a82; text-transform: uppercase; letter-spacing: 1px; margin-top:2px; }
.im-score-pill {
    margin-left: auto; padding: 5px 13px; border-radius: 20px;
    font-weight: 800; font-size: 0.82rem; font-family: 'Orbitron', monospace; flex-shrink: 0;
}
.im-elite { background: rgba(57,255,20,0.13);  color: #39ff14; border: 1px solid rgba(57,255,20,0.28); }
.im-good  { background: rgba(255,215,0,0.13);  color: #ffd700; border: 1px solid rgba(255,215,0,0.28); }
.im-avg   { background: rgba(255,140,0,0.13);  color: #ff8c00; border: 1px solid rgba(255,140,0,0.28); }
.im-low   { background: rgba(255,59,59,0.13);  color: #ff3b3b; border: 1px solid rgba(255,59,59,0.28); }

/* -- Info/warning boxes -- */
.info-box {
    background: rgba(0,229,255,0.04);
    border: 1px solid rgba(0,229,255,0.18);
    border-left: 3px solid #00e5ff;
    border-radius: 8px;
    padding: 11px 16px; font-size: 0.83rem; color: #7a9cc4; margin-bottom: 14px;
}

/* -- Widget overrides -- */
div[data-baseweb="select"] > div,
.stMultiSelect > div > div { background: #0a1428 !important; border-color: rgba(0,229,255,0.2) !important; }
div[data-baseweb="select"] * { color: #e0e6f0 !important; }
.stTextInput input, .stNumberInput input {
    background: #0a1428 !important; border-color: rgba(0,229,255,0.2) !important; color: #e0e6f0 !important;
}
.stSlider > div > div > div { background: rgba(0,229,255,0.3) !important; }
.stMetric label { color: #3a5a82 !important; font-size: 0.7rem !important; text-transform: uppercase; letter-spacing: 1.5px; }
.stMetric [data-testid="metric-container"] > div:nth-child(2) {
    color: #ffffff !important; font-family: 'Orbitron', monospace !important; font-size: 1.4rem !important; font-weight: 700 !important;
}
.stMetric [data-testid="metric-container"] > div:nth-child(3) { font-size: 0.75rem !important; }

/* -- Buttons -- */
.stButton > button {
    background: linear-gradient(90deg, #00c8ff, #0055ff) !important;
    color: #fff !important; font-weight: 700 !important; border: none !important;
    border-radius: 8px !important; padding: 9px 26px !important;
    font-size: 0.82rem !important; letter-spacing: 1.5px !important;
    text-transform: uppercase !important; transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.82 !important; }
.stDownloadButton > button {
    background: transparent !important;
    border: 1px solid rgba(0,229,255,0.3) !important;
    color: #00e5ff !important; border-radius: 8px !important; font-size: 0.78rem !important;
}

/* -- Expander -- */
[data-testid="stExpander"] {
    background: #08112a !important; border: 1px solid rgba(0,229,255,0.12) !important; border-radius: 10px !important;
}
[data-testid="stExpander"] summary { color: #c5d8f5 !important; font-size: 0.85rem !important; }

/* -- Dataframe -- */
[data-testid="stDataFrameResizable"] { border: 1px solid rgba(0,229,255,0.12) !important; border-radius: 10px; }

/* -- Page title bar -- */
.page-title-bar {
    display: flex; align-items: flex-start; flex-direction: column;
    padding: 12px 20px 10px;
    background: linear-gradient(90deg, rgba(0,229,255,0.05), transparent);
    border-left: 3px solid #00e5ff;
    border-radius: 0 10px 10px 0;
    margin-bottom: 18px;
}
.page-title-bar h1 {
    font-family: 'Orbitron', monospace; font-size: 1.2rem; font-weight: 700;
    color: #e0e6f0; margin: 0; letter-spacing: 2px; text-transform: uppercase;
}
.page-title-bar .page-desc { color: #3a5a82; font-size: 0.75rem; margin-top: 3px; }

/* -- Sidebar brand — Glassmorphism -- */
.sidebar-brand {
    text-align: center;
    padding: 20px 16px 14px;
    margin: 0 8px 4px;
    background: linear-gradient(135deg, rgba(0,229,255,0.06) 0%, rgba(0,85,255,0.04) 100%);
    border: 1px solid rgba(0,229,255,0.12);
    border-radius: 16px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    position: relative;
    overflow: hidden;
}
.sidebar-brand::before {
    content: '';
    position: absolute; top: -30px; right: -30px;
    width: 80px; height: 80px;
    background: radial-gradient(circle, rgba(0,229,255,0.10) 0%, transparent 70%);
    pointer-events: none;
}
.sidebar-brand::after {
    content: '';
    position: absolute; bottom: -20px; left: -20px;
    width: 60px; height: 60px;
    background: radial-gradient(circle, rgba(255,215,0,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.brand-logo {
    width: 72px;
    height: 72px;
    border-radius: 50%;
    object-fit: cover;
    border: 2px solid rgba(0,229,255,0.35);
    box-shadow: 0 0 18px rgba(0,229,255,0.15), 0 4px 12px rgba(0,0,0,0.4);
    display: block;
    margin: 0 auto 8px;
}
.brand-title {
    font-family: 'Orbitron', monospace; font-size: 0.9rem; font-weight: 900;
    background: linear-gradient(90deg, #00e5ff 20%, #ffffff 50%, #ffd700 80%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    letter-spacing: 3px; display: block; margin-top: 6px;
}
.brand-sub {
    font-size: 0.58rem; color: #3a5a82; letter-spacing: 2.5px;
    text-transform: uppercase; display: block; margin-top: 4px;
}
.sidebar-section-label {
    font-size: 0.56rem; color: #3a5a82; text-transform: uppercase;
    letter-spacing: 3px; padding: 14px 0 6px; display: block;
    font-weight: 600;
}

/* -- Sidebar dividers — glass style -- */
.sidebar-glass-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,229,255,0.18), transparent);
    margin: 10px 12px;
    border: none;
}

/* -- Sidebar stats panel — glass cards -- */
.sidebar-stat-card {
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(0,229,255,0.10);
    border-radius: 12px;
    padding: 14px 16px;
    transition: border-color 0.3s, transform 0.2s;
}
.sidebar-stat-card:hover {
    border-color: rgba(0,229,255,0.30);
    transform: translateY(-1px);
}
.sidebar-stat-card.gold {
    background: rgba(255,215,0,0.03);
    border-color: rgba(255,215,0,0.12);
}
.sidebar-stat-card.gold:hover { border-color: rgba(255,215,0,0.35); }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# SIDEBAR · Premium Design
# ------------------------------------------------------------------------------
st.sidebar.markdown(f"""
<div class="sidebar-brand">
  <img class="brand-logo" src="{_LOGO_URI}" alt="IPL Impact" />
  <span class="brand-title">IPL IMPACT</span>
  <span class="brand-sub">Analytics Platform · 2026</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-glass-divider"></div>', unsafe_allow_html=True)
st.sidebar.markdown('<span class="sidebar-section-label">Navigation</span>', unsafe_allow_html=True)

_NAV_ICONS = {
    "Leaderboard":              '<span class="ic ic-lg ic-chart"></span>',
    "Player Deep-Dive":         '<span class="ic ic-lg ic-target"></span>',
    "Team Impact Simulator":    '<span class="ic ic-lg ic-shield"></span>',
    "Head-to-Head":             '<span class="ic ic-lg ic-versus"></span>',
    "Smart Predictor":          '<span class="ic ic-lg ic-brain"></span>',
    "Advanced Analytics":       '<span class="ic ic-lg ic-analytics"></span>',
}
_NAV_PAGES = [
    "Leaderboard",
    "Player Deep-Dive",
    "Team Impact Simulator",
    "Head-to-Head",
    "Smart Predictor",
    "Advanced Analytics",
]
if "page" not in st.session_state:
    st.session_state["page"] = _NAV_PAGES[0]

for _np in _NAV_PAGES:
    _is_active = (st.session_state["page"] == _np)
    _btn_label = (f"▶ {_np}" if _is_active else _np)
    _btn_type  = "primary" if _is_active else "secondary"
    if st.sidebar.button(_btn_label, key=f"nav_{_np}", type=_btn_type, width='stretch'):
        st.session_state["page"] = _np
        st.rerun()

page = st.session_state["page"]

# Default role filter (all roles selected, no UI widget)
role_filter = ["Batter", "Bowler", "All-Rounder"]

st.sidebar.markdown('<div class="sidebar-glass-divider"></div>', unsafe_allow_html=True)
st.sidebar.markdown('<p style="font-size:0.54rem;color:#2a3a52;text-align:center;letter-spacing:1.5px;text-transform:uppercase;opacity:0.7">IPL Impact Metric · Hackathon 2026</p>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# THEME CONSTANTS
# ------------------------------------------------------------------------------
_BG   = "#050b18"
_PLOT = "#08112a"
_GRID = "rgba(0,229,255,0.07)"
_FONT = "#c5d8f5"
_DARK = dict(paper_bgcolor=_BG, plot_bgcolor=_PLOT, font=dict(color=_FONT, family="Inter, sans-serif"))

def _im_col(v):
    if v >= 67: return "#39ff14"
    if v >= 50: return "#ffd700"
    if v >= 33: return "#ff8c00"
    return "#ff3b3b"

def _im_pill(v):
    if v >= 67: return "im-elite"
    if v >= 50: return "im-good"
    if v >= 33: return "im-avg"
    return "im-low"

# ------------------------------------------------------------------------------
# HELPER · GAUGE
# ------------------------------------------------------------------------------
def make_gauge(player_name):
    row = final_df[final_df["player"]==player_name]
    if row.empty:
        return go.Figure()
    im   = row.iloc[0]["impact_metric"]
    rank = row.iloc[0]["rank"]
    role = row.iloc[0]["role"]
    col  = _im_col(im)
    fig  = go.Figure(go.Indicator(
        mode  ="gauge+number+delta",
        value = im,
        delta ={"reference":50,"increasing":{"color":"#39ff14"},"decreasing":{"color":"#ff3b3b"},"font":{"size":13}},
        number={"suffix":"/100","font":{"size":36,"color":"#ffffff","family":"Orbitron, monospace"}},
        title ={"text":f"<span style='color:#c5d8f5;font-weight:600'>{player_name}</span><br>"
                       f"<span style='color:#3a5a82;font-size:12px'>{role} &nbsp;·&nbsp; Rank #{rank}</span>",
                "font":{"size":13}},
        gauge={
            "axis":{"range":[0,100],"tickwidth":1,"tickcolor":"rgba(0,229,255,0.2)",
                    "tickfont":{"color":"#2a3a52","size":9}},
            "bar":{"color":col,"thickness":0.22},
            "bgcolor":_PLOT,
            "bordercolor":"rgba(0,229,255,0.15)",
            "steps":[
                {"range":[0,33],  "color":"rgba(255,59,59,0.10)"},
                {"range":[33,50], "color":"rgba(255,140,0,0.08)"},
                {"range":[50,67], "color":"rgba(255,215,0,0.08)"},
                {"range":[67,100],"color":"rgba(57,255,20,0.08)"},
            ],
            "threshold":{"line":{"color":"#00e5ff","width":3},"thickness":0.72,"value":50},
        },
    ))
    fig.update_layout(height=300, margin=dict(l=20,r=20,t=60,b=10), **_DARK)
    return fig

# ------------------------------------------------------------------------------
# HELPER · TREND
# ------------------------------------------------------------------------------
def make_trend(player_name):
    sub = bat_im[bat_im["striker"]==player_name].sort_values(["date","innings"]).tail(10).copy()
    if sub.empty:
        return None
    mu_  = bat_im["innings_im_raw"].mean()
    sig_ = bat_im["innings_im_raw"].std()+1e-8
    z_   = (sub["innings_im_raw"] - mu_) / sig_
    sub["im100"] = np.clip((1/(1+np.exp(-z_*1.5)))*100, 0, 100)
    sub.reset_index(drop=True, inplace=True)
    sub["x"] = [f"Inn {i+1}" for i in range(len(sub))]
    cmap   = {"Low":"#39ff14","Medium":"#ffd700","High":"#ff8c00","Extreme":"#ff3b3b"}
    colors = sub["pressure_level"].map(cmap).fillna("#5a7fa8").tolist()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sub["x"], y=sub["im100"],
        marker=dict(color=colors, line=dict(color="rgba(0,229,255,0.2)",width=0.8)),
        name="IM per innings",
        hovertemplate="<b>%{x}</b><br>IM: %{y:.1f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=sub["x"], y=sub["im100"].rolling(3,min_periods=1).mean(),
        mode="lines+markers", name="3-Inn Avg",
        line=dict(color="#00e5ff",width=2.5,dash="dot"),
        marker=dict(size=8,color="#00e5ff",line=dict(color=_BG,width=2)),
    ))
    fig.add_hline(y=50, line_dash="dash", line_color="rgba(0,229,255,0.35)",
                  annotation_text="Baseline (50)", annotation_font_color="#2a3a52",
                  annotation_font_size=10)
    fig.update_layout(
        height=310,
        title=dict(text=f"<b>{player_name}</b> · Impact Trend (last {len(sub)} innings)",
                   font=dict(color=_FONT, size=12)),
        yaxis=dict(range=[0,110], gridcolor=_GRID,
                   tickfont=dict(color="#2a3a52",size=9),
                   title=dict(text="Impact Metric",font=dict(color="#2a3a52",size=9))),
        xaxis=dict(gridcolor=_GRID, tickfont=dict(color="#2a3a52",size=9),
                   title=dict(text="Innings (oldest → newest)",font=dict(color="#2a3a52",size=9))),
        legend=dict(orientation="h", font=dict(color=_FONT,size=10), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10,r=10,t=44,b=10),
        **_DARK,
    )
    return fig


# ------------------------------------------------------------------------------
# PAGE: LEADERBOARD
# ------------------------------------------------------------------------------
if page == "Leaderboard":
    # Hero
    st.markdown("""
    <div class="hero-banner">
      <div class="hero-title"><span class="ic ic-xl ic-trophy"></span> IPL IMPACT LEADERBOARD</div>
      <div class="hero-sub">Season Performance Rankings · Multi-dimensional Impact Analysis</div>
      <span class="hero-badge"><span class="ic ic-bolt"></span> Live Analytics</span>
    </div>
    """, unsafe_allow_html=True)

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="stat-card"><span class="stat-icon"><span class="ic ic-xl ic-player"></span></span><span class="stat-number">{len(final_df)}</span><span class="stat-label">Total Players</span></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card blue"><span class="stat-icon"><span class="ic ic-xl ic-bat"></span></span><span class="stat-number">{int((final_df["role"]=="Batter").sum())}</span><span class="stat-label">Batters</span></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-card green"><span class="stat-icon"><span class="ic ic-xl ic-ball"></span></span><span class="stat-number">{int((final_df["role"]=="Bowler").sum())}</span><span class="stat-label">Bowlers</span></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="stat-card gold"><span class="stat-icon"><span class="ic ic-xl ic-star"></span></span><span class="stat-number">{int((final_df["role"]=="All-Rounder").sum())}</span><span class="stat-label">All-Rounders</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    # -- Innings-Based Leaderboard -----------------------------------------
    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
    st.markdown("""<div class="section-header"><div class="section-accent"></div><h2><span class="ic ic-analytics"></span> Innings-Based Leaderboard</h2></div>""", unsafe_allow_html=True)

    inn_c1, inn_c2, inn_c3 = st.columns([2, 2, 1])
    with inn_c1:
        inn_view = st.radio("View", ["Batting Innings", "Bowling Innings", "Total Innings"],
                            horizontal=True, key="inn_lb_view")
    with inn_c2:
        min_inn = st.slider("Minimum innings played", 1, 50, 5, key="inn_lb_min")
    with inn_c3:
        inn_top_n = st.slider("Show top", 5, 50, 15, key="inn_lb_top")

    # Prepare innings leaderboard data
    inn_df = final_df[final_df["role"].isin(role_filter)].copy()
    inn_df["total_innings"] = inn_df["bat_innings"] + inn_df["bowl_innings"]

    if inn_view == "Batting Innings":
        inn_df = inn_df[inn_df["bat_innings"] >= min_inn]
        inn_df = inn_df.sort_values("bat_im", ascending=False).head(inn_top_n)
        _x_col, _x_label = "bat_im", "Batting Impact Metric"
        _inn_col = "bat_innings"
        _colors = inn_df["bat_im"].apply(lambda v: "#00aaff" if v >= 60 else ("#1de9b6" if v >= 40 else "#3a5a82"))
    elif inn_view == "Bowling Innings":
        inn_df = inn_df[inn_df["bowl_innings"] >= min_inn]
        inn_df = inn_df.sort_values("bowl_im", ascending=False).head(inn_top_n)
        _x_col, _x_label = "bowl_im", "Bowling Impact Metric"
        _inn_col = "bowl_innings"
        _colors = inn_df["bowl_im"].apply(lambda v: "#39ff14" if v >= 60 else ("#ffd93d" if v >= 40 else "#3a5a82"))
    else:
        inn_df = inn_df[inn_df["total_innings"] >= min_inn]
        inn_df = inn_df.sort_values("impact_metric", ascending=False).head(inn_top_n)
        _x_col, _x_label = "impact_metric", "Overall Impact Metric"
        _inn_col = "total_innings"
        _colors = inn_df["impact_metric"].apply(_im_col)

    if len(inn_df) == 0:
        st.info("No players match the minimum innings filter. Try lowering the threshold.")
    else:
        fig_inn = go.Figure()
        fig_inn.add_trace(go.Bar(
            y=inn_df["player"][::-1],
            x=inn_df[_x_col][::-1],
            orientation="h",
            marker=dict(color=_colors[::-1].tolist(),
                        line=dict(color="rgba(0,229,255,0.12)", width=0.5)),
            text=[f"{int(v)} inn" for v in inn_df[_inn_col][::-1]],
            textfont=dict(color="#ffffff", size=9, family="Inter, sans-serif"),
            textposition="inside",
            customdata=inn_df[["role", "bat_innings", "bowl_innings", "bat_im", "bowl_im"]][::-1].values,
            hovertemplate=(
                "<b>%{y}</b><br>"
                f"{_x_label}: <b>%{{x:.1f}}</b><br>"
                "Role: %{customdata[0]}<br>"
                "Bat Inn: %{customdata[1]}  Bowl Inn: %{customdata[2]}<br>"
                "Bat IM: %{customdata[3]:.1f}  Bowl IM: %{customdata[4]:.1f}<extra></extra>"
            ),
        ))
        fig_inn.add_vline(x=50, line_dash="dot", line_color="rgba(0,229,255,0.4)",
                          annotation_text="League Avg",
                          annotation_font=dict(color="#2a3a52", size=9))
        fig_inn.update_layout(
            height=max(380, inn_top_n * 28),
            xaxis=dict(range=[0, 118], gridcolor=_GRID,
                       tickfont=dict(color="#2a3a52", size=9),
                       title=dict(text=_x_label, font=dict(color="#2a3a52", size=10))),
            yaxis=dict(tickfont=dict(color=_FONT, size=10)),
            margin=dict(l=180, r=50, t=14, b=24),
            **_DARK,
        )
        st.plotly_chart(fig_inn, width='stretch')

        # Summary table
        with st.expander("Innings Leaderboard Table", expanded=False):
            _inn_disp = inn_df[["player", "role", "bat_innings", "bowl_innings",
                                "bat_im", "bowl_im", "impact_metric"]].copy()
            _inn_disp["total_innings"] = _inn_disp["bat_innings"] + _inn_disp["bowl_innings"]
            _inn_disp = _inn_disp.rename(columns={
                "bat_innings": "Bat Inn", "bowl_innings": "Bowl Inn",
                "total_innings": "Total Inn", "bat_im": "Bat IM",
                "bowl_im": "Bowl IM", "impact_metric": "Overall IM",
            })
            st.dataframe(_inn_disp, width='stretch', height=350)

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
    st.download_button("Download Full Leaderboard CSV",
                       final_df.to_csv(index=False), "ipl_impact_leaderboard.csv", "text/csv")


# ------------------------------------------------------------------------------
# PAGE: PLAYER DEEP-DIVE
# ------------------------------------------------------------------------------
elif page == "Player Deep-Dive":
    st.markdown("""
    <div class="page-title-bar">
      <h1><span class="ic ic-lg ic-target"></span> Player Deep-Dive</h1>
      <div class="page-desc">Comprehensive individual player analytics · Impact Metric breakdown</div>
    </div>
    """, unsafe_allow_html=True)

    all_players = sorted(final_df["player"].tolist())
    player = st.selectbox("Select Player", all_players, index=0)

    # Gauge + trend
    col_g, col_t = st.columns([1, 2])
    with col_g:
        st.plotly_chart(make_gauge(player), width='stretch')
    with col_t:
        trend = make_trend(player)
        if trend:
            st.plotly_chart(trend, width='stretch')
        else:
            st.markdown('<div class="info-box">No batting innings data available for this player.</div>', unsafe_allow_html=True)

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    # Batting career stats
    bat_stats = bat_agg[bat_agg["striker"]==player]
    if not bat_stats.empty:
        st.markdown("""<div class="section-header"><div class="section-accent"></div><h2>Batting Career Stats</h2></div>""", unsafe_allow_html=True)
        sc1,sc2,sc3,sc4,sc5,sc6 = st.columns(6)
        sc1.metric("Innings",      len(bat_stats))
        sc2.metric("Total Runs",   int(bat_stats["runs_scored"].sum()))
        sc3.metric("Avg Runs",     f"{bat_stats['runs_scored'].mean():.1f}")
        sc4.metric("Avg SR",       f"{bat_stats['strike_rate'].mean():.1f}")
        sc5.metric("Total 4s",     int(bat_stats["fours"].sum()))
        sc6.metric("Total 6s",     int(bat_stats["sixes"].sum()))

        # Batting vs Bowling contribution chart
        fim = final_df[final_df["player"]==player].iloc[0] if not final_df[final_df["player"]==player].empty else None
        if fim is not None:
            bowl_stats2 = bowl_agg[bowl_agg["bowler"]==player]
            bat_contrib = fim["bat_im"] or 0
            bowl_contrib = fim["bowl_im"] or 0
            if bat_contrib > 0 or bowl_contrib > 0:
                st.markdown("""<div class="section-header"><div class="section-accent"></div><h2>Bat vs Bowl Contribution</h2></div>""", unsafe_allow_html=True)
                fig_cont = go.Figure()
                fig_cont.add_trace(go.Bar(
                    x=["Batting IM","Bowling IM"],
                    y=[bat_contrib, bowl_contrib],
                    marker=dict(
                        color=["#00aaff","#39ff14"],
                        line=dict(color="rgba(0,229,255,0.2)",width=1),
                    ),
                    text=[f"{bat_contrib:.1f}",f"{bowl_contrib:.1f}"],
                    textposition="outside",
                    textfont=dict(color="#ffffff",family="Orbitron, monospace",size=13),
                    hovertemplate="%{x}: <b>%{y:.1f}</b><extra></extra>",
                    width=0.4,
                ))
                fig_cont.add_hline(y=50, line_dash="dot", line_color="rgba(0,229,255,0.35)",
                                  annotation_text="League Avg", annotation_font=dict(color="#2a3a52",size=9))
                fig_cont.update_layout(
                    height=280,
                    yaxis=dict(range=[0,110], gridcolor=_GRID, tickfont=dict(color="#2a3a52",size=9)),
                    xaxis=dict(tickfont=dict(color=_FONT,size=11)),
                    margin=dict(l=20,r=20,t=14,b=24),
                    **_DARK,
                )
                st.plotly_chart(fig_cont, width='stretch')

        # Pressure distribution pie
        st.markdown("""<div class="section-header"><div class="section-accent"></div><h2>Pressure Distribution (Batting)</h2></div>""", unsafe_allow_html=True)
        pv = bat_stats["pressure_level"].value_counts().reset_index()
        pv.columns = ["pressure_level","count"]
        fig_p = go.Figure(go.Pie(
            labels=pv["pressure_level"], values=pv["count"],
            hole=0.52,
            marker=dict(colors=["#39ff14","#ffd700","#ff8c00","#ff3b3b"],
                        line=dict(color=_BG, width=2)),
            hovertemplate="<b>%{label}</b><br>%{value} innings (%{percent})<extra></extra>",
        ))
        fig_p.update_layout(
            height=300,
            legend=dict(font=dict(color=_FONT,size=10),bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10,r=10,t=14,b=10),
            annotations=[dict(text=f"<b style='color:{_FONT}'>{len(bat_stats)}</b><br><span style='color:#2a3a52;font-size:9px'>INNINGS</span>",
                              showarrow=False, font=dict(size=14))],
            **_DARK,
        )
        st.plotly_chart(fig_p, width='stretch')

    # Bowling career stats
    bowl_stats = bowl_agg[bowl_agg["bowler"]==player]
    if not bowl_stats.empty:
        st.markdown("""<div class="section-header"><div class="section-accent"></div><h2>Bowling Career Stats</h2></div>""", unsafe_allow_html=True)
        bc1,bc2,bc3,bc4 = st.columns(4)
        bc1.metric("Total Wickets",   int(bowl_stats["wickets"].sum()))
        bc2.metric("Total Overs",     f"{bowl_stats['overs_bowled'].sum():.1f}")
        bc3.metric("Avg Economy",     f"{bowl_stats['economy_rate'].mean():.2f}")
        bc4.metric("Bowling Innings", len(bowl_stats))

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    # -- CONSISTENCY SCORE --------------------------------------------------
    st.markdown("""<div class="section-header"><div class="section-accent"></div>
<h2>Player Consistency Score</h2></div>""", unsafe_allow_html=True)
    _prow = final_df[final_df["player"] == player]
    _cons = float(_prow["consistency_score"].values[0]) if not _prow.empty else 0.5
    _cons_label = ("Very Consistent" if _cons >= 0.75 else
                   "Consistent"      if _cons >= 0.50 else
                   "Variable"        if _cons >= 0.25 else "Inconsistent")
    _cons_col   = ("#39ff14" if _cons >= 0.75 else
                   "#ffd700" if _cons >= 0.50 else
                   "#ff9f1c" if _cons >= 0.25 else "#ff3b3b")
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Consistency Score",  f"{_cons:.3f}")
    cc2.metric("Consistency Rating", _cons_label)
    cc3.metric("Consistency %",      f"{_cons*100:.1f}%")

    # Last-10 batting innings IM bar chart
    _bat_last10 = (bat_im[bat_im["striker"] == player]
                   .sort_values(["date","innings"])
                   [["match_id","innings","innings_im_raw","pressure_level"]]
                   .tail(10).reset_index(drop=True))
    if not _bat_last10.empty:
        _pr_c = {"Low":"#2a3a52","Medium":"#ffd700","High":"#ff9f1c","Extreme":"#ff3b3b"}
        fig_cons = go.Figure()
        fig_cons.add_trace(go.Bar(
            x=list(range(1, len(_bat_last10)+1)),
            y=_bat_last10["innings_im_raw"].round(3),
            marker=dict(
                color=[_pr_c.get(p,"#2a3a52") for p in _bat_last10["pressure_level"]],
                line=dict(color="rgba(0,229,255,0.2)",width=1),
            ),
            hovertemplate="Innings %{x}: <b>%{y:.3f} IM</b><extra></extra>",
            name="Per-Innings IM",
        ))
        _avg_v = float(_bat_last10["innings_im_raw"].mean())
        _std_v = float(_bat_last10["innings_im_raw"].std()) if len(_bat_last10) > 1 else 0.0
        fig_cons.add_hline(y=_avg_v, line_dash="dash", line_color=_cons_col,
                           annotation_text=f"Avg {_avg_v:.3f}",
                           annotation_font=dict(color=_cons_col, size=9))
        fig_cons.update_layout(
            height=260,
            xaxis=dict(title=dict(text="Last 10 Innings ?", font=dict(color="#2a3a52")),
                       tickfont=dict(color=_FONT, size=10)),
            yaxis=dict(gridcolor=_GRID, tickfont=dict(color="#2a3a52", size=9),
                       title=dict(text="Raw Innings IM", font=dict(color="#2a3a52"))),
            margin=dict(l=30,r=30,t=14,b=30), **_DARK,
        )
        st.plotly_chart(fig_cons, width='stretch')
        st.markdown(
            '<div style="display:flex;gap:18px;font-size:11px;margin-bottom:6px">'
            '<span style="color:#2a3a52">&#9632; Low &nbsp;</span>'
            '<span style="color:#ffd700">&#9632; Medium &nbsp;</span>'
            '<span style="color:#ff9f1c">&#9632; High &nbsp;</span>'
            '<span style="color:#ff3b3b">&#9632; Extreme &nbsp;</span>'
            '<span style="color:#c5d8f5;margin-left:8px"><i>(bar colour = pressure level)</i></span>'
            '</div>',
            unsafe_allow_html=True,
        )
        if _std_v > 0:
            st.caption(f"Std deviation over last 10 innings: **{_std_v:.4f}**  ·  "
                       f"Consistency = 1/s (normalised)")

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    # -- TEAM CONTRIBUTION % ------------------------------------------------
    st.markdown("""<div class="section-header"><div class="section-accent"></div>
<h2>Team Impact Contribution</h2></div>""", unsafe_allow_html=True)
    _player_im  = float(_prow["impact_metric"].values[0]) if not _prow.empty else 0.0
    _total_im   = float(final_df["impact_metric"].sum())
    _team_pct   = round(_player_im / max(_total_im, 1) * 100, 3)
    _rank_n     = int(_prow["rank"].values[0]) if not _prow.empty else 999
    _tot_pl     = len(final_df)

    tc1, tc2, tc3 = st.columns(3)
    tc1.metric("Player IM",            f"{_player_im:.1f}")
    tc2.metric("League Total IM",      f"{_total_im:.0f}")
    tc3.metric("Team Contribution %",  f"{_team_pct:.3f}%")

    fig_tc = go.Figure()
    fig_tc.add_trace(go.Bar(
        x=[_team_pct, 100 - _team_pct],
        y=["Contribution"],
        orientation="h",
        marker=dict(color=["#00e5ff","rgba(42,58,82,0.4)"]),
        hovertemplate=[f"{player}: {_team_pct:.3f}%<extra></extra>",
                       "Rest of league<extra></extra>"],
        width=0.4,
    ))
    fig_tc.update_layout(
        barmode="stack", height=110,
        xaxis=dict(range=[0,100], ticksuffix="%",
                   tickfont=dict(color=_FONT,size=9), gridcolor=_GRID),
        yaxis=dict(tickfont=dict(color=_FONT,size=11)),
        margin=dict(l=20,r=20,t=10,b=10), **_DARK,
    )
    st.plotly_chart(fig_tc, width='stretch')
    st.caption(f"**#{_rank_n}** out of {_tot_pl} players · "
               f"Contribution = Player IM · S All Players IM")

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    # -- MATCH IMPACT TIMELINE ----------------------------------------------
    st.markdown("""<div class="section-header"><div class="section-accent"></div>
<h2>Match Impact Timeline</h2></div>""", unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">How a player\'s batting impact (Win Probability Added) '
        'evolved <b>over-by-over</b> in a specific match. '
        '<b style="color:#39ff14">Green bars</b> = positive WPA. '
        '<b style="color:#ff3b3b">Red bars</b> = negative WPA. '
        '<b style="color:#ffd700"><span class="ic ic-six"></span></b> = boundary-heavy over. '
        '<b style="color:#ff3b3b"><span class="ic ic-out"></span></b> = wicket fell.</div>',
        unsafe_allow_html=True,
    )
    _p_matches = sorted(timeline_df[timeline_df["player"]==player]["match_id"].unique().tolist())
    if _p_matches:
        def _match_label(m):
            _r = matches_df[matches_df["match_id"]==m]
            if not _r.empty:
                _d = str(_r["date"].values[0])[:10]
                _t1 = _r["team1"].values[0] if "team1" in _r.columns else ""
                _t2 = _r["team2"].values[0] if "team2" in _r.columns else ""
                return f"Match {m}  ({_d})  {_t1} vs {_t2}"
            return f"Match {m}"
        _sel_mid = st.selectbox("Select match", _p_matches,
                                index=len(_p_matches)-1,
                                format_func=_match_label,
                                key="timeline_match_sel")
        _tdf = (timeline_df[(timeline_df["player"]==player) &
                             (timeline_df["match_id"]==_sel_mid)]
                .sort_values(["innings","over"]))
        if not _tdf.empty:
            for _inn in sorted(_tdf["innings"].unique()):
                _inn_data = _tdf[_tdf["innings"]==_inn].reset_index(drop=True)
                _bar_colors = ["#39ff14" if v >= 0 else "#ff3b3b"
                               for v in _inn_data["delta_wp"]]
                fig_tl = go.Figure()
                fig_tl.add_trace(go.Bar(
                    x=_inn_data["over"],
                    y=_inn_data["delta_wp"].round(4),
                    name="WPA per over",
                    marker=dict(color=_bar_colors,
                                line=dict(color="rgba(0,229,255,0.15)", width=0.8)),
                    customdata=_inn_data[["runs"]].values,
                    hovertemplate="Over %{x}: WPA <b>%{y:.4f}</b>  |  %{customdata[0]} runs<extra></extra>",
                ))
                fig_tl.add_trace(go.Scatter(
                    x=_inn_data["over"],
                    y=_inn_data["cum_wp"].round(4),
                    name="Cumulative WPA",
                    mode="lines+markers",
                    line=dict(color="#00e5ff", width=2),
                    marker=dict(size=5, color="#00e5ff"),
                    yaxis="y2",
                    hovertemplate="Over %{x}: Cum WPA <b>%{y:.4f}</b><extra></extra>",
                ))
                _bnd = _inn_data[_inn_data["boundaries"] > 0]
                if not _bnd.empty:
                    fig_tl.add_trace(go.Scatter(
                        x=_bnd["over"], y=_bnd["delta_wp"],
                        mode="markers",
                        marker=dict(symbol="star", size=13, color="#ffd700",
                                    line=dict(color="#ffd700", width=1)),
                        name="Boundary over", yaxis="y",
                        customdata=_bnd["boundaries"].values,
                        hovertemplate="Over %{x}: <b>%{customdata} boundary ball(s)</b><extra></extra>",
                    ))
                _wkts = _inn_data[_inn_data["wickets"] > 0]
                if not _wkts.empty:
                    fig_tl.add_trace(go.Scatter(
                        x=_wkts["over"], y=_wkts["delta_wp"],
                        mode="markers",
                        marker=dict(symbol="x", size=13, color="#ff3b3b",
                                    line=dict(color="#ff3b3b", width=2)),
                        name="Wicket", yaxis="y",
                        hovertemplate="Over %{x}: <b>WICKET</b><extra></extra>",
                    ))
                fig_tl.update_layout(
                    title=dict(text=f"Innings {_inn}",
                               font=dict(color=_FONT, size=12, family="Orbitron,monospace")),
                    height=320,
                    xaxis=dict(title=dict(text="Over", font=dict(color="#2a3a52")),
                               tickfont=dict(color=_FONT, size=9), gridcolor=_GRID, dtick=1),
                    yaxis=dict(title=dict(text="WPA per Over", font=dict(color="#2a3a52")),
                               gridcolor=_GRID, tickfont=dict(color="#2a3a52", size=9),
                               zeroline=True, zerolinecolor="rgba(0,229,255,0.25)"),
                    yaxis2=dict(title=dict(text="Cumulative WPA", font=dict(color="#00e5ff")),
                                overlaying="y", side="right",
                                tickfont=dict(color="#00e5ff", size=9)),
                    legend=dict(font=dict(color=_FONT, size=9), bgcolor="rgba(0,0,0,0)",
                                orientation="h", y=1.12),
                    margin=dict(l=40,r=55,t=48,b=30), **_DARK,
                )
                st.plotly_chart(fig_tl, width='stretch')
        else:
            st.markdown('<div class="info-box">No over-by-over data found for this match.</div>',
                        unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">No match timeline data available · '
                    'this player may only appear as a bowler (no batting WPA tracked).</div>',
                    unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# PAGE: TEAM IMPACT SIMULATOR
# ------------------------------------------------------------------------------
elif page == "Team Impact Simulator":
    st.markdown("""
    <div class="page-title-bar">
      <h1><span class="ic ic-lg ic-shield"></span> Team Impact Simulator</h1>
      <div class="page-desc">Build two custom XIs and simulate a head-to-head team battle based on Impact Metrics</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Select <b>11 players per team</b> (4 Batters + 3 All-Rounders + 4 Bowlers).
    The simulator compares aggregate Impact Metrics to predict the stronger squad.
    </div>
    """, unsafe_allow_html=True)

    # ---- Helper functions ---------------------------------------------------
    def _available(role, already_chosen):
        """Return players of a given role not yet selected."""
        pool = final_df[final_df["role"] == role].sort_values("impact_metric", ascending=False)
        return [""] + [p for p in pool["player"].tolist() if p not in already_chosen]

    def build_team_selector(label, key_prefix):
        """Render selectors for one team and return list of chosen players."""
        st.markdown(f'<div class="section-header"><div class="section-accent"></div>'
                    f'<h2><span class="ic ic-shield"></span> {label}</h2></div>',
                    unsafe_allow_html=True)
        chosen = []

        st.markdown("**Batters (4)**")
        bat_cols = st.columns(4)
        for i in range(4):
            opts = _available("Batter", chosen)
            sel = bat_cols[i].selectbox(f"Batter {i+1}", opts, key=f"{key_prefix}_bat_{i}")
            if sel:
                chosen.append(sel)

        st.markdown("**All-Rounders (3)**")
        ar_cols = st.columns(3)
        for i in range(3):
            opts = _available("All-Rounder", chosen)
            sel = ar_cols[i].selectbox(f"All-Rounder {i+1}", opts, key=f"{key_prefix}_ar_{i}")
            if sel:
                chosen.append(sel)

        st.markdown("**Bowlers (4)**")
        bowl_cols = st.columns(4)
        for i in range(4):
            opts = _available("Bowler", chosen)
            sel = bowl_cols[i].selectbox(f"Bowler {i+1}", opts, key=f"{key_prefix}_bwl_{i}")
            if sel:
                chosen.append(sel)

        return chosen

    def compute_team_strength(players):
        """Compute batting, bowling, and total strength for a list of players."""
        df = final_df[final_df["player"].isin(players)].copy()
        bat_strength  = 0.0
        bowl_strength = 0.0
        total_im      = 0.0
        records       = []
        for _, r in df.iterrows():
            role = r["role"]
            if role == "Batter":
                contrib = float(r["bat_im"])
                bat_strength += contrib
            elif role == "Bowler":
                contrib = float(r["bowl_im"])
                bowl_strength += contrib
            else:  # All-Rounder
                b_im = float(r["bat_im"])
                w_im = float(r["bowl_im"])
                contrib = (b_im + w_im) / 2
                bat_strength  += b_im * 0.5
                bowl_strength += w_im * 0.5
            total_im += contrib
            records.append({"player": r["player"], "role": role, "contrib": round(contrib, 1),
                            "bat_im": round(float(r["bat_im"]), 1),
                            "bowl_im": round(float(r["bowl_im"]), 1),
                            "overall_im": round(float(r["impact_metric"]), 1)})
        avg_im = total_im / max(len(players), 1)
        return {
            "total": round(total_im, 1),
            "avg": round(avg_im, 1),
            "bat_strength": round(bat_strength, 1),
            "bowl_strength": round(bowl_strength, 1),
            "players": records,
        }

    def predict_match_winner(str_a, str_b):
        diff = str_a["total"] - str_b["total"]
        if abs(diff) < 5:
            return "Too close to call — evenly matched!", "#ffd93d"
        elif diff > 0:
            return "Team A is predicted to win!", "#39ff14"
        else:
            return "Team B is predicted to win!", "#00aaff"

    def predict_top_performer(str_a, str_b):
        all_players = str_a["players"] + str_b["players"]
        if not all_players:
            return None
        return max(all_players, key=lambda p: p["contrib"])

    # ---- Team builders side by side -----------------------------------------
    col_a, col_b = st.columns(2)
    with col_a:
        team_a = build_team_selector("Team A", "ta")
    with col_b:
        team_b = build_team_selector("Team B", "tb")

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    # Validation
    _overlap = set(team_a) & set(team_b)
    if _overlap:
        st.warning(f"Duplicate players across teams: {', '.join(_overlap)}. Please fix before simulating.")

    if st.button("Simulate Match", type="primary"):
        if len(team_a) < 11 or len(team_b) < 11:
            st.error("Each team must have exactly 11 players selected (4 Batters + 3 All-Rounders + 4 Bowlers).")
        elif _overlap:
            st.error("Remove duplicate players before simulating.")
        else:
            str_a = compute_team_strength(team_a)
            str_b = compute_team_strength(team_b)
            verdict, v_color = predict_match_winner(str_a, str_b)
            mvp = predict_top_performer(str_a, str_b)

            # -- Verdict banner ------------------------------------------------
            st.markdown(f"""
            <div style="text-align:center; padding:22px 0 12px 0;">
              <span style="font-family:Orbitron,monospace; font-size:1.4rem; color:{v_color};
                           text-shadow:0 0 18px {v_color}55;">{verdict}</span>
            </div>
            """, unsafe_allow_html=True)

            # -- KPI row -------------------------------------------------------
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Team A Total IM", str_a["total"])
            k2.metric("Team B Total IM", str_b["total"])
            k3.metric("Team A Avg IM", str_a["avg"])
            k4.metric("Team B Avg IM", str_b["avg"])

            st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

            # -- Team strength comparison bar chart ----------------------------
            st.markdown("""<div class="section-header"><div class="section-accent"></div>
            <h2><span class="ic ic-analytics"></span> Team Strength Comparison</h2></div>""",
                        unsafe_allow_html=True)

            cats = ["Total IM", "Batting Strength", "Bowling Strength", "Avg IM"]
            vals_a = [str_a["total"], str_a["bat_strength"], str_a["bowl_strength"], str_a["avg"]]
            vals_b = [str_b["total"], str_b["bat_strength"], str_b["bowl_strength"], str_b["avg"]]

            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                name="Team A", x=cats, y=vals_a,
                marker_color="#00e5ff",
                text=[f"{v}" for v in vals_a], textposition="outside",
                textfont=dict(color="#00e5ff", size=11, family="Orbitron,monospace"),
            ))
            fig_comp.add_trace(go.Bar(
                name="Team B", x=cats, y=vals_b,
                marker_color="#ffd700",
                text=[f"{v}" for v in vals_b], textposition="outside",
                textfont=dict(color="#ffd700", size=11, family="Orbitron,monospace"),
            ))
            fig_comp.update_layout(
                barmode="group", height=400,
                xaxis=dict(tickfont=dict(color=_FONT, size=11)),
                yaxis=dict(gridcolor=_GRID, tickfont=dict(color="#2a3a52", size=9),
                           title=dict(text="Score", font=dict(color="#2a3a52", size=10))),
                legend=dict(font=dict(color=_FONT)),
                margin=dict(l=20, r=20, t=30, b=20),
                **_DARK,
            )
            st.plotly_chart(fig_comp, width='stretch')

            # -- Batting vs Bowling strength -----------------------------------
            st.markdown("""<div class="section-header"><div class="section-accent"></div>
            <h2><span class="ic ic-bat"></span> Batting vs <span class="ic ic-ball"></span> Bowling</h2></div>""",
                        unsafe_allow_html=True)

            fig_bvb = go.Figure()
            fig_bvb.add_trace(go.Bar(
                name="Team A Batting", x=["Batting", "Bowling"],
                y=[str_a["bat_strength"], str_a["bowl_strength"]],
                marker_color=["#00aaff", "#39ff14"],
                text=[str_a["bat_strength"], str_a["bowl_strength"]],
                textposition="outside",
                textfont=dict(color="#ffffff", size=10, family="Inter,sans-serif"),
            ))
            fig_bvb.add_trace(go.Bar(
                name="Team B Batting", x=["Batting", "Bowling"],
                y=[str_b["bat_strength"], str_b["bowl_strength"]],
                marker_color=["rgba(0,170,255,0.5)", "rgba(57,255,20,0.5)"],
                text=[str_b["bat_strength"], str_b["bowl_strength"]],
                textposition="outside",
                textfont=dict(color="#ffffff", size=10, family="Inter,sans-serif"),
            ))
            fig_bvb.update_layout(
                barmode="group", height=340,
                xaxis=dict(tickfont=dict(color=_FONT, size=11)),
                yaxis=dict(gridcolor=_GRID, tickfont=dict(color="#2a3a52", size=9)),
                legend=dict(font=dict(color=_FONT)),
                margin=dict(l=20, r=20, t=20, b=20),
                **_DARK,
            )
            st.plotly_chart(fig_bvb, width='stretch')

            # -- Top Performers Section ----------------------------------------
            st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
            st.markdown("""<div class="section-header"><div class="section-accent"></div>
            <h2><span class="ic ic-trophy"></span> Top Performers Prediction</h2></div>""",
                        unsafe_allow_html=True)

            all_players = str_a["players"] + str_b["players"]
            _ap_df = pd.DataFrame(all_players)

            # Best Batter (highest bat_im among both teams)
            _batters = _ap_df[_ap_df["role"].isin(["Batter", "All-Rounder"])].copy()
            best_bat = _batters.sort_values("bat_im", ascending=False).iloc[0] if len(_batters) > 0 else None

            # Best Bowler (highest bowl_im among both teams)
            _bowlers = _ap_df[_ap_df["role"].isin(["Bowler", "All-Rounder"])].copy()
            best_bowl = _bowlers.sort_values("bowl_im", ascending=False).iloc[0] if len(_bowlers) > 0 else None

            # Overall MVP
            mvp_row = _ap_df.sort_values("contrib", ascending=False).iloc[0] if len(_ap_df) > 0 else None

            # Identify which team each player belongs to
            _team_a_set = set(p["player"] for p in str_a["players"])

            tp1, tp2, tp3 = st.columns(3)

            with tp1:
                if best_bat is not None:
                    _bt = "Team A" if best_bat["player"] in _team_a_set else "Team B"
                    st.markdown(f"""
                    <div style="text-align:center; padding:16px 8px;
                                background:rgba(0,170,255,0.06); border-radius:14px;
                                border:1px solid rgba(0,170,255,0.15);">
                      <span style="font-family:Orbitron,monospace; font-size:0.65rem;
                                   color:#3a5a82; letter-spacing:2px;">BEST BATTER</span><br>
                      <span class="ic ic-lg ic-bat"></span><br>
                      <span style="font-family:Orbitron,monospace; font-size:1.15rem;
                                   color:#00aaff; text-shadow:0 0 12px rgba(0,170,255,0.3);">
                        {best_bat['player']}</span><br>
                      <span style="color:#c5d8f5; font-size:0.8rem;">
                        {best_bat['role']} · {_bt}<br>
                        Bat IM: <b style="color:#00aaff">{best_bat['bat_im']}</b>
                      </span>
                    </div>
                    """, unsafe_allow_html=True)

            with tp2:
                if mvp_row is not None:
                    _mt = "Team A" if mvp_row["player"] in _team_a_set else "Team B"
                    st.markdown(f"""
                    <div style="text-align:center; padding:16px 8px;
                                background:rgba(255,215,0,0.06); border-radius:14px;
                                border:1px solid rgba(255,215,0,0.18);">
                      <span style="font-family:Orbitron,monospace; font-size:0.65rem;
                                   color:#3a5a82; letter-spacing:2px;">OVERALL MVP</span><br>
                      <span class="ic ic-lg ic-trophy"></span><br>
                      <span style="font-family:Orbitron,monospace; font-size:1.15rem;
                                   color:#ffd700; text-shadow:0 0 12px rgba(255,215,0,0.3);">
                        {mvp_row['player']}</span><br>
                      <span style="color:#c5d8f5; font-size:0.8rem;">
                        {mvp_row['role']} · {_mt}<br>
                        Contribution: <b style="color:#ffd700">{mvp_row['contrib']}</b>
                      </span>
                    </div>
                    """, unsafe_allow_html=True)

            with tp3:
                if best_bowl is not None:
                    _bwt = "Team A" if best_bowl["player"] in _team_a_set else "Team B"
                    st.markdown(f"""
                    <div style="text-align:center; padding:16px 8px;
                                background:rgba(57,255,20,0.06); border-radius:14px;
                                border:1px solid rgba(57,255,20,0.15);">
                      <span style="font-family:Orbitron,monospace; font-size:0.65rem;
                                   color:#3a5a82; letter-spacing:2px;">BEST BOWLER</span><br>
                      <span class="ic ic-lg ic-ball"></span><br>
                      <span style="font-family:Orbitron,monospace; font-size:1.15rem;
                                   color:#39ff14; text-shadow:0 0 12px rgba(57,255,20,0.3);">
                        {best_bowl['player']}</span><br>
                      <span style="color:#c5d8f5; font-size:0.8rem;">
                        {best_bowl['role']} · {_bwt}<br>
                        Bowl IM: <b style="color:#39ff14">{best_bowl['bowl_im']}</b>
                      </span>
                    </div>
                    """, unsafe_allow_html=True)

            # -- Squad detail tables -------------------------------------------
            st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
            det_a, det_b = st.columns(2)
            with det_a:
                st.markdown("**Team A Squad**")
                if str_a["players"]:
                    _da = pd.DataFrame(str_a["players"])
                    _da.index = range(1, len(_da) + 1)
                    st.dataframe(_da.rename(columns={
                        "player": "Player", "role": "Role", "contrib": "Contribution",
                        "bat_im": "Bat IM", "bowl_im": "Bowl IM", "overall_im": "Overall IM",
                    }), width='stretch', height=420)
            with det_b:
                st.markdown("**Team B Squad**")
                if str_b["players"]:
                    _db = pd.DataFrame(str_b["players"])
                    _db.index = range(1, len(_db) + 1)
                    st.dataframe(_db.rename(columns={
                        "player": "Player", "role": "Role", "contrib": "Contribution",
                        "bat_im": "Bat IM", "bowl_im": "Bowl IM", "overall_im": "Overall IM",
                    }), width='stretch', height=420)


# ------------------------------------------------------------------------------
# PAGE: HEAD-TO-HEAD
# ------------------------------------------------------------------------------
elif page == "Head-to-Head":
    st.markdown("""
    <div class="page-title-bar">
      <h1><span class="ic ic-lg ic-versus"></span> Head-to-Head Comparison</h1>
      <div class="page-desc">Overall player analysis · Batter vs Bowler phase-wise matchup · Radar · Trends</div>
    </div>
    """, unsafe_allow_html=True)

    h2h_tab1, h2h_tab2 = st.tabs(["Overall Player Comparison", "Batter vs Bowler Matchup"])

    # ----------------------------------------------------------------------
    # TAB 1 · Overall comparison (existing feature)
    # ----------------------------------------------------------------------
    with h2h_tab1:
        all_p = sorted(final_df["player"].tolist())
        col1, col2 = st.columns(2)
        with col1:
            p1 = st.selectbox("Player 1", all_p, index=0, key="p1")
        with col2:
            p2 = st.selectbox("Player 2", all_p, index=min(1,len(all_p)-1), key="p2")

        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(make_gauge(p1), width='stretch')
        with g2:
            st.plotly_chart(make_gauge(p2), width='stretch')

        st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

        r1 = final_df[final_df["player"]==p1].iloc[0] if not final_df[final_df["player"]==p1].empty else None
        r2 = final_df[final_df["player"]==p2].iloc[0] if not final_df[final_df["player"]==p2].empty else None
        if r1 is not None and r2 is not None:
            st.markdown("""<div class="section-header"><div class="section-accent"></div><h2>Impact Metric Comparison</h2></div>""", unsafe_allow_html=True)
            fig_bar2 = go.Figure()
            for _p, _r, _c in [(p1, r1, "#00aaff"), (p2, r2, "#ff6b9d")]:
                fig_bar2.add_trace(go.Bar(
                    x=[_p], y=[_r["impact_metric"]], name=_p,
                    marker=dict(color=_c, line=dict(color="rgba(0,229,255,0.2)",width=1)),
                    text=[f"{_r['impact_metric']:.1f}"],
                    textposition="outside",
                    textfont=dict(color="#ffffff",family="Orbitron, monospace",size=14),
                    width=0.35,
                ))
            fig_bar2.add_hline(y=50, line_dash="dot", line_color="rgba(0,229,255,0.35)",
                               annotation_text="League Avg (50)",annotation_font=dict(color="#2a3a52",size=9))
            fig_bar2.update_layout(
                height=280, barmode="group",
                yaxis=dict(range=[0,115], gridcolor=_GRID, tickfont=dict(color="#2a3a52",size=9)),
                xaxis=dict(tickfont=dict(color=_FONT,size=11)),
                legend=dict(font=dict(color=_FONT,size=10),bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=20,r=20,t=14,b=14),
                **_DARK,
            )
            st.plotly_chart(fig_bar2, width='stretch')

        st.markdown("""<div class="section-header"><div class="section-accent"></div><h2>Attribute Radar</h2></div>""", unsafe_allow_html=True)

        def get_radar_vals(player):
            bat_r  = bat_agg[bat_agg["striker"]==player]
            bowl_r = bowl_agg[bowl_agg["bowler"]==player]
            fim    = final_df[final_df["player"]==player]
            return {
                "IM Score"       : fim["impact_metric"].values[0] if len(fim)>0 else 50,
                "Avg Runs"       : min(bat_r["runs_scored"].mean()/60*100,100) if not bat_r.empty else 0,
                "Strike Rate"    : min(bat_r["strike_rate"].mean()/150*100,100) if not bat_r.empty else 0,
                "Wickets/Match"  : min(bowl_r["wickets"].mean()/3*100,100) if not bowl_r.empty else 0,
                "Bowl Economy"   : max(0,(1-bowl_r["economy_rate"].mean()/12)*100) if not bowl_r.empty else 0,
                "Pressure Index" : (bat_r["pressure_index"].mean()*100) if not bat_r.empty else 0,
            }

        v1, v2 = get_radar_vals(p1), get_radar_vals(p2)
        cats   = list(v1.keys())
        fig_r  = go.Figure()
        for _label, _vals, _col, _fill in [
            (p1, list(v1.values()), "#00aaff", "rgba(0,170,255,0.12)"),
            (p2, list(v2.values()), "#ff6b9d", "rgba(255,107,157,0.12)"),
        ]:
            fig_r.add_trace(go.Scatterpolar(
                r=_vals+[_vals[0]], theta=cats+[cats[0]],
                fill="toself", fillcolor=_fill,
                name=_label,
                line=dict(color=_col, width=2.5),
                marker=dict(size=7, color=_col),
            ))
        fig_r.update_layout(
            polar=dict(
                bgcolor=_PLOT,
                radialaxis=dict(range=[0,100], gridcolor=_GRID, tickfont=dict(color="#2a3a52",size=8),
                                linecolor="rgba(0,229,255,0.1)"),
                angularaxis=dict(tickfont=dict(color=_FONT,size=10), gridcolor=_GRID,
                                 linecolor="rgba(0,229,255,0.1)"),
            ),
            legend=dict(font=dict(color=_FONT,size=11), bgcolor="rgba(0,0,0,0)"),
            height=420,
            margin=dict(l=40,r=40,t=14,b=14),
            **_DARK,
        )
        st.plotly_chart(fig_r, width='stretch')

        st.markdown("""<div class="section-header"><div class="section-accent"></div><h2>Impact Trends</h2></div>""", unsafe_allow_html=True)
        t1, t2 = st.columns(2)
        with t1:
            tr = make_trend(p1)
            if tr: st.plotly_chart(tr, width='stretch')
            else: st.markdown(f'<div class="info-box">No batting data for {p1}</div>', unsafe_allow_html=True)
        with t2:
            tr = make_trend(p2)
            if tr: st.plotly_chart(tr, width='stretch')
            else: st.markdown(f'<div class="info-box">No batting data for {p2}</div>', unsafe_allow_html=True)

    # ----------------------------------------------------------------------
    # TAB 2 · Batter vs Bowler phase-wise matchup
    # ----------------------------------------------------------------------
    with h2h_tab2:
        st.markdown(
            '<div class="info-box" style="border-left-color:#a855f7;">'
            'Select a <b style="color:#00e5ff">Batter</b> and a <b style="color:#ff6b9d">Bowler</b> '
            'to see their complete head-to-head battle broken down by match phase.</div>',
            unsafe_allow_html=True,
        )

        # -- Player selectors ---------------------------------------------
        _all_batters = sorted(balls_df["striker"].dropna().unique().tolist())
        _all_bowlers = sorted(balls_df["bowler"].dropna().unique().tolist())

        bc1, bc2 = st.columns(2)
        with bc1:
            st.markdown('<p style="color:#00e5ff;font-weight:700;margin-bottom:4px"><span class="ic ic-bat"></span> Batter</p>', unsafe_allow_html=True)
            sel_batter = st.selectbox("", _all_batters, key="h2h_batter", label_visibility="collapsed")
        with bc2:
            st.markdown('<p style="color:#ff6b9d;font-weight:700;margin-bottom:4px"><span class="ic ic-ball"></span> Bowler</p>', unsafe_allow_html=True)
            sel_bowler = st.selectbox("", _all_bowlers, key="h2h_bowler", label_visibility="collapsed")

        st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

        # -- Filter head-to-head deliveries -------------------------------
        h2h_balls = balls_df[
            (balls_df["striker"] == sel_batter) &
            (balls_df["bowler"]  == sel_bowler)
        ].copy()

        # Re-map match_stage to user-friendly phase labels in correct order
        _phase_map   = {"powerplay": "Powerplay (1–6)", "middle": "Middle (7–15)", "death": "Death (16–20)"}
        _phase_order = ["Powerplay (1–6)", "Middle (7–15)", "Death (16–20)"]
        h2h_balls["phase"] = h2h_balls["match_stage"].map(_phase_map).fillna("Middle (7–15)")

        total_dels = len(h2h_balls[h2h_balls["is_legal"]])

        if total_dels == 0:
            st.markdown(
                f'<div class="info-box" style="border-left-color:#ff6b9d;">'
                f'<span class="ic ic-out"></span> <b>{sel_batter}</b> and <b>{sel_bowler}</b> have never faced each other in the dataset.</div>',
                unsafe_allow_html=True,
            )
        else:
            # -- Compute phase-wise stats ----------------------------------
            def _phase_stats(df):
                legal   = df[df["is_legal"]]
                runs    = int(df["runs_off_bat"].sum())
                balls   = int(legal.shape[0])
                fours   = int((df["runs_off_bat"] == 4).sum())
                sixes   = int((df["runs_off_bat"] == 6).sum())
                wkts    = int(df["is_wicket"].sum())
                sr      = round(runs / balls * 100, 1) if balls > 0 else 0.0
                eco     = round(
                    df["runs_off_bat"].sum() / (balls / 6) if balls >= 6 else
                    df["runs_off_bat"].sum() / max(balls / 6, 0.1), 2
                )
                return dict(runs=runs, balls=balls, fours=fours, sixes=sixes,
                            wickets=wkts, strike_rate=sr, economy=eco)

            phase_rows = []
            for ph in _phase_order:
                sub = h2h_balls[h2h_balls["phase"] == ph]
                if sub.empty:
                    phase_rows.append(dict(Phase=ph, Runs=0, Balls=0,
                                           SR=0, Fours=0, Sixes=0, Wickets=0, Economy=0))
                else:
                    s = _phase_stats(sub)
                    phase_rows.append(dict(Phase=ph, Runs=s["runs"], Balls=s["balls"],
                                           SR=s["strike_rate"], Fours=s["fours"],
                                           Sixes=s["sixes"], Wickets=s["wickets"],
                                           Economy=s["economy"]))
            phase_df = pd.DataFrame(phase_rows)

            # -- Overall summary KPIs --------------------------------------
            _tot = _phase_stats(h2h_balls)
            st.markdown("""<div class="section-header"><div class="section-accent"></div>
<h2>Overall Head-to-Head Summary</h2></div>""", unsafe_allow_html=True)

            kc = st.columns(6)
            kc[0].metric("Total Runs",    str(_tot["runs"]))
            kc[1].metric("Balls Faced",   str(_tot["balls"]))
            kc[2].metric("Strike Rate",   f"{_tot['strike_rate']}")
            kc[3].metric("Boundaries",    f"{_tot['fours']}×4  {_tot['sixes']}×6")
            kc[4].metric("Dismissals",    str(_tot["wickets"]))
            kc[5].metric("Economy (bowl)",f"{_tot['economy']}")

            st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

            # -- Dominance logic per phase ---------------------------------
            # Batter dominates  → SR > 130 AND wickets == 0
            # Bowler dominates  → wickets > 0 OR SR < 80
            # Contested         → otherwise
            def _dominance(row):
                if row["Wickets"] > 0 or row["SR"] < 80:
                    return ("Bowler Dominates", "rgba(255,59,59,0.13)", "#ff3b3b")
                elif row["SR"] >= 130 and row["Wickets"] == 0:
                    return ("Batter Dominates", "rgba(57,255,20,0.10)", "#39ff14")
                else:
                    return ("Contested",         "rgba(255,215,0,0.08)",  "#ffd700")

            # -- Phase breakdown table -------------------------------------
            st.markdown("""<div class="section-header"><div class="section-accent"></div>
<h2>Phase-wise Breakdown</h2></div>""", unsafe_allow_html=True)

            for _, prow in phase_df.iterrows():
                dom_label, dom_bg, dom_col = _dominance(prow)
                has_data = prow["Balls"] > 0
                _opacity = "1" if has_data else "0.4"
                st.markdown(f"""
<div style="background:{dom_bg if has_data else 'rgba(255,255,255,0.02)'};
border:1px solid {dom_col if has_data else 'rgba(255,255,255,0.06)'};
border-radius:12px;padding:16px 22px;margin-bottom:12px;opacity:{_opacity}">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:12px">
    <span style="font-family:Orbitron,monospace;font-size:0.88rem;font-weight:700;
    color:{dom_col}">{prow['Phase']}</span>
    <span style="background:{dom_col}22;color:{dom_col};border:1px solid {dom_col}55;
    padding:2px 10px;border-radius:20px;font-size:11px;font-weight:700">
    {dom_label if has_data else "No Deliveries"}</span>
  </div>
  <div style="display:flex;gap:24px;flex-wrap:wrap">
    <div style="text-align:center">
      <div style="color:#00e5ff;font-family:Orbitron,monospace;font-size:1.4rem;font-weight:900">{prow['Runs']}</div>
      <div style="color:#3a5a82;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px">Runs</div>
    </div>
    <div style="text-align:center">
      <div style="color:#c5d8f5;font-family:Orbitron,monospace;font-size:1.4rem;font-weight:700">{prow['Balls']}</div>
      <div style="color:#3a5a82;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px">Balls</div>
    </div>
    <div style="text-align:center">
      <div style="color:#ffd700;font-family:Orbitron,monospace;font-size:1.4rem;font-weight:700">{prow['SR']}</div>
      <div style="color:#3a5a82;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px">Strike Rate</div>
    </div>
    <div style="text-align:center">
      <div style="color:#39ff14;font-family:Orbitron,monospace;font-size:1.4rem;font-weight:700">{prow['Fours']}?4 &nbsp; {prow['Sixes']}?6</div>
      <div style="color:#3a5a82;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px">Boundaries</div>
    </div>
    <div style="text-align:center">
      <div style="color:#ff6b9d;font-family:Orbitron,monospace;font-size:1.4rem;font-weight:700">{prow['Wickets']}</div>
      <div style="color:#3a5a82;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px">Dismissals</div>
    </div>
    <div style="text-align:center">
      <div style="color:#ff9f1c;font-family:Orbitron,monospace;font-size:1.4rem;font-weight:700">{prow['Economy']}</div>
      <div style="color:#3a5a82;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px">Economy</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

            st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

            # -- Chart 1: Runs per phase bar chart -------------------------
            st.markdown("""<div class="section-header"><div class="section-accent"></div>
<h2>Runs Scored Across Phases</h2></div>""", unsafe_allow_html=True)

            _bar_cols = []
            for _, pr in phase_df.iterrows():
                dl, _, dc = _dominance(pr)
                _bar_cols.append(dc if pr["Balls"] > 0 else "#2a3a52")

            fig_runs = go.Figure()
            fig_runs.add_trace(go.Bar(
                x=phase_df["Phase"], y=phase_df["Runs"],
                marker=dict(color=_bar_cols, line=dict(color="rgba(0,229,255,0.2)", width=1)),
                text=phase_df["Runs"],
                textposition="outside",
                textfont=dict(color="#ffffff", family="Orbitron,monospace", size=13),
                name="Runs",
                hovertemplate="<b>%{x}</b><br>Runs: %{y}<extra></extra>",
            ))
            fig_runs.update_layout(
                height=320,
                xaxis=dict(tickfont=dict(color=_FONT, size=11)),
                yaxis=dict(gridcolor=_GRID, tickfont=dict(color="#2a3a52", size=9),
                           title=dict(text="Runs Scored", font=dict(color="#2a3a52"))),
                margin=dict(l=30, r=20, t=20, b=20),
                showlegend=False,
                **_DARK,
            )
            st.plotly_chart(fig_runs, width='stretch')

            # -- Chart 2: Strike Rate vs Economy per phase -----------------
            st.markdown("""<div class="section-header"><div class="section-accent"></div>
<h2>Strike Rate vs Economy by Phase</h2></div>""", unsafe_allow_html=True)

            fig_sre = go.Figure()
            _phases_label = [p.split("(")[0].strip() for p in phase_df["Phase"]]
            fig_sre.add_trace(go.Bar(
                x=_phases_label, y=phase_df["SR"],
                name=f"{sel_batter} SR",
                marker_color="#00e5ff",
                text=phase_df["SR"].astype(str),
                textposition="outside",
                textfont=dict(color="#00e5ff", size=11),
                yaxis="y",
                offsetgroup=1,
            ))
            fig_sre.add_trace(go.Scatter(
                x=_phases_label,
                y=phase_df["Economy"].replace(0, None),
                name=f"{sel_bowler} Eco",
                mode="lines+markers",
                line=dict(color="#ff6b9d", width=3, dash="dot"),
                marker=dict(size=10, color="#ff6b9d", symbol="diamond"),
                yaxis="y2",
            ))
            fig_sre.update_layout(
                height=340,
                yaxis=dict(title=dict(text="Strike Rate", font=dict(color="#00e5ff")),
                           gridcolor=_GRID, tickfont=dict(color="#2a3a52", size=9)),
                yaxis2=dict(title=dict(text="Economy Rate", font=dict(color="#ff6b9d")),
                            overlaying="y", side="right",
                            tickfont=dict(color="#2a3a52", size=9)),
                xaxis=dict(tickfont=dict(color=_FONT, size=11)),
                legend=dict(font=dict(color=_FONT, size=10), bgcolor="rgba(0,0,0,0)"),
                barmode="group",
                margin=dict(l=40, r=50, t=20, b=20),
                **_DARK,
            )
            st.plotly_chart(fig_sre, width='stretch')

            # -- Chart 3: Radar · batter vs bowler across 6 axes ----------
            st.markdown("""<div class="section-header"><div class="section-accent"></div>
<h2>Performance Radar</h2></div>""", unsafe_allow_html=True)

            _total_balls = max(_tot["balls"], 1)
            _bat_radar_vals = [
                min(_tot["runs"] / 30 * 100, 100),                    # Runs ability
                min(_tot["strike_rate"] / 200 * 100, 100),            # SR
                min((_tot["fours"] + _tot["sixes"]) / max(_total_balls/6,1) * 100, 100),  # Boundary %
                max(0, (1 - _tot["wickets"] / max(_total_balls/6,1)) * 100),              # Survival
                min(_total_balls / 20 * 100, 100),                    # Exposure
                50,                                                    # placeholder centre
            ]
            _bowl_radar_vals = [
                min(_tot["wickets"] / max(_total_balls/6,1) * 100, 100),                  # Wicket threat
                max(0, (1 - _tot["strike_rate"] / 200) * 100),        # SR restriction
                min(_tot["wickets"] * 25, 100),                        # Dismissal impact
                max(0, (1 - _tot["economy"] / 12) * 100),             # Economy control
                min(_total_balls / 20 * 100, 100),                    # Exposure
                50,
            ]
            _radar_cats = ["Scoring Power","SR / Restriction","Boundary / Wicket Threat",
                           "Survival / Economy","Exposure (balls)","Centre"]

            fig_h2h_radar = go.Figure()
            for _lbl, _rv, _col, _fc in [
                (sel_batter, _bat_radar_vals, "#00e5ff", "rgba(0,229,255,0.10)"),
                (sel_bowler, _bowl_radar_vals, "#ff6b9d", "rgba(255,107,157,0.10)"),
            ]:
                fig_h2h_radar.add_trace(go.Scatterpolar(
                    r=_rv + [_rv[0]],
                    theta=_radar_cats + [_radar_cats[0]],
                    fill="toself", fillcolor=_fc,
                    name=_lbl,
                    line=dict(color=_col, width=2.5),
                    marker=dict(size=7, color=_col),
                ))
            fig_h2h_radar.update_layout(
                polar=dict(
                    bgcolor=_PLOT,
                    radialaxis=dict(range=[0,100], gridcolor=_GRID,
                                   tickfont=dict(color="#2a3a52", size=8),
                                   linecolor="rgba(0,229,255,0.1)"),
                    angularaxis=dict(tickfont=dict(color=_FONT, size=10), gridcolor=_GRID,
                                    linecolor="rgba(0,229,255,0.1)"),
                ),
                legend=dict(font=dict(color=_FONT, size=11), bgcolor="rgba(0,0,0,0)"),
                height=420,
                margin=dict(l=40, r=40, t=20, b=20),
                **_DARK,
            )
            st.plotly_chart(fig_h2h_radar, width='stretch')

            # -- Chart 4: Runs per over scatter ---------------------------
            st.markdown("""<div class="section-header"><div class="section-accent"></div>
<h2>Ball-by-Ball Run Flow</h2></div>""", unsafe_allow_html=True)

            _bbdf = h2h_balls[h2h_balls["is_legal"]].copy()
            _bbdf["_seq"] = range(len(_bbdf))
            _bbdf["_dot"] = _bbdf["runs_off_bat"] == 0
            _bbdf["_bnd"] = _bbdf["runs_off_bat"].isin([4, 6])
            _bbdf["_wkt"] = _bbdf["is_wicket"]
            _dot_c = _bbdf["runs_off_bat"].apply(
                lambda r: "#e63946" if r == 0 else ("#39ff14" if r >= 4 else "#ffd700")
            )

            fig_flow = go.Figure()
            fig_flow.add_trace(go.Bar(
                x=_bbdf["_seq"],
                y=_bbdf["runs_off_bat"],
                marker_color=_dot_c.tolist(),
                name="Runs",
                hovertemplate="Ball %{x}: <b>%{y} runs</b><extra></extra>",
            ))
            # Mark wickets
            _wkt_rows = _bbdf[_bbdf["is_wicket"]]
            if not _wkt_rows.empty:
                fig_flow.add_trace(go.Scatter(
                    x=_wkt_rows["_seq"], y=_wkt_rows["runs_off_bat"] + 0.3,
                    mode="markers+text",
                    text=["W"] * len(_wkt_rows),
                    textfont=dict(color="#ff6b9d", size=11, family="Orbitron,monospace"),
                    textposition="top center",
                    marker=dict(symbol="x", size=12, color="#ff6b9d"),
                    name="Wicket",
                    hovertemplate="Wicket on ball %{x}<extra></extra>",
                ))
            fig_flow.update_layout(
                height=300,
                xaxis=dict(title=dict(text="Ball sequence", font=dict(color="#2a3a52")),
                           tickfont=dict(color="#2a3a52", size=8)),
                yaxis=dict(title=dict(text="Runs off bat", font=dict(color="#2a3a52")),
                           gridcolor=_GRID, tickfont=dict(color="#2a3a52", size=9)),
                legend=dict(font=dict(color=_FONT, size=10), bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=30, r=20, t=20, b=30),
                showlegend=True,
                **_DARK,
            )
            # Legend annotation
            st.markdown("""<div style="display:flex;gap:18px;flex-wrap:wrap;margin-bottom:8px;font-size:12px">
<span><span style="color:#e63946">⚫</span> Dot ball</span>
<span><span style="color:#ffd700">●</span> 1–3 runs</span>
<span><span style="color:#39ff14">💥</span> Boundary (4/6)</span>
<span><span style="color:#ff6b9d">🔴</span> Wicket</span>
</div>""", unsafe_allow_html=True)
            st.plotly_chart(fig_flow, width='stretch')

            # -- Phase dominance summary strip -----------------------------
            st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
            st.markdown("""<div class="section-header"><div class="section-accent"></div>
<h2>Phase Dominance Summary</h2></div>""", unsafe_allow_html=True)

            dom_cols = st.columns(3)
            for _ci, (_, _pr) in enumerate(phase_df.iterrows()):
                dl, db, dc = _dominance(_pr)
                with dom_cols[_ci]:
                    st.markdown(f"""
<div style="background:{db};border:1px solid {dc}55;border-radius:10px;
padding:14px 16px;text-align:center">
  <div style="font-family:Orbitron,monospace;font-size:0.72rem;
  color:#3a5a82;margin-bottom:6px">{_pr['Phase']}</div>
  <div style="color:{dc};font-weight:700;font-size:0.9rem">{dl}</div>
  <div style="color:#c5d8f5;font-size:0.75rem;margin-top:6px">
  SR {_pr['SR']} · {_pr['Wickets']} wkt{'s' if _pr['Wickets']!=1 else ''}</div>
</div>""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# PAGE: SMART PREDICTOR
# ------------------------------------------------------------------------------
elif page == "Smart Predictor":
    st.markdown("""
    <div class="page-title-bar">
      <h1><span class="ic ic-lg ic-brain"></span> Smart Player Predictor</h1>
      <div class="page-desc">AI-powered player recommendations based on live match situation · Pressure Fitness Analysis</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Enter the <b style="color:#00e5ff">current match situation</b> '
        'and get an AI-ranked list of the best available players based on their Impact Metric, '
        'pressure-fitness, and role suitability.</div>',
        unsafe_allow_html=True,
    )

    sit_col, space_col = st.columns([2, 1])
    with sit_col:
        situation = st.radio(
            "Match Situation",
            ["Batting — Who should bat next?", "Bowling — Who should bowl next?"],
            horizontal=True,
        )

    st.markdown("""<div class="section-header"><div class="section-accent"></div><h2>Match State</h2></div>""", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    runs_left    = c1.number_input("Runs required / target left", min_value=0, max_value=300, value=60)
    # Valid cricket overs: 0.1–0.5, 1.0–1.5, ·, 19.0–19.5, 20.0
    # (.6 does not exist · after 5 balls the over completes → next whole number)
    _ov_opts = (
        [round(b / 10, 1) for b in range(1, 6)]                          # 0.1 · 0.5
        + [round(o + b / 10, 1) for o in range(1, 20) for b in range(6)] # 1.0–19.5
        + [20.0]
    )
    _ov_default = _ov_opts.index(6.0)
    overs_left   = c2.selectbox("Overs remaining", options=_ov_opts,
                                index=_ov_default,
                                format_func=lambda x: f"{x:.1f}")
    wickets_fallen = c3.number_input("Wickets fallen",            min_value=0, max_value=10, value=3)
    innings_no   = c4.selectbox("Innings", [1, 2], index=1)

    top_n_pred   = st.slider("How many players to recommend?", 3, 15, 5)
    min_innings  = st.slider("Minimum innings played", 1, 50, 5,
                             help="Only players with at least this many innings (batting or bowling based on situation) will be considered.")

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    if st.button("Find Best Players", type="primary"):

        # -- Compute live pressure index from inputs ---------------------------
        wickets_left_live = max(10 - wickets_fallen, 1)
        rrr_live          = runs_left / max(overs_left, 0.1)
        over_no           = max(20 - overs_left, 0)
        is_batting        = "Batting" in situation

        # Game stage (used in narrative later)
        if over_no <= 5:   stage_live = "powerplay"
        elif over_no <= 14: stage_live = "middle"
        else:               stage_live = "death"

        # ---- Pressure logic based on Batting vs Bowling perspective ----------
        # Batting perspective:
        #   RRR <= 7         → Low   (comfortable chase / scoring)
        #   RRR  8-11        → Medium
        #   RRR > 11         → High
        #   Wickets >= 5     → upgrade one level (tail exposed)
        #   Wickets < 5      → keep as-is
        #
        # Bowling perspective (reverse):
        #   RRR <= 7         → High  (batsmen cruising = bowler under pressure)
        #   RRR  8-11        → Medium
        #   RRR > 11         → Low   (batsmen struggling = bowler comfortable)
        #   Wickets >= 5     → downgrade one level (bowler has done well)
        #   Wickets < 5      → upgrade one level (batsmen still set)

        _LEVELS = ["Low", "Medium", "High", "Extreme"]

        if is_batting:
            # Base level from required run rate
            if rrr_live <= 7:    base_idx = 0   # Low
            elif rrr_live <= 11: base_idx = 1   # Medium
            else:                base_idx = 2   # High
            # Wicket modifier
            if wickets_fallen >= 5:
                base_idx = min(base_idx + 1, 3)  # upgrade (worse for batsman)
        else:
            # Bowling: reverse run-rate logic
            if rrr_live <= 7:    base_idx = 2   # High (easy for batsmen → hard for bowler)
            elif rrr_live <= 11: base_idx = 1   # Medium
            else:                base_idx = 0   # Low  (hard for batsmen → easy for bowler)
            # Wicket modifier (reverse)
            if wickets_fallen >= 5:
                base_idx = max(base_idx - 1, 0)  # downgrade (bowler has wickets)
            elif wickets_fallen < 5:
                base_idx = min(base_idx + 1, 3)  # upgrade (batsmen still set)

        pressure_level = _LEVELS[base_idx]

        # Map level to numeric pi_live (used for pressure-fitness scoring downstream)
        _PI_MAP = {"Low": 0.15, "Medium": 0.40, "High": 0.65, "Extreme": 0.90}
        pi_live = _PI_MAP[pressure_level]

        _LABEL_MAP = {
            "Low":     ("Low",     "#6bcb77"),
            "Medium":  ("Medium",  "#ffd93d"),
            "High":    ("High",    "#ff9f1c"),
            "Extreme": ("Extreme", "#e63946"),
        }
        pressure_label_live = _LABEL_MAP[pressure_level]

        # -- Display live pressure ---------------------------------------------
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Required Run Rate",  f"{rrr_live:.2f} per over")
        kpi2.metric("Wickets Remaining",  str(wickets_left_live))
        kpi3.metric("Pressure Index",     f"{pi_live:.2f} / 1.00")
        kpi4.metric("Pressure Level",     pressure_label_live[0])

        # Live pressure gauge
        fig_pi = go.Figure(go.Indicator(
            mode  ="gauge+number",
            value = round(pi_live * 100, 1),
            number= {"suffix": "%", "font": {"size":32,"color":"#ffffff","family":"Orbitron,monospace"}},
            title = {"text": "<span style='color:#c5d8f5'>Live Pressure Index</span>",
                     "font": {"size": 13}},
            gauge = {
                "axis":  {"range": [0, 100], "tickcolor":"rgba(0,229,255,0.2)",
                          "tickfont":{"color":"#2a3a52","size":9}},
                "bar":   {"color": pressure_label_live[1], "thickness": 0.22},
                "bgcolor": _PLOT,
                "bordercolor": "rgba(0,229,255,0.15)",
                "steps": [
                    {"range": [0, 25],  "color": "rgba(57,255,20,0.08)"},
                    {"range": [25, 50], "color": "rgba(255,215,0,0.08)"},
                    {"range": [50, 75], "color": "rgba(255,140,0,0.08)"},
                    {"range": [75, 100],"color": "rgba(255,59,59,0.10)"},
                ],
                "threshold": {"line":{"color":"#00e5ff","width":3},"thickness":0.72,"value":50},
            },
        ))
        fig_pi.update_layout(height=270, margin=dict(l=20,r=20,t=50,b=10), **_DARK)
        st.plotly_chart(fig_pi)

        st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

        # -- Determine which roles to recommend --------------------------------
        is_batting = "Batting" in situation

        if is_batting:
            st.markdown('### <span class="ic ic-bat"></span> Best Batsmen / All-Rounders to Send In', unsafe_allow_html=True)
            if wickets_fallen <= 5:
                # Early wickets: pure batters are ideal, all-rounders secondary
                eligible_roles  = {"Batter": 1.00, "All-Rounder": 0.80}
                role_note = (
                    f"**{wickets_fallen} wickets fallen** · Early-mid innings. "
                    "Preferring **specialist batters** (pure run-scoring needed). "
                    "All-rounders also considered at slightly lower priority."
                )
            else:
                # Late wickets (>5): need players who can bat AND contribute under pressure
                eligible_roles = {"Batter": 0.90, "All-Rounder": 1.00}
                role_note = (
                    f"**{wickets_fallen} wickets fallen** · Tail approaching. "
                    "Preferring **All-Rounders** (bat + contribute under pressure). "
                    "Aggressive specialist batters also valid."
                )
        else:
            st.markdown('### <span class="ic ic-ball"></span> Best Bowlers / All-Rounders to Bowl Next', unsafe_allow_html=True)
            if wickets_fallen <= 5:
                # Batting side still strong · need wicket-takers
                eligible_roles  = {"Bowler": 1.00, "All-Rounder": 0.85}
                role_note = (
                    f"**{wickets_fallen} wickets fallen by batting side** · Batting team strong. "
                    "Need **specialist wicket-taking bowlers**. "
                    "All-rounders for variation."
                )
            else:
                # Batting side in trouble · contain + take tail wickets fast
                eligible_roles  = {"Bowler": 1.00, "All-Rounder": 0.90}
                role_note = (
                    f"**{wickets_fallen} wickets fallen by batting side** · Tail batting. "
                    "Economy + dot balls critical. **Bowlers and All-Rounders** ideal."
                )

        st.markdown(f'<div class="info-box">{role_note}</div>', unsafe_allow_html=True)

        # -- Score each player for this situation ------------------------------
        # pressure_fitness: how well does player perform at THIS pressure level?
        # We look at their avg performance in innings where pressure_index is close

        # Build pressure-context stats per player from historical data
        if is_batting:
            hist = bat_agg.copy()
            hist["player"] = hist["striker"]
        else:
            hist = bowl_agg.copy()
            hist["player"] = hist["bowler"]
            # Attach pressure from bat side to bowling innings
            bowl_pressure = bat_agg.groupby(["match_id","innings"])["pressure_index"].mean().reset_index()
            hist = hist.merge(bowl_pressure, on=["match_id","innings"], how="left")
            hist["pressure_index"] = hist["pressure_index"].fillna(0.35)

        PRESSURE_WINDOW = 0.15   # look at innings where pressure was within ·0.15
        lo = max(pi_live - PRESSURE_WINDOW, 0)
        hi = min(pi_live + PRESSURE_WINDOW, 1)

        context_perf = (
            hist[(hist["pressure_index"] >= lo) & (hist["pressure_index"] <= hi)]
            .groupby("player")["batting_perf_raw" if is_batting else "bowling_perf_raw"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "ctx_perf", "count": "ctx_innings"})
        )

        # Merge with final IM
        pred_df = final_df[final_df["role"].isin(list(eligible_roles.keys()))].copy()

        # Filter by minimum innings played
        if is_batting:
            pred_df = pred_df[pred_df["bat_innings"] >= min_innings]
        else:
            pred_df = pred_df[pred_df["bowl_innings"] >= min_innings]

        pred_df = pred_df.merge(context_perf, on="player", how="left")
        pred_df["ctx_perf"]    = pred_df["ctx_perf"].fillna(0)
        pred_df["ctx_innings"] = pred_df["ctx_innings"].fillna(0)

        # Role priority multiplier
        pred_df["role_mult"] = pred_df["role"].map(eligible_roles).fillna(0.8)

        # Pressure fitness: how often has this player performed in similar pi contexts
        pred_df["pressure_fitness"] = (
            np.clip(pred_df["ctx_perf"] / 0.4, 0, 1) * 0.6 +
            np.clip(pred_df["ctx_innings"] / 20, 0, 1) * 0.4
        )

        # Composite recommendation score
        pred_df["rec_score"] = (
            0.50 * (pred_df["impact_metric"] / 100)
          + 0.30 * pred_df["pressure_fitness"]
          + 0.20 * pred_df["role_mult"]
        ) * 100

        pred_df = pred_df.sort_values("rec_score", ascending=False).reset_index(drop=True)
        pred_df["rec_rank"] = pred_df.index + 1

        top_preds = pred_df.head(top_n_pred)

        # -- Recommendation bar chart
        role_colors = {"Batter": "#00aaff", "Bowler": "#39ff14", "All-Rounder": "#ffd700"}
        bar_cols    = top_preds["role"].map(role_colors).fillna("#c5d8f5").tolist()

        fig_rec = go.Figure()
        fig_rec.add_trace(go.Bar(
            x            = top_preds["rec_score"][::-1],
            y            = top_preds["player"][::-1],
            orientation  = "h",
            marker=dict(color=bar_cols[::-1], line=dict(color="rgba(0,229,255,0.15)",width=0.8)),
            text         = top_preds["rec_score"][::-1].round(1).astype(str),
            textposition = "outside",
            textfont     = dict(color="#ffffff", size=10, family="Orbitron, monospace"),
            customdata   = top_preds[["role","impact_metric","ctx_innings"]][::-1].values,
            hovertemplate = (
                "<b>%{y}</b><br>"
                "Rec Score: <b>%{x:.1f}</b><br>"
                "Role: %{customdata[0]}<br>"
                "Overall IM: %{customdata[1]:.1f}<br>"
                "Matches in similar pressure: %{customdata[2]:.0f}<extra></extra>"
            ),
        ))
        fig_rec.add_vline(x=50, line_dash="dot", line_color="rgba(0,229,255,0.4)",
                          annotation_text="Baseline (50)",
                          annotation_font=dict(color="#2a3a52",size=9))
        fig_rec.update_layout(
            height   = max(320, top_n_pred * 48),
            xaxis    = dict(range=[0,118], gridcolor=_GRID,
                            tickfont=dict(color="#2a3a52",size=9),
                            title=dict(text="Recommendation Score",font=dict(color="#2a3a52"))),
            yaxis    = dict(tickfont=dict(color=_FONT,size=10)),
            margin   = dict(l=180,r=60,t=14,b=20),
            **_DARK,
        )
        for _role, _col in role_colors.items():
            if _role in top_preds["role"].values:
                fig_rec.add_trace(go.Bar(x=[None],y=[None],name=_role,
                                         marker_color=_col,orientation="h",showlegend=True))
        fig_rec.update_layout(legend=dict(orientation="h",y=1.06,
                                           font=dict(color=_FONT,size=10),bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_rec)

        # ----------------------------------------------------------------------
        # AI MATCH INTELLIGENCE · Why These Players?
        # ----------------------------------------------------------------------
        st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            """<div class="section-header"><div class="section-accent"></div>
<h2>🧠 AI Match Intelligence · Why These Players?</h2></div>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="info-box" style="border-left-color:#a855f7;">'
            '<span style="background:linear-gradient(90deg,#a855f7,#7c3aed);color:#fff;'
            'padding:2px 9px;border-radius:4px;font-size:11px;font-family:Orbitron,monospace;'
            'font-weight:700;letter-spacing:1px;">🧠 AI INSIGHT</span>'
            '&nbsp;&nbsp;The system explains its reasoning for each recommendation based on '
            '<b>Impact Score</b>, <b>Pressure Fitness</b>, <b>Role Suitability</b>, '
            'match situation context, and historical performance data.</div>',
            unsafe_allow_html=True,
        )

        # -- Pre-compute career strike-rate and economy maps ----------------
        _sr_agg = (
            bat_agg[bat_agg["balls_faced"] > 0]
            .groupby("striker")[["runs_scored", "balls_faced"]]
            .sum()
            .assign(career_sr=lambda d: d["runs_scored"] / d["balls_faced"] * 100)
            .reset_index()
            .rename(columns={"striker": "player"})
        )
        _eco_agg = (
            bowl_agg[bowl_agg["overs_bowled"] > 0]
            .groupby("bowler")[["runs_conceded", "overs_bowled"]]
            .sum()
            .assign(career_eco=lambda d: d["runs_conceded"] / d["overs_bowled"])
            .reset_index()
            .rename(columns={"bowler": "player"})
        )
        _sr_map  = dict(zip(_sr_agg["player"],  _sr_agg["career_sr"]))
        _eco_map = dict(zip(_eco_agg["player"], _eco_agg["career_eco"]))

        # -- Explanation generator (closure over match-state vars) ----------
        def _ai_explain(row):
            _player = row["player"]
            _im     = row["impact_metric"]
            _pf     = row["pressure_fitness"] * 100
            _role   = row["role"]
            _ctx_n  = int(row["ctx_innings"])
            _rank   = int(row["rec_rank"])

            # Impact Score classification
            if _im >= 80:    _im_lbl, _im_c = "Elite Impact Score",   "#39ff14"
            elif _im >= 65:  _im_lbl, _im_c = "Very High Impact",     "#00e5ff"
            elif _im >= 50:  _im_lbl, _im_c = "High Impact",          "#ffd700"
            elif _im >= 35:  _im_lbl, _im_c = "Above-Average Impact", "#ff9f1c"
            else:            _im_lbl, _im_c = "Average Impact",       "#c5d8f5"

            # Pressure fitness label
            if _pf >= 70:    _pf_lbl = "Excellent"
            elif _pf >= 50:  _pf_lbl = "Good"
            elif _pf >= 30:  _pf_lbl = "Moderate"
            else:            _pf_lbl = "Limited"

            # Match stage label
            _stage_lbl = {
                "powerplay": "Powerplay (overs 1–5)",
                "middle":    "middle overs (6–15)",
                "death":     "death overs (16+)",
            }[stage_live]

            # Situation-aware narrative (innings 2 chase vs innings 1 set)
            if innings_no == 2:
                _rrr = runs_left / max(overs_left, 0.1)
                if _rrr < 7:
                    _sit = (f"With a required rate of <b>{_rrr:.2f}</b>, the chase is "
                            "manageable · an anchor who rotates strike efficiently fits perfectly.")
                elif _rrr < 10:
                    _sit = (f"The required rate of <b>{_rrr:.2f}</b> demands "
                            "calculated aggression and a high conversion of scoring opportunities.")
                else:
                    _sit = (f"A steep required rate of <b>{_rrr:.2f}</b> calls for an "
                            "explosive, high-impact performer who can shift momentum instantly.")
            else:
                if wickets_fallen < 4:
                    _sit = "Batting first · building a solid foundation and posting a strong total is the priority."
                else:
                    _sit = "Batting first with wickets lost · smart, pressure-resistant batting to rebuild or accelerate now."

            # Wicket-context sentence
            if is_batting:
                if wickets_fallen >= 7:
                    _wkt = (f"At <b>{wickets_fallen} wickets down</b>, a resilient finisher who "
                            "thrives under duress is a premium selection.")
                elif wickets_fallen >= 5:
                    _wkt = (f"At <b>{wickets_fallen} wickets down</b>, this player's ability to "
                            "stabilise and accelerate makes them highly valuable.")
                else:
                    _wkt = (f"With only <b>{wickets_fallen} wickets fallen</b>, there is room for "
                            "this specialist to build a match-defining innings.")
            else:
                if wickets_fallen >= 7:
                    _wkt = (f"The batting side is at <b>{wickets_fallen} wickets</b> · disciplined "
                            "bowling to contain and dismiss the tail is the brief.")
                else:
                    _wkt = (f"<b>{wickets_fallen} wickets fallen</b> by batting side · taking "
                            "another wicket quickly in this phase is the primary objective.")

            # Role-specific metric (SR for batters, Economy for bowlers)
            _role_b = ""
            if is_batting and _role in ("Batter", "All-Rounder"):
                _sr = _sr_map.get(_player)
                if _sr is not None:
                    _sr_d = ("an aggressive striker" if _sr >= 140
                             else ("a solid run-scorer" if _sr >= 120
                                   else "a steady accumulator"))
                    _role_b += (f'<li>Career strike rate <b style="color:#ffd700">{_sr:.1f}</b>'
                                f" · {_sr_d} who compounds pressure on the opposition</li>")
            if not is_batting and _role in ("Bowler", "All-Rounder"):
                _eco = _eco_map.get(_player)
                if _eco is not None:
                    _eco_d = ("elite economy control" if _eco < 7.5
                              else ("competitive economy" if _eco < 9.0
                                    else "aggressive wicket-seeking approach"))
                    _role_b += (f'<li>Career economy <b style="color:#ffd700">{_eco:.2f}</b>'
                                f" RPO · {_eco_d} well-suited to this phase</li>")
            if _role == "All-Rounder":
                _role_b += ("<li>Dual-threat <b>All-Rounder</b> · adds tactical flexibility "
                            "with contributions from both bat and ball</li>")

            # Context exposure sentence
            if _ctx_n >= 15:
                _ctx_s = (f"Proven performer with <b>{_ctx_n} innings</b> "
                          "recorded in historically similar pressure windows")
            elif _ctx_n >= 8:
                _ctx_s = (f"Good exposure across <b>{_ctx_n} innings</b> "
                          "in comparable pressure conditions")
            elif _ctx_n >= 3:
                _ctx_s = (f"Some experience across <b>{_ctx_n} innings</b> "
                          "in similar pressure scenarios")
            else:
                _ctx_s = ("Limited history in identical pressure windows · "
                          "the overall Impact Score is the primary signal here")

            # Assemble all bullet points
            _bullets = (
                f'<li><b style="color:{_im_c}">{_im_lbl} ({_im:.1f}/100)</b>'
                f" · system ranks this player in the top recommendation tier</li>"
                f"<li>Pressure Fitness: <b>{_pf_lbl}</b> ({_pf:.0f}%)"
                f" · measured performance under historically similar pressure conditions</li>"
                f"<li>{_ctx_s}</li>"
                f"<li>Optimal for the <b>{_stage_lbl}</b> · {_sit}</li>"
                f"<li>{_wkt}</li>"
                + _role_b
            )

            # Rank badge colour
            if _rank == 1:   _bc = "#ffd700"
            elif _rank <= 3: _bc = "#00e5ff"
            else:            _bc = "#a855f7"

            return (
                f'<div style="background:rgba(8,17,42,0.92);'
                f'border:1px solid rgba(168,85,247,0.22);'
                f'border-left:3px solid {_bc};border-radius:10px;'
                f'padding:16px 20px;margin-bottom:14px;'
                f'font-family:\'Inter\',sans-serif;">'
                f'<div style="display:flex;align-items:center;gap:10px;'
                f'margin-bottom:10px;flex-wrap:wrap;">'
                f'<span style="background:{_bc};color:#050b18;min-width:26px;height:26px;'
                f'border-radius:50%;display:inline-flex;align-items:center;'
                f'justify-content:center;font-family:Orbitron,monospace;'
                f'font-weight:900;font-size:12px;">#{_rank}</span>'
                f'<span style="font-size:16px;font-weight:700;color:#e8f4fd;">{_player}</span>'
                f'<span style="background:rgba(168,85,247,0.15);color:#a855f7;'
                f'padding:2px 9px;border-radius:4px;font-size:11px;font-weight:600;'
                f'border:1px solid rgba(168,85,247,0.3);">{_role}</span>'
                f'<span style="margin-left:auto;background:rgba(168,85,247,0.10);'
                f'color:#c084fc;padding:2px 9px;border-radius:4px;font-size:10px;'
                f'font-family:Orbitron,monospace;border:1px solid rgba(168,85,247,0.25);'
                f'letter-spacing:1px;">🧠 AI INSIGHT</span>'
                f'</div>'
                f'<ul style="margin:0;padding-left:20px;color:#c5d8f5;'
                f'font-size:13.5px;line-height:2.0;">'
                f'{_bullets}'
                f'</ul>'
                f'</div>'
            )

        for _, _ai_row in top_preds.iterrows():
            st.markdown(_ai_explain(_ai_row), unsafe_allow_html=True)

        st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

        # -- Detailed player cards
        st.markdown("""<div class="section-header"><div class="section-accent"></div><h2>Player Detail Cards</h2></div>""", unsafe_allow_html=True)
        for _, row in top_preds.iterrows():
            with st.expander(
                f"#{int(row['rec_rank'])}  {row['player']}  ·  "
                f"{row['role']}  ·  IM: {row['impact_metric']:.1f}  ·  "
                f"Rec Score: {row['rec_score']:.1f}"
            ):
                dc1, dc2, dc3, dc4 = st.columns(4)
                dc1.metric("Overall IM",          f"{row['impact_metric']:.1f} / 100")
                dc2.metric("Rec Score",            f"{row['rec_score']:.1f} / 100")
                dc3.metric("Pressure Fitness",    f"{row['pressure_fitness']*100:.1f}%")
                dc4.metric("Similar-ctx innings", f"{int(row['ctx_innings'])}")

                # Mini trend for this player
                mini_trend = make_trend(row["player"])
                if mini_trend:
                    mini_trend.update_layout(height=240, title="")
                    st.plotly_chart(mini_trend)

        # -- How scores are computed
        st.markdown("""<div class="section-header"><div class="section-accent"></div><h2>How Scores Are Computed</h2></div>""", unsafe_allow_html=True)
        st.latex(
            r"RecScore = 0.50 \times \frac{IM}{100} "
            r"+ 0.30 \times PressureFitness "
            r"+ 0.20 \times RoleMultiplier"
        )
        st.markdown("""
| Term               | Meaning |
|--------------------|----------|
| **IM**             | Overall Impact Metric (rolling last 10 innings, 0–100) |
| **PressureFitness**| How well the player performs in historically similar pressure situations |
| **RoleMultiplier** | Priority weight for the player's role in this situation (1.0 = ideal, 0.8 = secondary) |
""")
        if wickets_fallen > 5 and is_batting:
            st.markdown('<div class="info-box" style="border-left-color:#ffd700;color:#a08040"><span class="ic ic-warn"></span> <b>Late-order alert:</b> With 6+ wickets down, All-Rounders receive a +priority because they balance run-scoring ability with the resilience needed to bat under extreme pressure.</div>', unsafe_allow_html=True)
        elif wickets_fallen > 5 and not is_batting:
            st.markdown('<div class="info-box" style="border-left-color:#39ff14;color:#207a20"><span class="ic ic-bulb"></span> <b>Tail-batting situation:</b> Economy and dot-ball bowlers are premium. All-Rounders who bowl tight lines are also ranked highly.</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# PAGE: ADVANCED ANALYTICS
# ------------------------------------------------------------------------------
elif page == "Advanced Analytics":
    st.markdown("""
    <div class="page-title-bar">
      <h1><span class="ic ic-lg ic-analytics"></span> Advanced Analytics</h1>
      <div class="page-desc">Opposition Strength · Match Importance · Form Recovery · Phase Contribution · IM Formula Breakdown</div>
    </div>
    """, unsafe_allow_html=True)

    _adv_tabs = st.tabs([
        "Opposition Strength",
        "Match Importance",
        "Form Recovery",
        "Phase Contribution",
        "IM Component Breakdown",
    ])

    # ---------------------------------------------------------------------
    # TAB 1: Opposition Strength
    # ---------------------------------------------------------------------
    with _adv_tabs[0]:
        st.markdown("""<div class="section-header"><div class="section-accent"></div>
<h2>Opposition Strength vs Impact Score</h2></div>""", unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box">Players whose high Impact Scores are backed by '
            '<b style="color:#a855f7">strong opposition quality</b> demonstrate '
            'genuinely elite performance · not just statistical padding against weak attacks.</div>',
            unsafe_allow_html=True,
        )

        _opp_df = final_df[["player","role","impact_metric","opp_strength"]].dropna()
        _opp_top = min(80, len(_opp_df))
        _opp_plot = _opp_df.nlargest(_opp_top, "impact_metric")

        _role_c_map = {"Batter":"#00aaff","Bowler":"#39ff14","All-Rounder":"#ffd700"}
        fig_opp = px.scatter(
            _opp_plot, x="opp_strength", y="impact_metric",
            color="role",
            color_discrete_map=_role_c_map,
            hover_name="player",
            hover_data={"role":True,"impact_metric":":.1f","opp_strength":":.3f"},
            size="impact_metric",
            size_max=22,
            labels={"opp_strength":"Opposition Strength (0–1)","impact_metric":"Impact Metric"},
        )
        # Quadrant lines
        _opp_med = float(_opp_plot["opp_strength"].median())
        _im_med  = float(_opp_plot["impact_metric"].median())
        fig_opp.add_vline(x=_opp_med, line_dash="dot", line_color="rgba(0,229,255,0.25)")
        fig_opp.add_hline(y=_im_med,  line_dash="dot", line_color="rgba(0,229,255,0.25)")
        # Zone labels
        for _xt, _yt, _lbl, _lc in [
            (_opp_med+0.01, _im_med+1, "Elite Performers",   "#39ff14"),
            (_opp_med-0.12, _im_med+1, "Context Beneficiaries","#ffd700"),
            (_opp_med+0.01, _im_med-4, "Underperformers",   "#ff6b9d"),
            (_opp_med-0.12, _im_med-4, "Developing",        "#c5d8f5"),
        ]:
            fig_opp.add_annotation(x=_xt, y=_yt, text=_lbl,
                                   font=dict(color=_lc, size=9, family="Orbitron,monospace"),
                                   showarrow=False, opacity=0.7)
        fig_opp.update_traces(marker=dict(line=dict(width=1, color="rgba(0,229,255,0.3)")))
        fig_opp.update_layout(height=480, margin=dict(l=40,r=40,t=20,b=30),
                              legend=dict(font=dict(color=_FONT,size=10),bgcolor="rgba(0,0,0,0)"),
                              **_DARK)
        st.plotly_chart(fig_opp, width='stretch')

        # Top opposition-adjusted performers table
        st.markdown("""<div class="section-header"><div class="section-accent"></div>
<h2>Top Opposition-Adjusted Performers</h2></div>""", unsafe_allow_html=True)
        _adj = _opp_df.copy()
        _adj["opp_adj_im"] = (_adj["impact_metric"] * (0.7 + 0.3*_adj["opp_strength"])).round(2)
        _adj_top = _adj.nlargest(15,"opp_adj_im")[["player","role","impact_metric","opp_strength","opp_adj_im"]]
        _adj_top.columns = ["Player","Role","Raw IM","Opp Strength","Opp-Adjusted IM"]
        st.dataframe(_adj_top.reset_index(drop=True), height=380)

    # ---------------------------------------------------------------------
    # TAB 2: Match Importance
    # ---------------------------------------------------------------------
    with _adv_tabs[1]:
        st.markdown("""<div class="section-header"><div class="section-accent"></div>
<h2>Match Importance Distribution</h2></div>""", unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box">Performances in <b style="color:#ffd700">high-stakes matches</b> '
            '(Finals, Semi-Finals, Qualifiers) carry more weight in the Impact Metric · '
            'rewarding players who deliver when it matters most.</div>',
            unsafe_allow_html=True,
        )

        # Distribution across matches
        _mi_counts = matches_df["match_importance"].value_counts().reset_index()
        _mi_counts.columns = ["Weight","Count"]
        _mi_label = {1.0:"League Match",1.2:"Qualifier / Playoff",1.3:"Semi-Final",1.5:"Final"}
        _mi_counts["Stage"] = _mi_counts["Weight"].map(_mi_label).fillna("League Match")
        _mi_counts["Color"] = _mi_counts["Weight"].map(
            {1.0:"#2a3a52",1.2:"#00aaff",1.3:"#ffd700",1.5:"#39ff14"}
        ).fillna("#2a3a52")

        fig_mi = go.Figure(go.Bar(
            x=_mi_counts["Stage"], y=_mi_counts["Count"],
            marker_color=_mi_counts["Color"].tolist(),
            text=_mi_counts["Count"], textposition="outside",
            textfont=dict(color="#ffffff",family="Orbitron,monospace",size=12),
        ))
        fig_mi.update_layout(
            height=320,
            xaxis=dict(tickfont=dict(color=_FONT,size=11)),
            yaxis=dict(gridcolor=_GRID,tickfont=dict(color="#2a3a52",size=9),
                       title=dict(text="No. of Matches",font=dict(color="#2a3a52"))),
            margin=dict(l=30,r=20,t=24,b=20),
            **_DARK,
        )
        st.plotly_chart(fig_mi, width='stretch')

        # Top performers in high-importance matches
        st.markdown("""<div class="section-header"><div class="section-accent"></div>
<h2>Best Big-Match Performers</h2></div>""", unsafe_allow_html=True)
        _bmp = final_df[["player","role","impact_metric","match_importance"]].dropna()
        _bmp = _bmp.sort_values(["match_importance","impact_metric"],ascending=False)
        _bmp_show = _bmp.nlargest(15,"match_importance")[["player","role","impact_metric","match_importance"]]

        fig_bmp = go.Figure()
        _bmp_cols = {"Batter":"#00aaff","Bowler":"#39ff14","All-Rounder":"#ffd700"}
        for _role_b, _grp in _bmp_show.groupby("role"):
            fig_bmp.add_trace(go.Bar(
                x=_grp["player"], y=_grp["impact_metric"],
                name=_role_b,
                marker_color=_bmp_cols.get(_role_b,"#c5d8f5"),
                text=_grp["impact_metric"].round(1).astype(str),
                textposition="outside",
                textfont=dict(color="#ffffff",size=9,family="Orbitron,monospace"),
            ))
        fig_bmp.update_layout(
            height=340, barmode="group",
            xaxis=dict(tickfont=dict(color=_FONT,size=9),tickangle=-35),
            yaxis=dict(range=[0,115],gridcolor=_GRID,tickfont=dict(color="#2a3a52",size=9)),
            legend=dict(font=dict(color=_FONT,size=10),bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=20,r=20,t=20,b=80),
            **_DARK,
        )
        st.plotly_chart(fig_bmp, width='stretch')

    # ---------------------------------------------------------------------
    # TAB 3: Form Recovery
    # ---------------------------------------------------------------------
    with _adv_tabs[2]:
        st.markdown("""<div class="section-header"><div class="section-accent"></div>
<h2>Form Recovery Indicator</h2></div>""", unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box">The <b style="color:#39ff14">Form Recovery Score</b> measures '
            'how much a player outperforms their own rolling 5-innings average. '
            'A high score indicates a player who bounces back strongly after lean patches.</div>',
            unsafe_allow_html=True,
        )

        _fr_df = final_df[["player","role","impact_metric","form_recovery"]].dropna()
        _fr_df = _fr_df.nlargest(40, "form_recovery")
        _fr_df["form_pct"] = (_fr_df["form_recovery"] * 100).round(1)

        fig_fr = go.Figure()
        _fr_role_c = {"Batter":"#00aaff","Bowler":"#39ff14","All-Rounder":"#ffd700"}
        fig_fr.add_trace(go.Bar(
            x=_fr_df["player"][::-1],
            y=_fr_df["form_pct"][::-1],
            orientation="v",
            marker=dict(
                color=[_fr_role_c.get(r,"#c5d8f5") for r in _fr_df["role"][::-1]],
                line=dict(color="rgba(0,229,255,0.15)",width=0.8),
            ),
            text=_fr_df["form_pct"][::-1].astype(str)+"%",
            textposition="outside",
            textfont=dict(color="#ffffff",size=8,family="Orbitron,monospace"),
            hovertemplate="<b>%{x}</b><br>Form Recovery: %{y:.1f}%<extra></extra>",
        ))
        fig_fr.add_hline(y=50, line_dash="dot", line_color="rgba(0,229,255,0.3)",
                         annotation_text="Baseline (50%)",
                         annotation_font=dict(color="#2a3a52",size=9))
        fig_fr.update_layout(
            height=420,
            xaxis=dict(tickfont=dict(color=_FONT,size=8),tickangle=-45),
            yaxis=dict(range=[0,115],gridcolor=_GRID,
                       tickfont=dict(color="#2a3a52",size=9),
                       title=dict(text="Form Recovery Score (%)",font=dict(color="#2a3a52"))),
            margin=dict(l=20,r=20,t=20,b=100),
            **_DARK,
        )
        st.plotly_chart(fig_fr, width='stretch')

        # Scatter: Form Recovery vs Impact Metric
        st.markdown("""<div class="section-header"><div class="section-accent"></div>
<h2>Form Recovery vs Impact Score</h2></div>""", unsafe_allow_html=True)
        _fr_all = final_df[["player","role","impact_metric","form_recovery"]].dropna()
        fig_fr2 = px.scatter(
            _fr_all.nlargest(100,"impact_metric"),
            x="form_recovery", y="impact_metric",
            color="role",
            color_discrete_map=_fr_role_c,
            hover_name="player",
            labels={"form_recovery":"Form Recovery Score (0–1)","impact_metric":"Impact Metric"},
            size="impact_metric", size_max=18,
        )
        fig_fr2.add_vline(x=0.5, line_dash="dot", line_color="rgba(0,229,255,0.25)")
        fig_fr2.add_hline(y=50,  line_dash="dot", line_color="rgba(0,229,255,0.25)")
        fig_fr2.update_traces(marker=dict(line=dict(width=1,color="rgba(0,229,255,0.3)")))
        fig_fr2.update_layout(height=420,margin=dict(l=40,r=30,t=20,b=30),
                              legend=dict(font=dict(color=_FONT,size=10),bgcolor="rgba(0,0,0,0)"),
                              **_DARK)
        st.plotly_chart(fig_fr2, width='stretch')

    # ---------------------------------------------------------------------
    # TAB 4: Phase Contribution
    # ---------------------------------------------------------------------
    with _adv_tabs[3]:
        st.markdown("""<div class="section-header"><div class="section-accent"></div>
<h2>Phase-wise Impact Contribution</h2></div>""", unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box" style="border-left-color:#ffd700;">'
            'Phase weights applied to batting performance score: '
            '<b style="color:#c5d8f5">Powerplay · 0.9</b> · '
            '<b style="color:#ffd700">Middle Overs · 1.0</b> · '
            '<b style="color:#39ff14">Death Overs · 1.2</b> · '
            'rewarding contributions in the most impactful phase.</div>',
            unsafe_allow_html=True,
        )

        # Top-N player selector
        _ph_n = st.slider("Players to show", 10, 40, 20, key="phase_n")
        _top_bat = bat_agg.groupby("striker")[["pp_runs","middle_runs","death_runs"]].sum().reset_index()
        _top_bat["total"] = _top_bat["pp_runs"] + _top_bat["middle_runs"] + _top_bat["death_runs"]
        _top_bat = _top_bat[_top_bat["total"] > 50].nlargest(_ph_n, "total")

        fig_phase = go.Figure()
        for _ph_col, _ph_lbl, _ph_c, _ph_w in [
            ("pp_runs",     "Powerplay (x0.9)",   "#00aaff", 0.9),
            ("middle_runs", "Middle (x1.0)",      "#ffd700", 1.0),
            ("death_runs",  "Death (x1.2)",       "#39ff14", 1.2),
        ]:
            _weighted = _top_bat[_ph_col] * _ph_w
            fig_phase.add_trace(go.Bar(
                name=_ph_lbl,
                x=_top_bat["striker"],
                y=_weighted,
                marker_color=_ph_c,
                hovertemplate=f"<b>%{{x}}</b><br>{_ph_lbl}: <b>%{{customdata}}</b> runs ·{_ph_w}<extra></extra>",
                customdata=_top_bat[_ph_col].values,
            ))
        fig_phase.update_layout(
            barmode="stack",
            height=460,
            xaxis=dict(tickfont=dict(color=_FONT, size=9), tickangle=-45),
            yaxis=dict(gridcolor=_GRID, tickfont=dict(color="#2a3a52",size=9),
                       title=dict(text="Weighted Phase Runs",font=dict(color="#2a3a52"))),
            legend=dict(font=dict(color=_FONT,size=10),bgcolor="rgba(0,0,0,0)",
                        orientation="h",y=1.06),
            margin=dict(l=30,r=20,t=30,b=120),
            **_DARK,
        )
        st.plotly_chart(fig_phase, width='stretch')

        # Phase weight explanation
        st.markdown("""<div class="section-header"><div class="section-accent"></div>
<h2>Phase Multiplier Effect</h2></div>""", unsafe_allow_html=True)
        _pmx_data = pd.DataFrame({
            "Phase":   ["Powerplay (1–6)", "Middle Overs (7–15)", "Death Overs (16–20)"],
            "Weight":  [0.9, 1.0, 1.2],
            "Rationale": [
                "Foundation setting · slightly discounted as conditions favour batting",
                "Neutral weight · standard run-scoring period",
                "Premium for finishing under extreme pressure and required run-rate",
            ],
        })
        st.dataframe(_pmx_data, height=160)

    # ---------------------------------------------------------------------
    # TAB 5: IM Component Breakdown
    # ---------------------------------------------------------------------
    with _adv_tabs[4]:
        st.markdown("""<div class="section-header"><div class="section-accent"></div>
<h2>IM Component Breakdown per Player</h2></div>""", unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box" style="border-left-color:#a855f7;">'
            'Each component contributes independently to the Impact Metric. '
            'This helps understand <b style="color:#00e5ff">exactly why</b> a player is ranked where they are '
            '· and makes the metric <b>transparent and manipulation-resistant</b>.</div>',
            unsafe_allow_html=True,
        )

        _sel_p = st.selectbox(
            "Select player",
            sorted(final_df["player"].tolist()),
            key="adv_player_sel",
        )
        _prow = final_df[final_df["player"] == _sel_p]
        if not _prow.empty:
            _pr = _prow.iloc[0]

            # Component weights · values = contribution
            _comps = {
                "Performance": (0.30, float(_pr.get("avg_perf", 0.3))),
                "Pressure Index": (0.20, 0.5),  # stored as avg in bat_im; approx
                "WPA": (0.20, 0.5),
                "Clutch Factor": (0.10, 0.33),
                "Opp Strength": (0.10, float(_pr.get("opp_strength", 0.5))),
                "Match Importance": (0.05, float(_pr.get("match_importance", 0.4))),
                "Form Recovery": (0.05, float(_pr.get("form_recovery", 0.5))),
            }
            _comp_df = pd.DataFrame(
                [{"Component": k, "Weight": v[0], "Score (0–1)": round(v[1],3),
                  "Contribution": round(v[0]*v[1],4)} for k,v in _comps.items()]
            )

            # KPI strip
            kk = st.columns(4)
            kk[0].metric("Overall IM",      f"{_pr['impact_metric']:.1f}")
            kk[1].metric("Opp Strength",    f"{_pr.get('opp_strength',0.5):.3f}")
            kk[2].metric("Match Importance",f"{_pr.get('match_importance',0.4):.3f}")
            kk[3].metric("Form Recovery",   f"{_pr.get('form_recovery',0.5):.3f}")

            # Waterfall/stacked donut of contributions
            fig_comp = go.Figure(go.Bar(
                x=_comp_df["Component"],
                y=_comp_df["Contribution"],
                marker=dict(
                    color=["#39ff14","#00e5ff","#00e5ff","#ff9f1c","#a855f7","#ff6b9d","#ff6b9d"],
                    line=dict(color="rgba(0,229,255,0.2)",width=1),
                ),
                text=_comp_df["Contribution"].round(4).astype(str),
                textposition="outside",
                textfont=dict(color="#ffffff",size=10,family="Orbitron,monospace"),
                hovertemplate="<b>%{x}</b><br>Contribution: <b>%{y:.4f}</b><extra></extra>",
            ))
            fig_comp.update_layout(
                height=340,
                xaxis=dict(tickfont=dict(color=_FONT,size=10)),
                yaxis=dict(range=[0,0.35],gridcolor=_GRID,tickfont=dict(color="#2a3a52",size=9),
                           title=dict(text="Weighted Contribution",font=dict(color="#2a3a52"))),
                margin=dict(l=30,r=20,t=20,b=30),
                **_DARK,
            )
            st.plotly_chart(fig_comp, width='stretch')

            # Donut of contributions
            fig_donut = go.Figure(go.Pie(
                labels=_comp_df["Component"],
                values=_comp_df["Contribution"].clip(lower=0.001),
                hole=0.55,
                marker=dict(colors=["#39ff14","#00e5ff","#00bbff","#ff9f1c",
                                    "#a855f7","#ff6b9d","#ff4090"],
                            line=dict(color="#050b18",width=2)),
                textfont=dict(color="#ffffff",size=10),
                hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
            ))
            fig_donut.update_layout(
                height=380,
                annotations=[dict(text=f"IM\n{_pr['impact_metric']:.1f}",
                                  font=dict(color="#00e5ff",size=14,family="Orbitron,monospace"),
                                  x=0.5,y=0.5,showarrow=False)],
                legend=dict(font=dict(color=_FONT,size=10),bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=20,r=20,t=20,b=20),
                **_DARK,
            )
            st.plotly_chart(fig_donut, width='stretch')

            st.dataframe(_comp_df, height=280)

        # -- Anti-gaming explanation -----------------------------------------------------
        st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
        st.markdown("""<div class="section-header"><div class="section-accent"></div>
<h2><span class="ic ic-lg ic-shield"></span> Why This Metric Is Manipulation-Resistant</h2></div>""", unsafe_allow_html=True)
        st.markdown("""
<div style="background:rgba(8,17,42,0.92);border:1px solid rgba(0,229,255,0.15);
border-radius:12px;padding:20px 26px;font-size:13.5px;line-height:2.1;color:#c5d8f5">
<ul style="margin:0;padding-left:20px">
  <li><b style="color:#39ff14">Context-aware pressure</b> · raw stats without pressure context inflate scores<br>
      → the Pressure Index penalises comfortable conditions and rewards crisis performance</li>
  <li><b style="color:#00e5ff">Win Probability Added</b> · a player can score runs that don't help win;
      WPA ensures only match-shifting contributions are rewarded</li>
  <li><b style="color:#a855f7">Opposition Strength</b> · farming stats against weak attacks is discounted;
      performances against elite bowlers/batters receive a multiplicative boost</li>
  <li><b style="color:#ffd700">Match Importance</b> · inflated league-stage stats are discounted;
      finals and semis carry 1.5× weight</li>
  <li><b style="color:#ff9f1c">Clutch Factor</b> · compares high-pressure performance to personal average;
      a player cannot inflate this through easy innings</li>
  <li><b style="color:#ff6b9d">Recency Weighting</b> · rolling last-10 innings prevents one historic
      season from dominating the metric indefinitely</li>
  <li><b style="color:#c5d8f5">Form Recovery</b> · rewards genuine bouncebacks, not sustained
      luck streaks, by comparing to the player's own rolling baseline</li>
</ul>
</div>""", unsafe_allow_html=True)

# Footer is embedded in premium sidebar above
