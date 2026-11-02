# 🏏 IPL Player Impact Metric — Streamlit Dashboard

> **A full-stack cricket analytics platform** that processes every IPL ball-by-ball delivery, computes a multi-dimensional **Impact Metric (IM)** for every player, and renders six interactive pages inside a premium dark sports-analytics UI.

---

## Table of Contents

1. [Project Overview](#1-project-overview)  
2. [Tech Stack & Libraries](#2-tech-stack--libraries)  
3. [Project Structure](#3-project-structure)  
4. [Installation & Setup](#4-installation--setup)  
5. [Dataset Format](#5-dataset-format)  
6. [Data Pipeline](#6-data-pipeline)  
7. [Impact Metric Formula](#7-impact-metric-formula)  
8. [Consistency Score](#8-consistency-score)  
9. [Match Impact Timeline](#9-match-impact-timeline)  
10. [Role Classification](#10-role-classification)  
11. [Dashboard Pages](#11-dashboard-pages)  
12. [Running the App](#12-running-the-app)  
13. [Formulas Quick-Reference](#13-formulas-quick-reference)  

---

## 1. Project Overview

The dashboard answers a key question in cricket analytics:

> *Which player genuinely changed the outcome of a match, and who does it consistently across seasons?*

Standard metrics like runs or wickets ignore context. This system measures **impact** — how much each player shifted the probability of their team winning, under what pressure, against how strong an opposition, in how important a match.

**Key outputs:**

| Output | Description |
|--------|-------------|
| `impact_metric` | Final 0–100 score for every player |
| `bat_im` / `bowl_im` | Specialised batting and bowling sub-scores |
| `pressure_index` | Context difficulty at the time of play |
| `wpa_norm` | Normalised Win Probability Added |
| `clutch_score` | Performance uplift under high-pressure situations |
| `consistency_score` | Stability across last 10 innings |

---

## 2. Tech Stack & Libraries

| Library | Purpose |
|---------|---------|
| `streamlit` | Web UI framework (dark glassmorphism theme) |
| `pandas` | Data loading, aggregation, merging |
| `numpy` | Vectorised maths |
| `plotly` | Interactive charts — gauge, bar, scatter, radar, pie, waterfall |
| `plotly.express` | Quick statistical charts in Advanced Analytics |
| `scikit-learn` | Logistic Regression for Win Probability model |
| `matplotlib` | Fallback patches (chart legends) |

---

## 3. Project Structure

```
hackemind/
├── ipl_impact_dashboard.py   ← Single-file Streamlit application (3300+ lines)
├── logo.jpg                  ← Sidebar brand logo (base64-embedded at runtime)
├── requirements.txt          ← Python dependencies
├── README.md                 ← This file
├── .gitignore                ← Excludes dataset, images, caches
│
└── ipl_male_csv2 (1)/        ← Raw ball-by-ball CSVs (NOT in repo — see Setup)
    ├── 1082591.csv               Ball-by-ball data
    ├── 1082591_info.csv          Match metadata
    └── ...
```

> **Note:** The dataset folder (`ipl_male_csv2 (1)/`) is **not** included in the repository due to size. Download it separately (see Installation below).

---

## 4. Installation & Setup

### Prerequisites

- Python 3.9+
- `pip`

### Step 1 — Clone the repo

```bash
git clone https://github.com/jyotkonkani30/Hackamind-ipl-impact-metric.git
cd Hackamind-ipl-impact-metric
```

### Step 2 — Create virtual environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Place the dataset

Download the **Cricsheet IPL ball-by-ball CSV** dataset and place it in the project root as:

```
ipl_male_csv2 (1)/
```

Each match has two files: `{match_id}.csv` (deliveries) and `{match_id}_info.csv` (metadata).

### Step 5 — Run the dashboard

```bash
streamlit run ipl_impact_dashboard.py
```

Open **http://localhost:8501** in your browser.

---

## 5. Dataset Format

The dashboard reads two CSV types from `ipl_male_csv2 (1)/` per match:

### `{match_id}_info.csv` — Match metadata

Each row is `info,key,value`. Fields extracted:

| Field | Description |
|-------|-------------|
| `team` | Both team names (two rows) |
| `winner` | Winning team name |
| `toss_winner`, `toss_decision` | Toss result |
| `venue`, `city` | Ground info |
| `season` | IPL season year |
| `event` | Match stage (Final, Qualifier, etc.) |
| `date` | Match date |

### `{match_id}.csv` — Ball-by-ball deliveries

| Column | Description |
|--------|-------------|
| `match_id` | Match identifier |
| `innings` | 1 or 2 |
| `ball` | Over.delivery (e.g. `3.4`) |
| `striker` | Batter facing |
| `bowler` | Bowler delivering |
| `batting_team` / `bowling_team` | Team names |
| `runs_off_bat` | Runs scored by batter |
| `extras`, `wides`, `noballs` | Extra runs |
| `wicket_type` | How the wicket fell (empty = no wicket) |
| `player_dismissed` | Dismissed player name |

---

## 6. Data Pipeline

```
Raw CSVs
  │
  ├─ Step 1:  Parse info CSVs → matches_df (metadata + match_importance)
  ├─ Step 2:  Load ball CSVs → balls DataFrame
  │           • Derive: total_runs, is_wicket, is_legal, over, match_stage
  │           • Merge: matches_df (winner, date, season, match_importance)
  │           • Cumulative: cum_runs, cum_wickets per match+innings
  ├─ Step 3:  Batting aggregation (per match–innings–striker)
  ├─ Step 4:  Pressure Index calculation → per innings
  ├─ Step 5:  Batting Performance Score (phase-weighted)
  ├─ Step 6:  Bowling aggregation (per match–innings–bowler)
  ├─ Step 7:  Opposition Strength (bat & bowl)
  ├─ Step 8:  Win Probability model (Logistic Regression)
  ├─ Step 9:  Clutch Factor
  ├─ Step 10: Per-innings IM (7 components)
  ├─ Step 11: Recency-weighted rolling IM (exponential decay, factor 0.6)
  ├─ Step 12: Consistency Score (1/σ of last 10 innings IM)
  ├─ Step 13: Match Impact Timeline (over-by-over WPA per match)
  └─ Step 14: Final table (role classification, 0–100 normalisation, rank)
```

All data loading and computation is wrapped in `@st.cache_data` so it runs once and is cached across page navigations.

---

## 7. Impact Metric Formula

### 7.1 Batting Performance Score

```
Phase Multiplier = (pp_runs/total × 0.9) + (middle_runs/total × 1.0) + (death_runs/total × 1.2)
                   clipped to [0.9, 1.2]

run_component = clip(runs / 50,  0, 1)
sr_component  = clip(strike_rate / 150, 0, 1)
bnd_component = clip(boundary_runs / runs, 0, 1)

batting_perf_raw = (0.40×run + 0.30×sr + 0.20×boundary + 0.10×pressure×0.15)
                   × phase_multiplier
```

### 7.2 Bowling Performance Score

```
wkt_component = clip(wickets / 3,          0, 1)
eco_component = clip(1 - economy_rate/12,  0, 1)
dot_pct       = dot_balls / legal_balls

bowling_perf_raw = 0.50×wkt + 0.30×eco + 0.20×dot_pct
```

### 7.3 Pressure Index

**Innings 1 (batting first):**
```
pressure = clip(0.4×(wickets/10) + 0.6×stage_multiplier, 0, 1)
```

**Innings 2 (chasing):**
```
rrr_pressure = min(required_run_rate / 12, 1.0)
pressure = clip(0.5×rrr_pressure + 0.3×(wickets/10) + 0.2×stage_mult, 0, 1)
```

| Range | Label |
|-------|-------|
| 0.00 – 0.24 | Low |
| 0.25 – 0.49 | Medium |
| 0.50 – 0.74 | High |
| 0.75 – 1.00 | Extreme |

### 7.4 Win Probability Added (WPA)

Logistic Regression trained on match state features → predicts `P(batting team wins)` at each delivery.

```
delta_wp[i]  = win_prob[i] - win_prob[i-1]
WPA_bat      = Σ delta_wp (balls faced)
WPA_bowl     = Σ (-delta_wp) (balls bowled)
wpa_norm     = sigmoid(WPA × 3)
```

### 7.5 Clutch Factor

```
clutch_ratio = clip(high_pressure_avg / overall_avg, 0, 3)
clutch_score = clutch_ratio / 3
```

### 7.6 Opposition Strength

- **Batters**: based on career economy of bowlers faced → `clip(1 - avg_eco/12, 0, 1)` min-max normalised  
- **Bowlers**: based on career strike rate of batters faced → `clip(avg_sr/150, 0, 1)` min-max normalised

### 7.7 Match Importance

| Stage | Normalised Weight |
|-------|------------------|
| Final | 0.80 |
| Semi-final | 0.65 |
| Qualifier / Eliminator | 0.55 |
| Group stage | 0.40 |

### 7.8 Form Recovery

```
z_score = (current_perf - rolling_5_mean) / (rolling_5_std + 0.01)
form_recovery = clip(z_score/4 + 0.5, 0, 1)
```

### 7.9 Final IM Assembly

$$
\text{IM}_{\text{innings}} = 0.30 \times \text{Perf} + 0.20 \times \text{PI} + 0.20 \times \text{WPA} + 0.10 \times \text{CF} + 0.10 \times \text{Opp} + 0.05 \times \text{MI} + 0.05 \times \text{FR}
$$

| Component | Weight |
|-----------|--------|
| Performance Score | 30% |
| Pressure Index | 20% |
| Win Probability Added | 20% |
| Clutch Factor | 10% |
| Opposition Strength | 10% |
| Match Importance | 5% |
| Form Recovery | 5% |

### 7.10 Recency-Weighted Rolling IM

Exponential decay (factor = 0.6) over last 10 innings, most recent gets highest weight.

### 7.11 Sigmoid Normalization to 0–100

```
z      = (rolling_IM - mean) / std
im_100 = sigmoid(z × 1.5) × 100
```

Maps every player to a 0–100 scale with 50 as the league average.

---

## 8. Consistency Score

```
σ = std of innings IM over last 10 innings
raw_consistency = clip(1 / (σ + 0.01), 0, 200)
consistency_score = clip(raw / P95, 0, 1)
```

- **1.0** = extremely consistent · **0.0** = highly erratic

---

## 9. Match Impact Timeline

Per-over WPA tracking for a selected match:

```
delta_wp_over = Σ delta_wp for all deliveries in that over
cum_wp        = cumulative Σ from over 1
```

Rendered as combined bar (WPA per over) + line (cumulative WPA) chart with boundary stars and wicket markers.

---

## 10. Role Classification

| Condition | Role |
|-----------|------|
| bat_innings ≥ 5 AND bowl_innings ≥ 5 | All-Rounder |
| bat_innings > 0 AND bowl_innings < 3 | Batter |
| bowl_innings > 0 AND bat_innings < 3 | Bowler |
| Otherwise (ratio-based with total_matches) | All-Rounder |

Small-sample shrinkage pulls low-innings players toward the league average of 50.

---

## 11. Dashboard Pages

The app has **6 pages** navigated via a glassmorphism sidebar with custom SVG cricket icons.

### 📊 Leaderboard
- KPI cards (total players, batters, bowlers, all-rounders)
- **Innings-Based Leaderboard**: filter by Batting / Bowling / Total innings with min-innings slider
- Horizontal bar chart with league-average reference line
- Expandable data table + CSV download

### 🎯 Player Deep-Dive
- **Impact Gauge**: 0–100 semi-circle gauge with delta vs baseline
- **Innings Trend**: last-10 innings IM bars coloured by pressure level + 3-inn moving average
- **Batting Career Stats**: innings, runs, avg, SR, 4s, 6s
- **Bat vs Bowl Contribution**: grouped bar chart
- **Pressure Distribution**: donut chart (Low / Medium / High / Extreme)
- **Bowling Career Stats**: wickets, overs, economy, innings
- **Consistency Score**: last-10 IM bar chart with avg line + σ deviation
- **Team Contribution %**: horizontal stacked bar vs rest of league
- **Match Impact Timeline**: select any match → over-by-over WPA bars + cumulative line + boundary/wicket markers

### 🛡️ Team Impact Simulator
- Build **two custom XI teams** (4 Batters + 3 All-Rounders + 4 Bowlers)
- Validation: no duplicate players across teams
- **Team Strength Comparison**: total IM, batting strength, bowling strength, avg IM
- **Batting vs Bowling breakdown** per team
- **Top Performers Prediction**: best batter, overall MVP, best bowler cards
- **Squad Detail Tables** with per-player contribution stats

### ⚔️ Head-to-Head
- **Tab 1 — Overall Player Comparison**: dual gauges, IM comparison bar, radar chart (7 components), innings trend overlay
- **Tab 2 — Batter vs Bowler Matchup**: phase-wise breakdown (powerplay/middle/death), dominance logic, runs chart, SR vs Economy, radar, ball-by-ball flow, phase dominance summary

### 🧠 Smart Predictor
- **Match Situation Input**: batting/bowling perspective, match state (score, wickets, overs, target)
- **Pressure-Aware Logic**: separate pressure calculation for innings 1 vs innings 2 chasing scenarios
- **Role Eligibility Filtering**: only shows players suited for the situation
- **Pressure Fitness Scoring**: recommendation bar chart ranked by suitability
- **AI Match Intelligence**: detailed explanations for each recommended player
- **Player Detail Cards**: with sparkline trend data
- Minimum innings filter to exclude low-sample players

### 📐 Advanced Analytics
- **Opposition Strength Analysis**: scatter plot + opposition-adjusted table
- **Match Importance Distribution**: histogram + big-match performers table
- **Form Recovery**: bar chart + scatter of recovery scores
- **Phase Contribution**: stacked bar chart (PP/middle/death) + phase multiplier table
- **IM Component Breakdown**: bar chart + donut chart + anti-gaming explanation

---

## 12. Running the App

```bash
streamlit run ipl_impact_dashboard.py
```

Access at **http://localhost:8501**

To specify a port:
```bash
streamlit run ipl_impact_dashboard.py --server.port 8501 --server.headless true
```

---

## 13. Formulas Quick-Reference

| Metric | Formula |
|--------|---------|
| Strike Rate | `runs / balls_faced × 100` |
| Economy Rate | `runs_conceded / overs_bowled` |
| Boundary % | `boundary_runs / runs_scored` |
| Dot Ball % | `dot_balls / legal_balls` |
| Phase Multiplier | `Σ(phase_share × weight)` — PP=0.9, Mid=1.0, Death=1.2 |
| Pressure (inn 1) | `0.4×(wkts/10) + 0.6×stage_mult` |
| Pressure (inn 2) | `0.5×rrr_pressure + 0.3×(wkts/10) + 0.2×stage_mult` |
| WPA | `Σ Δwin_prob per player per innings` |
| WPA normalised | `1 / (1 + e^{-3×WPA})` |
| Clutch | `clip(high_pressure_avg / overall_avg, 0, 3) / 3` |
| Opposition (bat) | `clip(1 − avg_bowler_eco/12, 0, 1)` min-max normalised |
| Opposition (bowl) | `clip(avg_batter_sr/150, 0, 1)` min-max normalised |
| Match Importance | `clip((stage_weight − 1)/0.5 × 0.8 + 0.4, 0.4, 0.9)` |
| Form Recovery | `clip(z_score/4 + 0.5, 0, 1)` |
| Per-innings IM | `0.30P + 0.20PI + 0.20WPA + 0.10CF + 0.10Opp + 0.05MI + 0.05FR` |
| Rolling IM | Exponential decay (0.6) weighted avg over last 10 innings |
| Final IM (0–100) | `sigmoid(z × 1.5) × 100` |
| Consistency | `clip(1/(σ₁₀ + 0.01) / P95, 0, 1)` |

---

## UI Design

- **Theme**: Dark navy (`#050b18`) with cyan (`#00e5ff`) accent, gold (`#ffd700`) highlights
- **Fonts**: Orbitron (headings) + Inter (body) via Google Fonts
- **Icons**: 20+ custom inline SVG cricket icons (bat, ball, stumps, trophy, brain, shield, etc.)
- **Sidebar**: Glassmorphism with blur, gradient backgrounds, animated hover transitions
- **Cards**: Gradient stat cards with top-edge accent bars and hover lift effects

---

*Built for IPL Impact Analytics Hackathon 2026 · Powered by Cricsheet ball-by-ball data*
