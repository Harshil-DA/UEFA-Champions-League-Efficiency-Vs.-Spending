# UEFA-Champions-League-Efficiency-Vs.-Spending

# Does Money Buy a Better Attack? Transfer Spending vs. Champions League Scoring

An end-to-end data analysis of 15 UEFA Champions League seasons (2011-12 to 2025-26). It combines match results with transfer-market data to test whether clubs that spend more build better attacks, and which clubs beat or miss what their money predicts.

**Author:** Harshil Verma | **Tools:** Python, pandas, NumPy, matplotlib, statsmodels, scikit-learn

---

## Key findings

1. **Squad value predicts scoring, but only partly.** A squad's market value alone explains about 30% of the variation in goals per match (R² = 0.30, correlation about 0.55). Doubling a squad's value goes with roughly +0.27 goals per match.
2. **Extra transfer spending adds almost nothing once squad value is known.** Three-year net spend is not statistically significant (p = 0.08, and p = 0.29 once season effects are added). Its 95% confidence interval (-0.006 to +0.092 goals per match per €100m) includes zero.
3. **The result holds under different set-ups.** Weighting teams by matches played, or keeping only teams with 8+ matches, leaves squad value significant (p < 0.001) and net spend not significant (p = 0.15 to 0.17).
4. **The model beats a naive guess on unseen seasons.** Trained on 2011-12 to 2021-22 and tested on 2022-23 to 2025-26, its average error was 0.42 goals per match against 0.50 for guessing the average, and its test R² was 0.29.
5. **Some clubs consistently beat their money.** Over 5+ seasons, Bayern München (+0.73 goals per match above prediction), Paris Saint-Germain, Real Madrid, Ajax and Barcelona score more than their squad value predicts. Marseille, AC Milan, Inter, AS Monaco and Atlético Madrid score less.
6. **Five club types emerge from clustering,** and the heaviest spenders are not the best scorers (table below).
7. **The Champions League has become more attacking:** total goals per match rose from 2.76 in 2011-12 to 3.47 in 2025-26 (extra-time goals included).

![Squad value vs goals per match](charts/eda/2_value_vs_goals.png)

### Five types of club (K-means clustering)

| Type | Club-seasons | Avg squad value | Avg 3-yr net spend | Goals per match | Typical clubs |
|---|---|---|---|---|---|
| Mega-spenders | 27 | €862m | +€414m | 1.93 | Man City, PSG, Man United, Chelsea |
| Elite scorers | 74 | €664m | +€112m | 2.35 | Bayern, Barcelona, Real Madrid, Liverpool |
| Mid-to-big squads, quiet attack | 85 | €434m | +€85m | 1.30 | Juventus, Atlético, Napoli, Inter |
| Selling clubs, solid attack | 89 | €279m | -€81m | 1.64 | Benfica, Porto, Dortmund, Ajax |
| Small budgets, few goals | 159 | €114m | -€14m | 0.95 | Shakhtar, CSKA, Celtic, Club Brugge |

![Cluster averages](charts/clusters/2_cluster_averages.png)

![Clubs that beat or miss expectations](charts/regression/B_club_over_under.png)

---

## Question and approach

**Question:** Do clubs that spend more on transfers have better attacks in the Champions League?

**Measures**
- *Attack output:* goals scored per match (all stages, extra time included, penalty shootouts excluded).
- *Spending:* net transfer spend (fees paid minus fees received) in the same season and over a rolling 3 seasons, and squad market value on 1 September.

**Pipeline**

1. **Collect:** Champions League results (openfootball, plain-text files) and transfers, valuations and matches (Transfermarkt dataset).
2. **Parse and clean:** turned about 2,000 text lines into a match table with regular expressions, then standardized the different spellings of club names into 108 clubs.
3. **Validate:** compared the goals and matches of 456 club-seasons against the Transfermarkt match data (see "Data validation").
4. **Engineer features:** spend, income, net spend, 3-year rolling spend, and squad value built from player valuations.
5. **Explore (EDA):** distributions, scatter plots, rankings and a scoring trend.
6. **Model:** OLS regression with club-clustered standard errors, robustness checks, a train/test split on later seasons, and residual analysis.
7. **Cluster:** K-means on standardized squad value, net spend and goals per match.

---

## Data

| Source | Used for | License |
|---|---|---|
| [openfootball/champions-league](https://github.com/openfootball/champions-league) | Match results 2011-12 to 2025-26 (included in `data/raw/`) | CC0 (public domain) |
| [Transfermarkt dataset](https://github.com/dcaribou/transfermarkt-datasets) (Kaggle: `davidcariboo/player-scores`) | Transfers, player valuations, match cross-check | See the dataset's own terms |

Every column of the cleaned tables is explained in the [data dictionary](docs/data_dictionary.md).

The Transfermarkt files are not in this repository because of their size. Download `games.csv`, `club_games.csv`, `transfers.csv` and `player_valuations.csv` into `data/raw/transfermarkt/`.

### Data validation (cross-check)

Goals and matches per club-season were compared between the two sources (456 club-seasons, 2012-13 to 2025-26):

- **439** match exactly.
- **14** match once penalty-shootout goals are added. Transfermarkt counts shootout goals in scores and this project does not.
- **2** differ because Transfermarkt is missing the 2025-26 final.
- **1** (Arsenal, 2023-24) differs by one goal in a shootout and is unexplained.

Full results are in `data/clean/crossmatch_validation.csv`.

### Data decisions

- **Penalty shootouts are excluded** from goals scored, because they are not open-play attacking output.
- **Fees:** only disclosed fees above zero count as spend. Free transfers and loans show as 0 and undisclosed fees are blank (counted in `fee_unknown_signings_count`).
- **Squad value** is the sum of each player's latest valuation in the 12 months before 1 September, assigned to the club on that valuation record.
- **Season alignment:** Transfermarkt season "23/24" (summer 2023 plus January 2024 windows) is matched to Champions League 2023-24.
- **Quality filter:** club-seasons with fewer than 15 valued players were dropped (54 of 488), because their squad values are far too low. For example, BATE Borisov 2012-13 had only 2 valued players. This leaves 434 club-seasons from 88 clubs.

---

## Limitations

- **This measures scoring output, not true efficiency.** Shot data (and therefore conversion rate and xG) was not available for all seasons, so goals per match is the outcome.
- **Small samples:** many clubs play only 6 matches, so their goals-per-match figures are noisy. Several of the largest "under-performers" are single group stages, where a few goals decide the result.
- **Linear fit under-predicts the elite:** the biggest clubs often sit above the line, so the true relationship may be steeper at the top, or Transfermarkt values may understate them.
- **Squad value is an estimate** and valuation coverage varies by club and season.
- **Correlation is not causation:** wealthy clubs may also have better coaching and recruitment.
- **Playing style is not modeled.** Defensive-minded teams may score less regardless of money. This is a hypothesis, not a tested result.
- **Clusters overlap:** the silhouette score is 0.32, so club types form a continuum. Groups were stable across 10 random restarts (agreement 0.99), and the names are my own labels.

---

## How to reproduce

```bash
git clone https://github.com/<your-username>/ucl-spending-vs-efficiency.git
cd ucl-spending-vs-efficiency
pip install -r requirements.txt

# 1. Put the 4 Transfermarkt CSV files in data/raw/transfermarkt/

python src/01_build_ucl_tables.py        # parse results, clean names, build tables
python src/02_merge_transfermarkt.py     # match clubs, validate, add spending features
python analysis/eda.py                   # exploratory charts
python analysis/regression.py            # models, robustness, train/test, residuals
python analysis/clustering.py            # K-means club types
```

## Project structure

```
data/raw/            original files (never edited)
data/clean/          cleaned and merged tables (main file: ucl_team_seasons_enriched.csv)
src/                 data preparation scripts
analysis/            EDA, regression and clustering scripts
charts/              figures
results/             model residuals and cluster assignments
```

## Possible next steps

- Add shots and xG for recent seasons to measure true efficiency (conversion rate).
- Model non-linear effects, or add playing-style and coach variables.
- Build an interactive dashboard (Streamlit or Plotly) for exploring any club or season.

## Skills demonstrated

- **Data wrangling:** regex parsing of semi-structured text, standardizing entities across sources, fuzzy matching with manual correction, joins and rolling-window features in pandas
- **Data validation:** cross-checking two independent sources and documenting every discrepancy
- **Statistics:** OLS regression with club-clustered standard errors, robustness checks, a time-based train/test split, and residual analysis
- **Machine learning:** K-means clustering with feature scaling, silhouette-based model selection and a stability check
- **Communication:** reproducible scripts, charts, and a README that states limitations honestly

## License

The code is released under the [MIT License](LICENSE). The data keeps the licenses of its sources (see the Data section above).

## Credits

Match data from the [openfootball](https://github.com/openfootball) project (CC0). Transfer and valuation data from the [Transfermarkt datasets](https://github.com/dcaribou/transfermarkt-datasets) project, which compiles data from transfermarkt.com.
