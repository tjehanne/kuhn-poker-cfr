import pandas as pd
import numpy as np

df = pd.read_csv("data/premier_league_ready.csv")

# proba implicites des bookmakers
df["p_home_book"] = 1 / df["OddsHome"]
df["p_draw_book"] = 1 / df["OddsDraw"]
df["p_away_book"] = 1 / df["OddsAway"]

# Normalisation
s = df["p_home_book"] + df["p_draw_book"] + df["p_away_book"]
df[["p_home_book", "p_draw_book", "p_away_book"]] /= s.values[:, None]

#comparaison avec mon modèle
print(df[["p_home_book", "p_draw_book", "p_away_book"]].head())
