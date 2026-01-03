import pickle
import pandas as pd
import json
import numpy as np

with open("data/fit.pkl", "rb") as f:
    fit = pickle.load(f)

with open("data/team_mapping.json") as f:
    team2id = json.load(f)

id2team = {v: k for k, v in team2id.items()}

posterior = fit.draws_pd()

attack_cols = [c for c in posterior.columns if "attack[" in c]
defense_cols = [c for c in posterior.columns if "defense[" in c]

attack_mean = posterior[attack_cols].mean().values
defense_mean = posterior[defense_cols].mean().values

ranking = pd.DataFrame({
    "Team": [id2team[i + 1] for i in range(len(attack_mean))],
    "Attack": attack_mean,
    "Defense": defense_mean
}).sort_values("Attack", ascending=False)

print("="*40)
print("Team ranking by attack/defense strength:")
print(ranking.head(10))
print()
print("="*40)
print("Home advantage home_adv (log-scale):")

home_adv = posterior["home_adv"]

print("mean :", home_adv.mean())
print("std  :", home_adv.std())
print("95% CI :", np.percentile(home_adv, [2.5, 97.5]))
print("exp(mean) :", np.exp(home_adv.mean()))
