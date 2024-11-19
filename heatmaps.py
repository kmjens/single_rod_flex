import json
import matplotlib
import os
import signac

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from collections import defaultdict

all_results = []

project = signac.get_project()
for job in project:
    try:
        with open(job.fn('analysis_data.json'), 'r') as f:
            data = json.load(f)
            all_results.append(data)
    except FileNotFoundError:
        print(f"Missing analysis_data.json for job: {job.id}")

df = pd.DataFrame(all_results)
df.to_csv("aggregated_results.csv", index=False)

statepoint_values = defaultdict(set)
for job in project:
    for key, value in job.sp.items():
        if key != "seed":
            statepoint_values[key].add(value)

keys_of_interest = []
for key, values in statepoint_values.items():
    if len(values) > 1:
        keys_of_interest.append(key)

data_values = ["MSD_correlation", "eccentricity"]
df = pd.read_csv("aggregated_results.csv")

if not os.path.exists('heatmaps'):
    os.makedirs('heatmaps')

# Iterate through pairs of keys of interest
for i, key1 in enumerate(keys_of_interest):
    for key2 in keys_of_interest[i + 1:]:
        for data_value in data_values:
            pivot_table = df.pivot_table(
                index=key1, columns=key2, values=data_value, aggfunc="mean")
            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot_table, annot=True, cmap="viridis", cbar_kws={'label': data_value})
            plt.title(f"{data_value} as a function of {key1} and {key2}")
            plt.xlabel(key2)
            plt.ylabel(key1)
            plt.savefig(f"heatmaps/heatmap_{data_value}_{key1}_vs_{key2}.png")
            plt.close()
