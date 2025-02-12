import json
import matplotlib
import os
import signac

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from itertools import combinations
from collections import defaultdict

class PlottingFxns:
    '''
    __init__: fill in the analysis results that you care about plotting in data_values     
    keys_of_interest: will automatically be propogated with any signac SP that has multiple values except for seed.
    aggregate_results: will collect these into a data frame self.df and print this in a CSV file
    collect_values: organizes the data into a dictionary by which SPs are keys of interest
    
    '''
    def __init__(self):
        self.project = None
        self.key_of_interest = []
        self.data_values = ["MSD_correlation", 'mesh_len_ratio_xz', "ellipsoidal_volume", "sphericity", "eccentricity"]
        self.df = None


    def aggregate_results(self):
        print("Aggregating results...")
        all_results = []

        self.project = signac.get_project()
        for job in self.project:
            try:
                with open(job.fn('signac_job_document.json'), 'r') as f:
                    data = json.load(f)
                    all_results.append(data)
            except FileNotFoundError:
                print(f"Missing analysis_data.json for job: {job.id}")

        self.df = pd.DataFrame(all_results)
        self.df.to_csv("aggregated_results.csv", index=False)


    def collect_values(self):
        print('Collecting statepoint values...')

        statepoint_values = defaultdict(set)
        for job in self.project:
            for key, value in job.sp.items():
                if key != "seed":
                    statepoint_values[key].add(value)

        # Determine keys of interest (those with multiple values)
        self.keys_of_interest = [key for key, values in statepoint_values.items() if len(values) > 1]

        assert all(key in self.df.columns for key in self.keys_of_interest), "Some keys are missing in the DataFrame."
        assert all(data_value in self.df.columns for data_value in self.data_values), "Data value column is missing in the DataFrame."


    def plot_heatmaps(self):
        print('Begin plotting heatmaps...')

        if not self.keys_of_interest:
            raise ValueError("keys_of_interest is empty. Did collect_values() fail?")

        print("Keys of Interest before plotting:", self.keys_of_interest)  # Debug print


        # Make directories as needed
        os.makedirs('plots/heatmaps', exist_ok=True)
        for data_value in self.data_values:
            os.makedirs(f"plots/{data_value}", exist_ok=True)
            for i, y_key in enumerate(self.keys_of_interest):
                for x_key in self.keys_of_interest[i + 1:]:
                    os.makedirs(f"plots/{data_value}/{x_key}_V_{y_key}", exist_ok=True)

        # Plot for every key of interest for every combo of other keys
        for i, y_key in enumerate(self.keys_of_interest):
            for x_key in self.keys_of_interest[i + 1:]:

                print('Producing heat maps for: ', x_key,' VS ',y_key,'...')
                other_keys = [key for key in self.keys_of_interest if key not in (y_key, x_key)]

                if not other_keys:
                    print(f"No valid keys left for grouping! x_key={x_key}, y_key={y_key}, keys_of_interest={self.keys_of_interest}")
                    # Use the entire DataFrame instead of grouping
                    group_df = self.df
                    group_title = "Full dataset"
                    group_label = "full_dataset"

                    for data_value in self.data_values:
                        num_flattener_dir = f"plots/{data_value}/{x_key}_V_{y_key}"
                        os.makedirs(num_flattener_dir, exist_ok=True)

                        pivot_table = group_df.pivot_table(index=y_key, columns=x_key, values=data_value)

                        if not pivot_table.empty:
                            plt.figure(figsize=(10, 8))
                            sns.heatmap(pivot_table, annot=True, fmt=".2f",
                                        cmap="viridis", cbar_kws={'label': data_value},
                                        vmin=-1, vmax=1)
                            plt.title(f"{data_value} as a function of {x_key} and {y_key}\n({group_title})")
                            plt.xlabel(x_key)
                            plt.ylabel(y_key)
                            plt.savefig(f"{num_flattener_dir}/heatmap_{group_label}.png")
                            plt.close()
                            print(f"SUCCESS: {group_title}...")
                        else:
                            print(f"NAN For: {group_title}...")


                else:
                    grouped = self.df.groupby(other_keys)

                    for group_values, group_df in grouped:
                        group_dict = dict(zip(other_keys, group_values))

                        num_flattener = group_dict.pop('num_flattener', None)
                        group_title = ", ".join([f"{key}={value}" for key, value in group_dict.items()])
                        group_label = "_".join([f"{key}={value}" for key, value in group_dict.items()])


                        for data_value in self.data_values:
                            if num_flattener is not None:
                                num_flattener_dir = f"plots/{data_value}/{x_key}_V_{y_key}/num_flattener_{num_flattener}"
                                if not os.path.exists(num_flattener_dir):
                                    os.makedirs(num_flattener_dir)
                            else:
                                num_flattener_dir = f"plots/{data_value}/{x_key}_V_{y_key}"
                                if not os.path.exists(num_flattener_dir):
                                    os.makedirs(num_flattener_dir)

                            pivot_table = group_df.pivot_table(index=y_key, columns=x_key, values=data_value)
                            if not pivot_table.empty:
                                plt.figure(figsize=(10, 8))
                                sns.heatmap(pivot_table, annot=True, fmt=".2f",
                                            cmap="viridis", cbar_kws={'label': data_value},
                                            vmin=-1, vmax=1)
                                plt.title(f"{data_value} as a function of {x_key} and {y_key}\n({group_title})")
                                plt.xlabel(x_key)
                                plt.ylabel(y_key)
                                plt.savefig(f"{num_flattener_dir}/heatmap_{group_label}.png")
                                plt.close()
                                print(f"SUCCESS: {group_title}...")
                            else:
                                print(f"NAN For: {group_title}...")

                print('\n')
        print('Heatmap plotting function completed.')


    def plot_lines(self):
        print('Begin plotting line plots...')
        '''
        # Make directories as needed
        os.makedirs('plots/linear', exist_ok=True)
        for data_value in self.data_values:
            os.makedirs(f"plots/{data_value}", exist_ok=True)
            for i, y_key in enumerate(self.keys_of_interest):
                for x_key in self.keys_of_interest[i + 1:]:
                    os.makedirs(f"plots/linear/{data_value}/{x_key}_V_{y_key}", exist_ok=True)
        '''

