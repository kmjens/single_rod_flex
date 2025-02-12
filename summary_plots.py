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

from figs_utility import PlottingFxns

PLOT_HEATMAPS = False
PLOT_LINES = False

def get_user_choice(prompt, default=False):
    """
    Prompt the user for a Yes/No choice, returning True for Yes and False for No.
    Default value is used if the user provides no input.
    """
    choice = input(f"{prompt} (y/n, default {'y' if default else 'n'}): ").strip().lower()
    if choice == 'y':
        return True
    elif choice == 'n':
        return False
    else:
        return default

if __name__ == "__main__":

    # Ask the user to toggle PLOT_HEATMAPS and PLOT_LINES
    PLOT_HEATMAPS = get_user_choice("Do you want to plot heatmaps?", default=False)
    PLOT_LINES = get_user_choice("Do you want to plot line plots?", default=False)

    # Display the choices
    print(f"\nOptions selected:")
    print(f"  Plot heatmaps: {'Yes' if PLOT_HEATMAPS else 'No'}")
    print(f"  Plot line plots: {'Yes' if PLOT_LINES else 'No'}\n")

    plotter = PlottingFxns()
    plotter.aggregate_results()
    plotter.collect_values()

    if PLOT_HEATMAPS:
        plotter.plot_heatmaps()
    if PLOT_LINES:
        plotter.plot_lines()
