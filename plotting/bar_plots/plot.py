import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 1. Load Data
df = pd.read_csv("new_bar.csv")

# 2. Setup Global Color Consistency
all_methods = sorted(df["method"].unique())
palette = dict(zip(all_methods, sns.color_palette("tab10", len(all_methods))))

# 3. Filter for 'Real Time'
df_rt = df[df["time_type"] == "Real Time"].copy()

# 4. Prepare Data for Plotting
# We need to separate the 'threshold' metric to transform it,
# then combine it back with the other metrics.

# Split the dataframe into actual metrics and the threshold rows
df_metrics = df_rt[df_rt["metric"] != "threshold"].copy()
df_thresh = df_rt[df_rt["metric"] == "threshold"].copy()


# Transform Threshold: result becomes (1 - result)
def process_threshold(val):
    if val == float("inf") or np.isinf(val):
        return 0.0
    return 1.0 - val


df_thresh["result"] = df_thresh["result"].apply(process_threshold)
df_thresh["metric"] = "1 - Threshold"

# Combine them back
plot_df = pd.concat([df_metrics, df_thresh]).reset_index(drop=True)

# Rename for your existing plotting logic
plot_df = plot_df.rename(
    columns={"metric": "Category", "result": "PlotValue", "step_setting": "setting"}
)

# 5. Generate Separate Plots for each (Dataset, Setting)
datasets = plot_df["dataset"].unique()
settings = plot_df["setting"].unique()

for ds in datasets:
    for st in settings:
        subset = plot_df[(plot_df["dataset"] == ds) & (plot_df["setting"] == st)].copy()

        if subset.empty:
            continue

        present_methods = [m for m in all_methods if m in subset["method"].unique()]

        # Define Category Order
        metrics_in_subset = [
            c for c in subset["Category"].unique() if "Threshold" not in c
        ]
        order = sorted(metrics_in_subset) + ["1 - Threshold"]

        subset["Category"] = pd.Categorical(
            subset["Category"], categories=order, ordered=True
        )
        subset = subset.sort_values("Category")

        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")

        ax = sns.barplot(
            data=subset,
            x="Category",
            y="PlotValue",
            hue="method",
            palette=palette,
            hue_order=present_methods,
        )

        plt.title(f"Dataset {ds} ({st})", fontsize=14, fontweight="bold")
        plt.ylabel("Value", fontsize=12)
        plt.xlabel("Metric", fontsize=12)
        plt.ylim(0, 1.1)

        plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        # Saving with the new 'st' (formerly step_setting)
        folder = "new/"
        file_name = f'bar_plot_{ds.replace(" ", "_")}_{str(st).replace(" ", "_")}.svg'
        os.makedirs(folder, exist_ok=True)
        plt.savefig(folder + file_name, format="svg", bbox_inches="tight")
        plt.close()
