import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Data
# csv_data = """method,dataset,setting,metric,time_type,result,threshold
# Cooccurrence,Suo,All Paths,F1,Real Time,0.549,0.161
# Cooccurrence,Suo,All Paths,Accuracy,Real Time,0.661,0.161
# Cooccurrence,Suo,All Paths,AUROC,Real Time,0.369,0.161
# Cooccurrence,Suo,All Paths,Jaccard,Real Time,0.379,0.161
# Moscot,Suo,Simple,F1,Real Time,0.311,0.028
# Moscot,Suo,Simple,Accuracy,Real Time,0.744,0.028
# Moscot,Suo,Simple,AUROC,Real Time,0.677,0.028
# Moscot,Suo,Simple,Jaccard,Real Time,0.184,0.028"""

# df = pd.read_csv(StringIO(csv_data))
df = pd.read_csv("bar.csv")

# 2. Setup Global Color Consistency
all_methods = sorted(df["method"].unique())
palette = dict(zip(all_methods, sns.color_palette("tab10", len(all_methods))))

# 3. Filter for 'Real Time' and Process Thresholds
df_rt = df[df["time_type"] == "Real Time"].copy()


def process_threshold(val):
    if val == float("inf") or np.isinf(val):
        return 0.0
    return 1.0 - val


df_rt["adj_threshold"] = df_rt["threshold"].apply(process_threshold)

# 4. Prepare Data for Plotting
# Create Metric rows
df_metrics = df_rt.copy()
df_metrics["Category"] = df_metrics["metric"]
df_metrics["PlotValue"] = df_metrics["result"]

# Create Threshold rows
df_thresh = (
    df_rt[["method", "dataset", "setting", "adj_threshold"]].drop_duplicates().copy()
)
df_thresh["Category"] = "1 - Threshold"
df_thresh["PlotValue"] = df_thresh["adj_threshold"]

plot_df = pd.concat(
    [
        df_metrics[["method", "dataset", "setting", "Category", "PlotValue"]],
        df_thresh[["method", "dataset", "setting", "Category", "PlotValue"]],
    ]
).reset_index(drop=True)

# 5. Generate Separate Plots for each (Dataset, Setting)
datasets = plot_df["dataset"].unique()
settings = plot_df["setting"].unique()


# ... (Previous data processing remains the same)

for ds in datasets:
    for st in settings:
        # Subset data for this specific plot
        subset = plot_df[(plot_df["dataset"] == ds) & (plot_df["setting"] == st)].copy()

        # Omit the plot if it's entirely empty
        if subset.empty:
            continue

        # 1. Identify which methods actually have data in this subset
        # This prevents "gaps" for failed methods
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

        # 2. Use 'present_methods' for hue_order to collapse gaps
        ax = sns.barplot(
            data=subset,
            x="Category",
            y="PlotValue",
            hue="method",
            palette=palette,  # Colors stay mapped to specific methods
            hue_order=present_methods,  # This removes the empty gaps
        )

        plt.title(f"Dataset {ds} ({st})", fontsize=14, fontweight="bold")
        plt.ylabel("Value", fontsize=12)
        plt.xlabel("Metric", fontsize=12)
        plt.ylim(0, 1.1)

        # Move legend outside so it doesn't overlap bars
        plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(
            f'bar_plot_{ds.replace(" ", "_")}_{st.replace(" ", "_")}.svg',
            format="svg",
            bbox_inches="tight",
        )
        plt.close()
        # plt.show()
