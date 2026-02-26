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

# 2. Filter for 'Real Time'
df_rt = df[df["time_type"] == "Real Time"].copy()


# 3. Handle Threshold Logic: (1 - threshold), and if inf then 0
def process_threshold(val):
    if val == float("inf") or np.isinf(val):
        return 0.0
    return 1.0 - val


df_rt["adj_threshold"] = df_rt["threshold"].apply(process_threshold)

# 4. Prepare Metric Bars
df_metrics = df_rt.copy()
df_metrics["Category"] = df_metrics["setting"] + " " + df_metrics["metric"]
df_metrics["PlotValue"] = df_metrics["result"]

# 5. Prepare Threshold Bars (One per setting per method)
df_thresh = (
    df_rt[["method", "dataset", "setting", "adj_threshold"]].drop_duplicates().copy()
)
df_thresh["Category"] = df_thresh["setting"] + " Threshold (1-thr)"
df_thresh["PlotValue"] = df_thresh["adj_threshold"]

# 6. Combine and Reset Index (To avoid duplicate label error)
plot_df = pd.concat(
    [
        df_metrics[["method", "dataset", "Category", "PlotValue"]],
        df_thresh[["method", "dataset", "Category", "PlotValue"]],
    ]
).reset_index(drop=True)

# 7. Plotting
datasets = plot_df["dataset"].unique()

for ds in datasets:
    ds_data = plot_df[plot_df["dataset"] == ds].copy()

    # Define a clean order for the x-axis
    order = sorted(ds_data["Category"].unique(), key=lambda x: ("Threshold" in x, x))
    ds_data["Category"] = pd.Categorical(
        ds_data["Category"], categories=order, ordered=True
    )
    ds_data = ds_data.sort_values("Category")

    plt.figure(figsize=(14, 7))
    sns.set_style("whitegrid")

    ax = sns.barplot(
        data=ds_data, x="Category", y="PlotValue", hue="method", palette="muted"
    )

    plt.title(
        f"Comprehensive Metrics for Dataset: {ds}", fontsize=16, fontweight="bold"
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Value", fontsize=12)
    plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    plt.savefig(f"bar_plot_{ds}.svg", format="svg", bbox_inches="tight")
    plt.close()
