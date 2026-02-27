import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Data
df = pd.read_csv("scatter.csv")

# 2. Reshape Data
# Filter for Real Time and the metrics we care about
metrics_of_interest = ["threshold", "PRECISION", "RECALL"]
df_filtered = df[
    (df["time_type"] == "Real Time") & (df["metric"].isin(metrics_of_interest))
]

# Pivot so 'threshold', 'PRECISION', and 'RECALL' become columns
# Index includes method, dataset, and step_setting to keep unique experiments together
df_pivot = df_filtered.pivot(
    index=["method", "dataset", "step_setting"], columns="metric", values="result"
).reset_index()

# 3. Create the Visualization
sns.set_style("whitegrid")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Threshold vs Precision
sns.scatterplot(
    data=df_pivot,
    x="threshold",
    y="PRECISION",
    hue="method",
    style="dataset",
    s=100,
    ax=ax1,
)
ax1.set_title("Relationship: Threshold vs Precision", fontweight="bold")
ax1.set_ylim(-0.05, 1.05)

# Plot 2: Threshold vs Recall
sns.scatterplot(
    data=df_pivot,
    x="threshold",
    y="RECALL",
    hue="method",
    style="dataset",
    s=100,
    ax=ax2,
)
ax2.set_title("Relationship: Threshold vs Recall", fontweight="bold")
ax2.set_ylim(-0.05, 1.05)

# Improve Legend Handling
ax1.get_legend().remove()
plt.legend(title="Method / Dataset", bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()
plt.savefig("threshold_relationships.png", dpi=300)
plt.close()
