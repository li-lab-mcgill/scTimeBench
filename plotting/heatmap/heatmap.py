import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
INPUT_FILE = "heatmap.csv"
OUTPUT_FILE = "final_filtered_heatmap.svg"

# Specify exactly which metrics you want to see
METRICS_OF_INTEREST = ["AUC_PRC", "AUC_ROC", "JaccardSimilarity"]

# 1. Load and Rename
df = pd.read_csv(INPUT_FILE)


# Renaming logic for 3x groups
def rename_3x_groups(row):
    if "3x" in str(row["dataset"]):
        return f"{row['method']}-3x"
    return row["method"]


df["method"] = df.apply(rename_3x_groups, axis=1)

# 2. Pivot & Calculate LFC
pivot_df = df.pivot_table(
    index=["dataset", "step_setting", "metric", "method", "prc_threshold"],
    columns="time_type",
    values="result",
).reset_index()

# Handle potential zeros before log2
pivot_df["LFC"] = np.log2(pivot_df["Pseudotime"] / pivot_df["Real Time"])

# 3. SUBSET LOGIC: Filter for specific metrics and path settings
# This ensures we only keep the rows you explicitly care about
pivot_df = pivot_df[
    (pivot_df["metric"].isin(METRICS_OF_INTEREST))
    & (
        pivot_df["prc_threshold"]
        == True
        # ((pivot_df["step_setting"] == "all_paths") & (pivot_df["prc_threshold"] == True)) |
        # ((pivot_df["step_setting"] != "all_paths") & (pivot_df["prc_threshold"] == False))
    )
]

# 4. Final Pivot
final_pivot = pivot_df.pivot_table(
    index=["dataset", "step_setting", "metric"], columns="method", values="LFC"
)

# 5. Symmetrical Scaling
clean_vals = final_pivot.replace([np.inf, -np.inf], np.nan)
max_abs = clean_vals.abs().max().max()
if pd.isna(max_abs) or max_abs == 0:
    max_abs = 1

# 6. Plotting
plt.figure(figsize=(14, 10))
ax = plt.gca()
ax.set_facecolor("#E0E0E0")  # Grey background for empty cells

sns.heatmap(
    final_pivot,
    annot=False,
    cmap="RdBu_r",
    center=0,
    vmin=-max_abs,
    vmax=max_abs,
    square=True,
    linewidths=0.5,
    cbar_kws={"label": "Log2 Fold Change"},
    ax=ax,
)

# 7. MANUALLY DRAW SLASHES
rows, cols = final_pivot.shape
for y in range(rows):
    for x in range(cols):
        val = final_pivot.iloc[y, x]
        if pd.isna(val) or np.isinf(val):
            ax.text(
                x + 0.5,
                y + 0.5,
                "/",
                ha="center",
                va="center",
                color="black",
                fontsize=14,
                fontweight="bold",
            )

plt.title(f"LFC Heatmap: {', '.join(METRICS_OF_INTEREST)}")
plt.tight_layout()
plt.savefig(OUTPUT_FILE, format="svg")
# plt.show()
