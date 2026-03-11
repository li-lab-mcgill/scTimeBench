import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
INPUT_FILE = "data_v2.csv"
OUTPUT_FILE = "split_filtered_heatmap_v2.svg"
METRICS_OF_INTEREST = ["AUC_PRC", "AUC_ROC", "JaccardSimilarity"]

# 1. Load and Rename
df = pd.read_csv(INPUT_FILE)


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

pivot_df["LFC"] = np.log2(
    pivot_df["Pseudotime"] / pivot_df["Real Time"] + np.finfo(float).eps
)  # Add small value to avoid log(0)

# 3. Filter Logic
pivot_df = pivot_df[
    (pivot_df["metric"].isin(METRICS_OF_INTEREST)) & (pivot_df["prc_threshold"] == True)
]

# 4. Get unique step_settings to determine number of subplots
# ... (previous code)

# 4. Get unique step_settings
step_settings = pivot_df["step_setting"].unique()
fig, axes = plt.subplots(len(step_settings), 1, figsize=(14, 8 * len(step_settings)))

if len(step_settings) == 1:
    axes = [axes]

# STEEPER GRADIENT LOGIC:
# We cap the visual range at 1.5. This means anything > 1.5x (or < -1.5x)
# is fully saturated, making the "middle" transitions much sharper.
VISUAL_LIMIT = 1.5
clean_vals = pivot_df.replace([np.inf, -np.inf], np.nan)
actual_max = clean_vals["LFC"].abs().max()
v_limit = min(actual_max, VISUAL_LIMIT) if not pd.isna(actual_max) else 1
max_abs = actual_max

# 5. Plotting Loop
for i, setting in enumerate(step_settings):
    ax = axes[i]
    # Filter data for this specific setting
    setting_df = pivot_df[pivot_df["step_setting"] == setting]

    # --- SWAPPED INDEX AND COLUMNS HERE ---
    # Rows (index) are now methods
    # Columns are now the multi-index of dataset and metric
    subplot_pivot = setting_df.pivot_table(
        index="method", columns=["dataset", "metric"], values="LFC"
    )

    # Set grey background for NaNs
    ax.set_facecolor("#E0E0E0")

    from matplotlib.colors import TwoSlopeNorm

    # Recalculate norm for this specific plot or use global
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

    # Draw Heatmap
    sns.heatmap(
        subplot_pivot,
        annot=False,
        cmap="seismic",  # Or "RdBu_r" based on your preference
        norm=norm,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Log2 Fold Change"} if i == 0 else None,
        ax=ax,
    )

    # 6. MANUALLY DRAW SLASHES (Adjusted for new shape)
    rows, cols = subplot_pivot.shape
    for y in range(rows):
        for x in range(cols):
            val = subplot_pivot.iloc[y, x]
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

    ax.set_title(f"Step Setting: {setting}", fontsize=16, fontweight="bold")
    ax.set_xlabel("Dataset / Metric")  # Changed from ylabel
    ax.set_ylabel("Method")  # Changed from "Dataset / Metric"

plt.tight_layout()
plt.savefig(OUTPUT_FILE, format="svg")
# plt.show()
