import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

df = pd.read_csv("full.csv")
# 1. Configuration & Skip List
# Add any metric names here that you want to exclude from the plot
metrics_to_skip = ["threshold", "PRECISION", "RECALL"]

# Markers for Dataset | Method combinations
markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]
# Colormap for Step Setting | Time Type combinations
cmap = plt.get_cmap("tab10")

# 2. Preprocessing
# Ensure result is numeric and filter out skipped metrics immediately
df["result"] = pd.to_numeric(df["result"], errors="coerce")
df_filtered = df[~df["metric"].isin(metrics_to_skip)].copy()

# 3. Pivot/Merge to get X (False) and Y (True) coordinates
df_false = df_filtered[df_filtered["prc_threshold"] == False].rename(
    columns={"result": "x"}
)
df_true = df_filtered[df_filtered["prc_threshold"] == True].rename(
    columns={"result": "y"}
)

# Merge on all identifying columns
merge_cols = ["method", "dataset", "step_setting", "metric", "time_type"]
merged_df = pd.merge(df_false, df_true, on=merge_cols).dropna(subset=["x", "y"])

# 4. Create Encoding Keys
# Shape = Dataset + Method
merged_df["shape_key"] = merged_df["dataset"] + " | " + merged_df["method"]
unique_shapes = sorted(merged_df["shape_key"].unique())
shape_map = {key: markers[i % len(markers)] for i, key in enumerate(unique_shapes)}

# Color = Step Setting + Time Type
merged_df["color_key"] = merged_df["step_setting"] + " | " + merged_df["time_type"]
unique_colors = sorted(merged_df["color_key"].unique())
color_map = {
    key: cmap(i / max(1, len(unique_colors) - 1)) for i, key in enumerate(unique_colors)
}

# 5. Plotting
fig, ax = plt.subplots(figsize=(12, 8))

# Plot each data point with its specific encoding
for _, row in merged_df.iterrows():
    ax.scatter(
        row["x"],
        row["y"],
        marker=shape_map[row["shape_key"]],
        color=color_map[row["color_key"]],
        s=160,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.7,
        label="_nolegend_",  # We build custom legends below
    )

# Draw the y = x reference line
all_vals = pd.concat([merged_df["x"], merged_df["y"]])
line_lims = [all_vals.min() * 0.95, all_vals.max() * 1.05]
ax.plot(line_lims, line_lims, "k--", alpha=0.3, zorder=0)

# 6. Build Dual Legends
# Legend 1: Shapes (Dataset & Method)
shape_elements = [
    mlines.Line2D(
        [],
        [],
        color="gray",
        marker=shape_map[k],
        linestyle="None",
        markersize=10,
        label=k,
    )
    for k in shape_map
]
leg1 = ax.legend(
    handles=shape_elements,
    title="Dataset | Method",
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
)

# Legend 2: Colors (Step Setting | Time Type)
color_elements = [
    mlines.Line2D(
        [], [], color=color_map[k], marker="o", linestyle="None", markersize=10, label=k
    )
    for k in color_map
]
leg2 = ax.legend(
    handles=color_elements,
    title="Step Setting | Time Type",
    loc="lower left",
    bbox_to_anchor=(1.02, 0),
)

# Add the first legend back manually so it's not hidden by the second
ax.add_artist(leg1)

# Formatting
ax.set_xlabel("AUROC threshold metric result")
ax.set_ylabel("AUPRC threshold metric result")
ax.set_title(f'Performance Comparison (Excluding: {", ".join(metrics_to_skip)})')
plt.grid(True, linestyle=":", alpha=0.4)
plt.tight_layout()

plt.savefig("prc_vs_auroc_plot.png", dpi=300, bbox_inches="tight")
plt.show()
