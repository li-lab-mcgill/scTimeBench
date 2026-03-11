import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# Load data
df = pd.read_csv("data_v3.csv")

# 1. Filter Data
metrics_to_plot = ["JaccardSimilarity", "AUC_PRC", "AUC_ROC"]
plot_df = df[
    (df["prc_threshold"] == True)
    & (df["time_type"] == "Real Time")
    & (df["metric"].isin(metrics_to_plot))
].copy()

# Sort and prepare combined labels
plot_df["metric"] = pd.Categorical(
    plot_df["metric"], categories=metrics_to_plot, ordered=True
)
plot_df = plot_df.sort_values(["metric", "dataset"])

dataset_order = list(dict.fromkeys(plot_df["dataset"]))


# 2. Setup the Unified Palette and Explicit Order
unique_methods = plot_df["method"].unique()
# special_methods = ["Correlation", "Random"]
special_methods = ["Correlation"]
other_methods = [m for m in unique_methods if m not in special_methods]

# Force Correlation and Random to the front of the list
ordered_methods = special_methods + other_methods

# Get enough colors from Set2 for the non-highlighted methods
set2_colors = sns.color_palette("Set2", len(other_methods))

# Build the combined palette dictionary using the ordered list
custom_palette = {}
color_index = 0

for method in ordered_methods:
    if method == "Correlation":
        custom_palette[method] = "#E63946"
    else:
        custom_palette[method] = set2_colors[color_index]
        color_index += 1

all_methods = (
    ordered_methods  # This is now in the desired order for plotting and legend
)

# 3. Setup the Grid
g = sns.FacetGrid(
    plot_df,
    row="step_setting",
    col="metric",
    col_order=metrics_to_plot,
    height=3.8,
    aspect=0.78,
    margin_titles=True,
    sharex=True,
    sharey=True,
)


# 4. Define Plotting Function
def layered_swarm(data, **kwargs):
    ax = plt.gca()

    # Now we just plot everything in one go using the unified palette
    sns.swarmplot(
        data=data,
        x="dataset",
        y="result",
        hue="method",
        order=dataset_order,
        palette=custom_palette,
        size=5,
        dodge=False,
        ax=ax,
        alpha=1.0,
    )

    # Keep seaborn's jitter placement, then replace Correlation circles with diamonds.
    correlation_color = np.array(mcolors.to_rgba(custom_palette["Correlation"]))
    for collection in ax.collections:
        facecolors = collection.get_facecolors()
        if facecolors is None or len(facecolors) == 0:
            continue

        # seaborn puts multiple method colors in the same PathCollection,
        # so we mask per point (not per collection).
        rgb_close = np.isclose(facecolors[:, :3], correlation_color[:3], atol=1e-3)
        corr_mask = np.all(rgb_close, axis=1)
        if not np.any(corr_mask):
            continue

        offsets = collection.get_offsets()
        if offsets is None or len(offsets) == 0:
            continue

        corr_offsets = offsets[corr_mask]
        ax.scatter(
            corr_offsets[:, 0],
            corr_offsets[:, 1],
            marker="D",
            s=25,
            c=[custom_palette["Correlation"]],
            edgecolors="black",
            linewidths=0.6,
            zorder=collection.get_zorder() + 0.1,
        )

        updated_facecolors = facecolors.copy()
        updated_facecolors[corr_mask, 3] = 0.0
        collection.set_facecolors(updated_facecolors)

        edgecolors = collection.get_edgecolors()
        if edgecolors is not None and len(edgecolors) > 0:
            if len(edgecolors) == 1 and len(updated_facecolors) > 1:
                edgecolors = np.repeat(edgecolors, len(updated_facecolors), axis=0)
            if len(edgecolors) == len(updated_facecolors):
                updated_edgecolors = edgecolors.copy()
                updated_edgecolors[corr_mask, 3] = 0.0
                collection.set_edgecolors(updated_edgecolors)

    if ax.get_legend():
        ax.get_legend().remove()


# 5. Map the function
g.map_dataframe(layered_swarm)

# 6. Final Polish
g.set(ylim=(0, 1.05))
g.set_axis_labels("Dataset", "Score (0.0 - 1.0)")

for row_idx, axes_row in enumerate(g.axes):
    for col_idx, ax in enumerate(axes_row):
        ax.set_xticks(range(len(dataset_order)))
        ax.set_xticklabels(dataset_order, rotation=45, ha="right", fontsize=7)
        ax.set_xlim(-0.5, len(dataset_order) - 0.5)
        ax.tick_params(axis="y", labelsize=7, labelleft=True, left=True)
        ax.yaxis.set_label_position("left")
        ax.yaxis.tick_left()
        ax.set_ylabel(f"{metrics_to_plot[col_idx]} (0.0 - 1.0)", fontsize=8)

# 7. Custom Legend
# 7. Custom Legend
legend_elements = []

# Iterate through all methods and use the custom_palette dictionary for the colors
for m in all_methods:
    marker = "D" if m == "Correlation" else "o"
    marker_edge_color = "black" if m == "Correlation" else "none"
    marker_edge_width = 0.6 if m == "Correlation" else 0.0
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker=marker,
            color="w",
            label=m,
            markerfacecolor=custom_palette[m],
            markersize=6,
            markeredgecolor=marker_edge_color,
            markeredgewidth=marker_edge_width,
        )
    )

# Calculate the number of columns needed to split the items evenly into 2 rows
num_columns = (len(all_methods) + 1) // 2

g.fig.legend(
    handles=legend_elements,
    loc="upper center",  # Anchor point on the legend box
    bbox_to_anchor=(0.5, 0.0),  # Position relative to the figure (x=center, y=bottom)
    title="Methods",
    fontsize="small",
    ncol=num_columns,  # Automatically splits into 2 rows
)

plt.subplots_adjust(top=0.9, right=0.98, bottom=0.25, hspace=0.4, wspace=0.32)
plt.savefig("dot_plot_custom.svg", format="svg", bbox_inches="tight")
