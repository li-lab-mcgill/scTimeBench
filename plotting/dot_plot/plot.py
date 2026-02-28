import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

# 1. Filter the data
metrics_to_plot = ["JaccardSimilarity", "AUC_PRC", "AUC_ROC"]
plot_df = df[
    (df["prc_threshold"] == True)
    & (df["time_type"] == "Real Time")
    & (df["metric"].isin(metrics_to_plot))
].copy()

# Separate Cooccurrence from the rest for layered plotting
field_df = plot_df[plot_df["method"] != "Cooccurrence"]
cooc_df = plot_df[plot_df["method"] == "Cooccurrence"]

# 2. Setup the Grid
g = sns.FacetGrid(
    plot_df, row="dataset", col="step_setting", height=4, aspect=1.4, margin_titles=True
)


# 3. Custom plotting function to handle layering
def layered_swarm(data, **kwargs):
    ax = plt.gca()

    # Layer 1: The Field (Colored dots)
    # Using 'Set2' or 'Paired' gives distinct colors to each method
    sns.swarmplot(
        data=data[data["method"].isin(["Cooccurrence", "Worst Case"]) == False],
        x="metric",
        y="result",
        hue="method",
        palette="Set2",
        size=7,
        dodge=False,
        ax=ax,
    )

    # Layer 2: Cooccurrence (Larger, different shape)
    # We plot this on top so it's never hidden by the cluster

    # 1. Define your highlight colors
    highlight_palette = {
        "Cooccurrence": "#E63946",  # Deep Red
        "Worst Case": "#1D3557",  # Dark Navy (or any contrasting color)
    }

    # 2. Update your Layer 2 code
    sns.swarmplot(
        data=data[data["method"].isin(["Cooccurrence", "Worst Case"])],
        x="metric",
        y="result",
        hue="method",  # Enable hue to use the palette
        palette=highlight_palette,
        marker="D",
        size=7,
        linewidth=1,
        edgecolor="black",
        dodge=False,
        ax=ax,
        legend=False,  # Usually False here to avoid duplicate legends in facets
    )

    # Clean up duplicate legends
    handles, labels = ax.get_legend_handles_labels()
    ax.legend_.remove() if ax.legend_ else None


# 4. Map the custom function
g.map_dataframe(layered_swarm)

# 5. Final Polish
g.set(ylim=(0, 1))
g.set_axis_labels("", "Score (0.0 - 1.0)")

for (row_val, col_val), ax in g.axes_dict.items():
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.3)
    # Fill the 'Underperformance' area
    ax.axhspan(0, 0.5, color="red", alpha=0.02)

# Create a custom legend manually to ensure the Diamond shows up
from matplotlib.lines import Line2D

legend_elements = [
    Line2D(
        [0],
        [0],
        marker="D",
        color="w",
        label="Cooccurrence",
        markerfacecolor="#E63946",
        markeredgecolor="black",
    ),
    Line2D(
        [0],
        [0],
        marker="D",
        color="w",
        label="Worst Case",
        markerfacecolor="#457B9D",
        markeredgecolor="black",
    ),
]

# Add the rest of the methods to the legend
other_methods = field_df["method"].unique()
colors = sns.color_palette("Set2", len(other_methods))
for i, m in enumerate(other_methods):
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=m,
            markerfacecolor=colors[i],
            markersize=6,
        )
    )

g.fig.legend(
    handles=legend_elements,
    loc="center right",
    bbox_to_anchor=(1.15, 0.5),
    title="Methods",
)

plt.subplots_adjust(top=0.9, right=0.85)
g.fig.suptitle("Cooccurrence vs. Others: Absolute Performance Gap", fontsize=15)

plt.savefig("dot_plot.png", dpi=300, bbox_inches="tight")
plt.show()
