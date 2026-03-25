"""
This module contains the Plotting class, which is responsible for all plotting functions related to scTimeBench results.
"""
from scTimeBench.config import Config
import os


class Plotting:
    """
    Class responsible for all plotting functions related to scTimeBench results, including:
    - Graph similarity heatmaps and scatter plots
    """

    def __init__(self, config: Config):
        self.config = config
        os.makedirs(self.config.plot_output_dir, exist_ok=True)

    def plot_graph_sim_from_csv(self, csv_path):
        """
        Plots graph similarity metrics from a CSV file.
        """
        self.plot_graph_sim_heatmap(csv_path)
        self.plot_graph_sim_scatter(csv_path)

    def plot_graph_sim_heatmap(self, csv_path):
        import pandas as pd
        import numpy as np
        import seaborn as sns
        import matplotlib.pyplot as plt
        import logging

        # first turn off matplotlib debugging
        logging.getLogger("matplotlib").setLevel(logging.WARNING)

        # --- CONFIGURATION ---
        INPUT_FILE = csv_path
        OUTPUT_FILE = self.config.plot_output_dir + "/log_fold_change_heatmap.svg"
        METRICS_OF_INTEREST = [
            "AUC_PRC",
            "AUC_ROC",
            "JaccardSimilarity",
        ]  # Adjust as needed

        # 1. Load and Rename
        df = pd.read_csv(INPUT_FILE)

        def rename_3x_groups(row):
            if "3x" in str(row["dataset"]):
                return f"{row['method']}-3x"
            return row["method"]

        df["method"] = df.apply(rename_3x_groups, axis=1)

        # 2. Pivot & Calculate LFC
        pivot_df = df.pivot_table(
            index=["dataset", "step_setting", "metric", "method", "threshold_type"],
            columns="time_type",
            values="result",
        ).reset_index()

        pivot_df["LFC"] = np.log2(
            pivot_df["Pseudotime"] / pivot_df["Real Time"] + np.finfo(float).eps
        )  # Add small value to avoid log(0)

        # 3. Filter Logic
        pivot_df = pivot_df[
            (pivot_df["metric"].isin(METRICS_OF_INTEREST))
            & (pivot_df["threshold_type"] == "prc")
        ]

        # 4. Get unique step_settings to determine number of subplots
        # ... (previous code)

        # 4. Get unique step_settings
        step_settings = pivot_df["step_setting"].unique()
        _, axes = plt.subplots(
            len(step_settings), 1, figsize=(14, 8 * len(step_settings))
        )

        if len(step_settings) == 1:
            axes = [axes]

        # STEEPER GRADIENT LOGIC:
        # We cap the visual range at 1.5. This means anything > 1.5x (or < -1.5x)
        # is fully saturated, making the "middle" transitions much sharper.
        clean_vals = pivot_df.replace([np.inf, -np.inf], np.nan)
        actual_max = clean_vals["LFC"].abs().max()
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

    def plot_graph_sim_scatter(self, csv_path):
        """
        Plots a scatter plot of graph similarity metrics from a CSV file, with custom colors and legend.
        """
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        # Load data
        df = pd.read_csv(csv_path)

        # 1. Filter Data
        metrics_to_plot = ["JaccardSimilarity", "AUC_PRC", "AUC_ROC"]
        plot_df = df[
            (df["threshold_type"] == "prc")
            & (df["time_type"] == "Real Time")
            & (df["metric"].isin(metrics_to_plot))
        ].copy()

        # Sort and prepare combined labels
        plot_df["metric"] = pd.Categorical(
            plot_df["metric"], categories=metrics_to_plot, ordered=True
        )
        plot_df = plot_df.sort_values(["metric", "dataset"])

        plot_df["x_axis_group"] = (
            plot_df["dataset"].astype(str) + " | " + plot_df["metric"].astype(str)
        )
        group_order = list(dict.fromkeys(plot_df["x_axis_group"]))

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
            height=4,
            aspect=1.8,
            margin_titles=True,
            sharex=True,
        )

        # 4. Define Plotting Function
        def layered_swarm(data, **kwargs):
            ax = plt.gca()

            # Now we just plot everything in one go using the unified palette
            sns.swarmplot(
                data=data,
                x="x_axis_group",
                y="result",
                hue="method",
                order=group_order,
                palette=custom_palette,
                size=5,
                dodge=False,
                ax=ax,
                alpha=1.0,
            )

            if ax.get_legend():
                ax.get_legend().remove()

        # 5. Map the function
        g.map_dataframe(layered_swarm)

        # 6. Final Polish
        g.set(ylim=(0, 1.05))
        g.set_axis_labels("", "Score (0.0 - 1.0)")

        for ax in g.axes.flat:
            ax.set_xticks(range(len(group_order)))
            ax.set_xticklabels(group_order, rotation=90, ha="center", fontsize=7)

            for i in range(len(group_order)):
                if (i + 1) % len(metrics_to_plot) == 0:
                    ax.axvline(
                        i + 0.5, color="black", linestyle="-", alpha=0.1, linewidth=1
                    )

        # 7. Custom Legend
        # 7. Custom Legend
        legend_elements = []

        # Iterate through all methods and use the custom_palette dictionary for the colors
        for m in all_methods:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=m,
                    markerfacecolor=custom_palette[m],
                    markersize=6,
                )
            )

        # Calculate the number of columns needed to split the items evenly into 2 rows
        num_columns = (len(all_methods) + 1) // 2

        g.fig.legend(
            handles=legend_elements,
            loc="upper center",  # Anchor point on the legend box
            bbox_to_anchor=(
                0.5,
                0.0,
            ),  # Position relative to the figure (x=center, y=bottom)
            title="Methods",
            fontsize="small",
            ncol=num_columns,  # Automatically splits into 2 rows
        )

        plt.subplots_adjust(top=0.9, right=0.88, bottom=0.35, hspace=0.4)
        plt.savefig(
            f"{self.config.plot_output_dir}/graph_sim_scatter.svg",
            format="svg",
            bbox_inches="tight",
        )
