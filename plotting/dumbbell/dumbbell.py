import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def save_dataset_plots(df):
    unique_datasets = df["dataset"].unique()

    # Define your preferred order for metrics
    metric_order = ["F1", "Accuracy", "Jaccard", "AUROC"]

    for ds_name in unique_datasets:
        print(f"Processing plot for: {ds_name}...")

        # 1. Filter and Pivot
        ds_df = df[df["dataset"] == ds_name]

        # Pivot so 'Real Time' and 'Pseudotime' are columns
        # This assumes your 'time_type' column contains exactly these strings
        plot_df = ds_df.pivot_table(
            index=["method", "setting", "metric"], columns="time_type", values="result"
        ).reset_index()

        # 2. Setup Plot
        sns.set_theme(style="white")
        g = sns.FacetGrid(plot_df, col="setting", height=7, aspect=0.55, sharey=True)

        # 3. Corrected Internal Function
        def draw_metric_dumbbells(data, **kwargs):
            ax = plt.gca()
            is_leftmost = ax.get_subplotspec().colspan.start == 0

            # Identify methods present in this dataset
            method_list = sorted(data["method"].unique().tolist())

            # Spacing for the 4 metrics per method
            offsets = {met: (i * 0.22) - 0.33 for i, met in enumerate(metric_order)}

            for i, mth in enumerate(method_list):
                # Zebra striping
                if i % 2 == 0:
                    ax.axhspan(i - 0.5, i + 0.5, color="#f7f7f7", zorder=0)

                # Method label on left facet
                if is_leftmost:
                    ax.text(
                        -0.38,
                        i,
                        mth,
                        rotation=90,
                        va="center",
                        ha="center",
                        fontweight="bold",
                        fontsize=14,
                        color="black",
                    )

            for _, row in data.iterrows():
                # Get Y position
                m_idx = method_list.index(row["method"])
                y_pos = m_idx + offsets.get(row["metric"], 0)

                # Draw Jump Line
                # Ensure the columns 'Real Time' and 'Pseudotime' exist after pivot
                if "Real Time" in row and "Pseudotime" in row:
                    ax.plot(
                        [row["Real Time"], row["Pseudotime"]],
                        [y_pos, y_pos],
                        color="black",
                        alpha=0.2,
                        linewidth=1,
                        zorder=1,
                    )

                    # Real Time (Blue)
                    ax.scatter(
                        row["Real Time"],
                        y_pos,
                        color="#1f77b4",
                        s=60,
                        zorder=3,
                        edgecolors="white",
                        linewidth=0.8,
                    )

                    # Pseudotime (Red)
                    ax.scatter(
                        row["Pseudotime"],
                        y_pos,
                        color="#d62728",
                        s=60,
                        zorder=4,
                        edgecolors="white",
                        linewidth=0.8,
                    )

                if is_leftmost:
                    ax.text(
                        -0.03,
                        y_pos,
                        row["metric"],
                        va="center",
                        ha="right",
                        fontsize=9,
                        color="#444444",
                    )

        # 4. Map the fixed function
        g.map_dataframe(draw_metric_dumbbells)

        # 5. Final Formatting
        g.set_titles(col_template="{col_name}")
        for ax in g.axes.flat:
            ax.set_yticks([])
            ax.set_xlabel("Score", fontsize=11)
            ax.set_xlim(0, 1.0)

            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])
            ax.grid(axis="x", linestyle="--", alpha=0.3)
            sns.despine(left=True, ax=ax)

        plt.subplots_adjust(wspace=0.05)

        # Legend
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="white",
                label="Real Time",
                markerfacecolor="#1f77b4",
                markersize=10,
                markeredgecolor="white",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="white",
                label="Pseudotime",
                markerfacecolor="#d62728",
                markersize=10,
                markeredgecolor="white",
            ),
        ]
        g.figure.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.6, 1.05),
            ncol=2,
            frameon=False,
            fontsize=11,
        )

        g.figure.suptitle(f"{ds_name} Dataset", fontsize=16, x=0.6, y=1.08)

        # 6. Save and Clean up
        filename = f"{ds_name.replace(' ', '_')}_dumbbell.svg"
        g.figure.savefig(filename, format="svg", bbox_inches="tight")
        plt.close(g.figure)
        print(f"Successfully saved {filename}")


# To use with your real CSV:
# --- MOCK DATA GENERATOR (For testing the script) ---
# This simulates your CSV structure
# methods = ['Method 1', 'Method 2', 'Method 3']
# datasets = ['Liver', 'Lung', 'Brain']
# settings = ['Simple', 'All Paths']
# metrics = ['F1', 'Accuracy', 'Jaccard', 'AUROC']
# time_types = ['Real Time', 'Pseudotime']

# mock_rows = []
# for d in datasets:
#     for m in methods:
#         for s in settings:
#             for met in metrics:
#                 for t in time_types:
#                     res = 0.5 + (0.3 if t == 'Pseudotime' else 0.2)
#                     mock_rows.append([m, d, s, met, t, res])

# df = pd.DataFrame(mock_rows, columns=['method', 'dataset', 'setting', 'metric', 'time_type', 'result'])
# --- END MOCK DATA ---
df = pd.read_csv("small_dumbbell.csv")
save_dataset_plots(df)
