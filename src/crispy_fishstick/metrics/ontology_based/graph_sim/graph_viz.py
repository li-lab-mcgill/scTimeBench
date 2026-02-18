from crispy_fishstick.metrics.ontology_based.graph_sim.base import (
    GraphSimMetric,
    AdjacencyMatrixType,
    CELL_TYPE_TO_ID_KEY,
    DATASET_NAME_KEY,
)
from crispy_fishstick.shared.utils import load_test_dataset
from crispy_fishstick.shared.constants import ObservationColumns
import os
import logging


class GraphVisualization(GraphSimMetric):
    def _graph_sim_eval(self, graph_pred, graph_ref):
        """
        This is a special metric that uses graphviz to generate visualizations of the trajectory

        We build both the predicted and reference graphs and save them as images.
        """
        import graphviz

        def build_graph_image(
            adj_matrix, cell_id_to_type, output_path, is_weighted=False
        ):
            dot = graphviz.Digraph(format="png")
            num_nodes = adj_matrix.shape[0]

            # Add nodes with labels
            for node_id in range(num_nodes):
                cell_type = cell_id_to_type.get(node_id, "Unknown")
                dot.node(str(node_id), label=cell_type)

            # Add edges
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if adj_matrix[i, j] < self._defaults()["edge_threshold"]:
                        continue
                    if is_weighted:
                        dot.edge(str(i), str(j), label=f"{adj_matrix[i, j]:.2f}")
                    else:
                        dot.edge(str(i), str(j))

            dot.render(output_path, cleanup=True)
            logging.info(f"Graph visualization saved to {output_path}.png")

        # take the reverse dictionary
        cell_id_to_type = {v: k for k, v in graph_ref[CELL_TYPE_TO_ID_KEY].items()}

        build_graph_image(
            graph_ref[AdjacencyMatrixType.UNWEIGHTED],
            cell_id_to_type,
            os.path.join(self.output_path, "reference_graph"),
        )
        build_graph_image(
            graph_pred[AdjacencyMatrixType.WEIGHTED],
            cell_id_to_type,
            os.path.join(self.output_path, "predicted_graph"),
            is_weighted=True,
        )
        build_graph_image(
            graph_pred[AdjacencyMatrixType.UNWEIGHTED],
            cell_id_to_type,
            os.path.join(self.output_path, "predicted_unweighted_graph"),
        )

        return  # Visualization metric does not return a numeric score


class StackedDensityPlot(GraphSimMetric):
    def _graph_sim_eval(self, graph_pred, graph_ref):
        """
        This is a special metric that generates stacked density plots for the predicted trajectory.

        We build both the predicted and reference graphs and save them as images.
        """
        import matplotlib.pyplot as plt
        import pandas as pd

        TIMEPOINT_COL = "Time Point"
        CELLTYPE_COL = "Cell Type"
        COUNT_COL = "Count"

        def plot_stacked_density(traj_data, output_path, title):
            """
            Plots a stacked area chart of cell type ratios over time.
            Expects traj_data to be a DataFrame with columns for Time, Cell Type, and Count.
            """
            # 1. Pivot the data:
            # Rows = Timepoints, Columns = Cell Types, Values = Counts
            # We use pivot_table to handle potential duplicate entries safely (summing them)
            df_pivot = traj_data.pivot_table(
                index=TIMEPOINT_COL,
                columns=CELLTYPE_COL,
                values=COUNT_COL,
                aggfunc="sum",
            ).fillna(0)

            # 2. Sort index to ensure time flows correctly (6 -> 21)
            df_pivot.sort_index(inplace=True)

            # 3. Normalize to 1 (100%) to create a "Density" / Ratio plot
            # div(..., axis=0) divides each row by its sum
            df_ratios = df_pivot.div(df_pivot.sum(axis=1), axis=0)

            # 4. Plot
            plt.figure(figsize=(15, 6))

            # 'area' creates a stacked plot by default.
            # It uses the Index (your real float timepoints) as the x-axis automatically.
            ax = df_ratios.plot(
                kind="area",
                stacked=True,
                cmap="viridis",
                alpha=0.8,
                figsize=(10, 6),
                linewidth=0,  # Removes lines between stacks for a smoother look
            )

            # 5. Force exact X-axis ticks
            # This ensures you see 6, 9, 12... instead of auto-generated ticks
            real_timepoints = df_ratios.index.tolist()
            ax.set_xticks(real_timepoints)

            # Format: 1 decimal place only
            labels = [f"{x:.1f}" for x in real_timepoints]

            # Apply rotation
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

            # Add a little padding at the bottom so rotated text doesn't get cut off
            plt.subplots_adjust(bottom=0.2)
            plt.xlim(min(real_timepoints), max(real_timepoints))

            plt.title(title)
            plt.xlabel("Time Point")
            plt.ylabel("Proportion")

            # Move legend outside if it's crowded
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()

            plt.savefig(output_path)
            plt.close()  # Close the figure to free memory
            logging.info(f"Stacked area plot saved to {output_path}")

        per_tp_traj = self.trajectory_infer_model.infer_trajectory(
            self.output_path, per_tp=True
        )

        # need to get the original test-ann-data to build the proper last timepoint
        test_ann_data = load_test_dataset(self.output_path)
        unique_tps = (
            test_ann_data.obs[ObservationColumns.TIMEPOINT.value].unique().tolist()
        )
        unique_tps.sort()
        last_tp = unique_tps[-1]

        logging.debug(f"Per-timepoint predicted trajectory: {per_tp_traj}")

        # we can get the source cell type counts by aggregating over the target cell types
        source_records = []
        target_records = []

        for tp, traj in per_tp_traj.items():
            tp = float(tp)
            target_cell_types = {}

            for source_cell_type, target_distribution in traj.items():
                total_count = sum(target_distribution.values())
                source_records.append(
                    {
                        TIMEPOINT_COL: tp,
                        CELLTYPE_COL: source_cell_type,
                        COUNT_COL: total_count,
                    }
                )

                for target_cell_type, count in target_distribution.items():
                    if target_cell_type not in target_cell_types:
                        target_cell_types[target_cell_type] = 0
                    target_cell_types[target_cell_type] += count

            assert tp < last_tp, "Last timepoint should not have target cell types."

            for cell_type, count in target_cell_types.items():
                target_records.append(
                    {
                        TIMEPOINT_COL: unique_tps[unique_tps.index(tp) + 1],
                        CELLTYPE_COL: cell_type,
                        COUNT_COL: count,
                    }
                )

        # then finally we add the real last timepoint from the test data
        last_tp_data = test_ann_data.obs[
            test_ann_data.obs[ObservationColumns.TIMEPOINT.value] == last_tp
        ]
        last_tp_cell_type_counts = last_tp_data[
            ObservationColumns.CELL_TYPE.value
        ].value_counts()
        for cell_type, count in last_tp_cell_type_counts.items():
            source_records.append(
                {
                    TIMEPOINT_COL: last_tp,
                    CELLTYPE_COL: cell_type,
                    COUNT_COL: count,
                }
            )

        # then let's create DataFrames
        source_df = pd.DataFrame(source_records)
        target_df = pd.DataFrame(target_records)
        # ignore the debug of matplotlib
        logging.getLogger("matplotlib").setLevel(logging.WARNING)

        plot_stacked_density(
            source_df,
            os.path.join(self.output_path, "source_stacked_density_plot.png"),
            f"True Cell Type Proportions Over Time for {graph_ref[DATASET_NAME_KEY]}",
        )
        plot_stacked_density(
            target_df,
            os.path.join(self.output_path, "target_stacked_density_plot.png"),
            f'Predicted Target Cell Type Proportions Over Time for {self.config.model["name"]} on {graph_ref[DATASET_NAME_KEY]}',
        )

        return  # Visualization metric does not return a numeric score
