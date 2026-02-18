from crispy_fishstick.metrics.ontology_based.graph_sim.base import (
    GraphSimMetric,
    AdjacencyMatrixType,
    CELL_TYPE_TO_ID_KEY,
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

        ref_graph_output = os.path.join(self.dataset_dir, "reference_graph")

        if not os.path.exists(ref_graph_output + ".png"):
            build_graph_image(
                graph_ref[AdjacencyMatrixType.UNWEIGHTED],
                cell_id_to_type,
                ref_graph_output,
            )
        build_graph_image(
            graph_pred[AdjacencyMatrixType.WEIGHTED],
            cell_id_to_type,
            os.path.join(self.traj_dir, "predicted_graph"),
            is_weighted=True,
        )
        build_graph_image(
            graph_pred[AdjacencyMatrixType.UNWEIGHTED],
            cell_id_to_type,
            os.path.join(self.traj_dir, "predicted_unweighted_graph"),
        )

        return  # Visualization metric does not return a numeric score


class StackedBarPlot(GraphSimMetric):
    def _graph_sim_eval(self, graph_pred, graph_ref):
        """
        This is a special metric that generates stacked bar plots for the predicted trajectory.

        We build both the predicted and reference graphs and save them as images.
        """
        import matplotlib.pyplot as plt
        import pandas as pd

        TIMEPOINT_COL = "Time Point"
        CELLTYPE_COL = "Cell Type"
        COUNT_COL = "Count"

        def plot_stacked_bar(traj_data, output_path, title):
            """
            Plots a stacked bar chart of cell type ratios over time.
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

            # 3. Normalize to 1 (100%) to create a "bar" / Ratio plot
            # div(..., axis=0) divides each row by its sum
            df_ratios = df_pivot.div(df_pivot.sum(axis=1), axis=0)

            # 4. Plot as stacked bars
            ax = df_ratios.plot(
                kind="bar",
                stacked=True,
                cmap="viridis",
                alpha=0.9,
                figsize=(10, 6),
                width=0.85,
            )

            # 5. Force exact X-axis labels from the real timepoints
            labels = [f"{x:.2f}" for x in df_ratios.index.tolist()]
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

            # Keep the y-axis bounded to proportions
            ax.set_ylim(0, 1)

            # Add a little padding at the bottom so rotated text doesn't get cut off
            plt.subplots_adjust(bottom=0.2)

            plt.title(title)
            plt.xlabel("Time Point")
            plt.ylabel("Proportion")

            # Move legend outside if it's crowded
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()

            plt.savefig(output_path)
            plt.close()  # Close the figure to free memory
            logging.info(f"Stacked bar plot saved to {output_path}")

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

        ref_plot_output = os.path.join(
            self.dataset_dir, "reference_stacked_bar_plot.png"
        )
        if not os.path.exists(ref_plot_output):
            plot_stacked_bar(
                source_df,
                ref_plot_output,
                f"True Cell Type Proportions Over {self.time_label} for {self.dataset_name}",
            )
        plot_stacked_bar(
            target_df,
            os.path.join(self.traj_dir, "target_stacked_bar_plot.png"),
            f'Predicted Target Cell Type Proportions Over {self.time_label} for {self.config.model["name"]} on {self.dataset_name}',
        )

        return  # Visualization metric does not return a numeric score
