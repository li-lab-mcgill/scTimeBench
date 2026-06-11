"""
Meta-Based Metrics.

This is a metric that operates by running multiple submetrics.

It takes in as its parameters a set of submetrics to run, and returns this
to the metric to figure out what to do with that information.

This is useful for things such as calculating the error bar of a metric,
by running it multiple times with different seeds, or also for calculating
the perturbation's baselines as well.
"""
from scTimeBench.metrics.base import BaseMetric, create_submetric_instance
from scTimeBench.shared.constants import RequiredOutputFiles, ObservationColumns
from scTimeBench.shared.utils import load_test_dataset
from scTimeBench.shared.dataset.registry import GarciaAlonsoDataset

import logging
import json
import os
import random


class MetaMetric(BaseMetric):
    def _setup_supported_datasets(self):
        # ** NOTE: must define the following two attributes, though each subclass **
        # ** Must also define required_feature_specs and output_path_name individually, as they likely require **
        # ** different output files. **
        self.supported_datasets = [
            GarciaAlonsoDataset.__name__,
        ]
        self.default_dataset_group = "ontology_based"

        # get the path to the shared default datasets config
        self.default_datasets_path = os.path.join(
            os.path.dirname(__file__), "..", "shared", "default_datasets.yaml"
        )

        self.optional_datasets_path = os.path.join(
            os.path.dirname(__file__), "..", "shared", "optional_datasets.yaml"
        )

    def _defaults(self):
        """The default parameters for meta-based metrics."""
        return {}

    def _setup_method_output_requirements(self):
        """Skip this, as it's a higher level class."""
        # because we're running submetrics, we just ignore this
        self.required_outputs = [RequiredOutputFiles.META_FLAG]

    def _prep_kwargs_for_submetric_eval(self, output_path, dataset, method):
        return {"output_path": output_path, "dataset": dataset, "method": method}

    def _run_submetric(self, submetric_config):
        """
        Run a submetric and return the result from eval.
        """
        # here, we'll change config to force a rerun because
        # the submetric will likely require different outputs and require
        # redoing the metric entirely
        self.config.force_rerun = True
        submetric_instance = create_submetric_instance(
            self.config, self.db_manager, submetric_config
        )
        return submetric_instance.eval()


class MetaPerturbation(MetaMetric):
    def _defaults(self):
        return {
            "random_trials": 1,
        }

    def _handle_transition(
        self, transition, cell_type_to_timepoint, gene_col_name, gene_list
    ):
        """
        Given transition of the format:
          - start: 'GC'
            targets:
            - end: 'oogonia'
                genes: ['STRA8', 'ZGLP1', 'ZIC1']
            - end: 'pre_spermatogonia'
                genes: ['SOX4', 'EGR4', 'KLF6', 'KLF7']

        We run and get the difference in the submetric for each of these targets
        and return the submetric configuration which should look like:
            name: PerturbationCellTypeProportion
            perturbation_set_config:
            - gene_col_name: gene_symbols
                knockin_genes: ['SOX4', 'EGR4', 'KLF6', 'KLF7']
                knockout_genes: ['STRA8', 'ZGLP1', 'ZIC1']
                timepoint_idx: 0
            trajectory_infer_model:
                name: CellTypist
                renormalize: True
        except we infer the timepoint_idx to be the timepoint of the start cell type,
        with the largest number of cells in the test dataset.

        We also return the cell types that are used.
        """
        # should have start and targets
        start = transition["start"]
        targets = transition["targets"]

        submetrics = []
        timepoint_idx = cell_type_to_timepoint.get(start)
        assert (
            timepoint_idx is not None
        ), f"Could not infer timepoint idx for cell type {start}."

        cell_types = []
        all_genes = set()

        def generate_submetric(alias, knockin_genes=[], knockout_genes=[]):
            perturbations = []
            if len(knockin_genes) > 0 or len(knockout_genes) > 0:
                perturbations.append(
                    {
                        "gene_col_name": gene_col_name
                        if gene_col_name is not None
                        else None,
                        "knockin_genes": knockin_genes,
                        "knockout_genes": knockout_genes,
                        "timepoint_idx": 0,
                    }
                )
            return {
                "name": "PerturbationCellTypeProportion",
                "alias": alias,
                "perturbation_set_config": {
                    "filter_cell_type": start,
                    "filter_tp_idx": timepoint_idx,
                    "perturbations": perturbations,
                },
                "trajectory_infer_model": {
                    "name": "CellTypist",
                    "renormalize": True,
                },
            }

        max_knockin = 0
        max_knockout = 0

        for target in targets:
            logging.debug(f"Running submetric for transition {start} -> {target}")
            end = target["end"]
            cell_types.append(end)
            knockin_genes = target["genes"]
            all_genes.update(knockin_genes)
            max_knockin = max(max_knockin, len(knockin_genes))
            logging.debug(f"Genes for transition {start} -> {end}: {knockin_genes}")

            # knockout genes should be the ones not included here, but are in other targets
            # for example, if we have A to B: [g1, g2], A to C: [g3, g4],
            # then A to B knockout genes should be [g3, g4], and A to C knockout genes should be [g1, g2]
            knockout_genes = []
            for other_target in targets:
                if other_target["end"] != end:
                    knockout_genes.extend(other_target["genes"])
            max_knockout = max(max_knockout, len(knockout_genes))
            logging.debug(
                f"Knockout genes for transition {start} -> {end}: {knockout_genes}"
            )

            submetrics.append(generate_submetric(end, knockin_genes, knockout_genes))

        # add a baseline one
        submetrics.append(generate_submetric("baseline"))

        # finally, also add a random one to check, where we perturb 5 random genes on
        # and 5 random genes off, that are not in the original perturbation set
        # we need to sort it so that the random genes are the same across different runs
        # redo the random seed to ensure consistency across multiple runs
        random.seed(self.config.random_seed)

        for i in range(self.params["random_trials"]):
            random_genes = sorted(list(set(gene_list) - all_genes))
            random_knockin_genes = random.sample(
                random_genes, min(max_knockin, len(random_genes))
            )
            remaining_genes = sorted(
                list(set(random_genes) - set(random_knockin_genes))
            )
            random_knockout_genes = random.sample(
                remaining_genes, min(max_knockout, len(remaining_genes))
            )
            logging.debug(f"Random knockin genes: {random_knockin_genes}")
            logging.debug(f"Random knockout genes: {random_knockout_genes}")
            submetrics.append(
                generate_submetric(
                    f"random_{i}", random_knockin_genes, random_knockout_genes
                )
            )

        return submetrics, cell_types

    def _cell_type_to_timepoint_idx(self, test_ann_data):
        """
        Returns a mapping of the cell type to the inferred timepoint idx, which is the timepoint with the largest number of cells for that cell type.
        """
        cell_types = test_ann_data.obs[ObservationColumns.CELL_TYPE.value].unique()
        timepoint_mapping = {}
        tps = sorted(test_ann_data.obs[ObservationColumns.TIMEPOINT.value].unique())
        for cell_type in cell_types:
            cells = test_ann_data[
                test_ann_data.obs[ObservationColumns.CELL_TYPE.value] == cell_type
            ]
            timepoint_counts = cells.obs[
                ObservationColumns.TIMEPOINT.value
            ].value_counts()
            # get the timepoint with the largest number of cells
            inferred_timepoint = timepoint_counts.idxmax()
            logging.debug(
                f"Inferred timepoint for cell type {cell_type}: {inferred_timepoint}"
            )
            timepoint_idx = tps.index(inferred_timepoint)
            logging.debug(
                f"Inferred timepoint idx for cell type {cell_type}: {timepoint_idx}"
            )
            timepoint_mapping[cell_type] = timepoint_idx
        return timepoint_mapping

    def _submetric_eval(self, output_path, dataset, method):
        logging.debug(f"Dataset cell lineage genes: {dataset.cell_lineage_genes}")

        test_ann_data = load_test_dataset(output_path)
        cell_type_to_timepoint = self._cell_type_to_timepoint_idx(test_ann_data)

        eval = {}

        gene_col_name = dataset.cell_lineage_genes.get("gene_col_name", None)

        if gene_col_name is not None:
            gene_list = test_ann_data.var[gene_col_name].tolist()
        else:
            gene_list = test_ann_data.var_names.tolist()

        for transition in dataset.cell_lineage_genes["cell_lineage_genes"]:
            logging.debug(f"Handling transition: {transition}")
            submetrics, cell_types = self._handle_transition(
                transition,
                cell_type_to_timepoint,
                gene_col_name,
                gene_list,
            )
            logging.debug(
                f"Submetrics for transition {transition['start']} -> {[t['end'] for t in transition['targets']]}: {submetrics}"
            )

            results = {}
            # now let's run the submetric
            for submetric in submetrics:
                result = self._run_submetric(submetric)
                logging.debug(f"Result of submetric {submetric['alias']}: {result}")
                results[submetric["alias"]] = result

            # now we do for each cell type, we're going to calculate
            # 1. difference to baseline
            # 2. average difference to others
            # which is the average of the difference for the same cell type
            # for the baseline, we'll report the raw score

            # here we calculate the differences of percentage for a specific cell type
            # as described above
            for cell_type in cell_types:
                logging.debug(
                    f"Results for cell type {cell_type}: {results[cell_type]}"
                )
                baseline_result = results["baseline"].get(cell_type, 0)

                increase_cell_type_result = results[cell_type].get(cell_type, 0)
                baseline_delta = increase_cell_type_result - baseline_result

                # actually let's not create the average but instead have each difference be calculated
                other_cells_delta = {}

                for other_cell_type in cell_types:
                    if other_cell_type == cell_type:
                        continue
                    other_cells_delta[other_cell_type] = (
                        results[other_cell_type].get(cell_type, 0) - baseline_result
                    )

                # now we get the random baseline delta, which is the average of the random trials
                random_deltas = []
                for i in range(self.params["random_trials"]):
                    random_delta = (
                        results[f"random_{i}"].get(cell_type, 0) - baseline_result
                    )
                    random_deltas.append(random_delta)
                avg_random_delta = sum(random_deltas) / len(random_deltas)
                min_random_delta = min(random_deltas)
                max_random_delta = max(random_deltas)

                eval[f'{transition["start"]}->{cell_type}'] = {
                    "baseline": baseline_result,
                    "baseline_delta": baseline_delta,
                    "other_cells_delta": other_cells_delta,
                    "avg_random_delta": avg_random_delta,
                    "min_random_delta": min_random_delta,
                    "max_random_delta": max_random_delta,
                }

        logging.debug(f"Results of evaluation: {eval}")

        # now we can try to aggregate this information to get an overall score
        # we can do:
        # 1. a) Accuracy of predicting the correct direction for increase
        #    b) Accuracy of predicting correct direction for decrease
        # 2. Average increase in the perturbed cell type proportion compared to baseline
        # 3. Average percentage increase compared to the random baseline
        # 4. Accuracy of better increase compared to random baseline
        aggregate_scores = {
            "pos_direction_accuracy": 0,
            "neg_direction_accuracy": 0,
            "avg_increase": 0,
            "avg_increase_random_baseline": 0,
            "beat_random_accuracy": 0,
        }
        total_other_cells = 0
        for _, result in eval.items():
            if result["baseline_delta"] > 0:
                aggregate_scores["pos_direction_accuracy"] += 1

            # now we iterate through all the other cells
            for other_cell_type, delta in result["other_cells_delta"].items():
                if delta < 0:
                    aggregate_scores["neg_direction_accuracy"] += 1
                total_other_cells += 1

            aggregate_scores["avg_increase"] += result["baseline_delta"]
            aggregate_scores["avg_increase_random_baseline"] += (
                result["baseline_delta"] - result["avg_random_delta"]
            )
            if result["baseline_delta"] > result["avg_random_delta"]:
                aggregate_scores["beat_random_accuracy"] += 1

        num_cell_types = len(eval)
        aggregate_scores = {
            k: v
            / (total_other_cells if k == "neg_direction_accuracy" else num_cell_types)
            for k, v in aggregate_scores.items()
        }

        logging.debug(f"Aggregate scores: {aggregate_scores}")

        final_result = {
            "aggregate": aggregate_scores,
            "eval": eval,
        }

        final_json = json.dumps(final_result, sort_keys=True)
        self.db_manager.insert_eval(
            method, self.__class__.__name__, self._get_param_encoding(), final_json
        )
