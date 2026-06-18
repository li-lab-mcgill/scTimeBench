"""
Measures average gene expression over time for perturbation analyses.
"""
from scTimeBench.metrics.perturbation.base import (
    PerturbationBasedMetrics,
    GlobalPerturbationBasedMetrics,
)
from scTimeBench.shared.utils import (
    load_output_file,
    load_test_dataset,
    is_raw,
    undo_log_normalization,
)
from scTimeBench.shared.constants import RequiredOutputFiles, ObservationColumns

import logging
import os


class PerturbationGeneExpression(PerturbationBasedMetrics):
    def _defaults(self):
        return {
            # List of genes to evaluate, if None, will evaluate genes found in perturbation
            "perturbed_genes": None,
        }

    def _prep_kwargs_for_submetric_eval(self, output_path, dataset, method):
        # We need to modify this so it saves it to eval output path
        return {
            "output_path": output_path,
            "method": method,
        }

    def _submetric_eval(self, output_path, method):
        logging.debug(
            f"Returning the average gene expression for the perturbed genes across time"
        )
        genes = self.params["perturbed_genes"]
        if genes is None:
            # if not specified, we will evaluate all the genes found in the perturbation set
            genes = self.perturbation_set.get_genes()

        logging.debug(f"Evaluating genes: {genes}")

        # now we will get the gene expression for each of the perturbed genes across time
        # we will return a dictionary of the form:
        # gene -> timepoint -> expression data (list of expression values across cells)
        gene_expression_over_time = {}
        eval_output_path = os.path.join(output_path, self._get_relative_output_path())
        perturbed_data = load_output_file(
            eval_output_path, RequiredOutputFiles.PERTURBED_TEST_ANN_DATA
        )

        # then let's first un-log normalize the data
        if not is_raw(perturbed_data):
            perturbed_data = undo_log_normalization(perturbed_data)

        tps = sorted(
            list(perturbed_data.obs[ObservationColumns.TIMEPOINT.value].unique())
        )

        test_ann_data = load_test_dataset(output_path)
        if self.perturbation_set.gene_col_name is None:
            gene_names_list = test_ann_data.var_names.tolist()
        else:
            gene_names_list = test_ann_data.var[
                self.perturbation_set.gene_col_name
            ].tolist()

        for gene in genes:
            gene_col_idx = gene_names_list.index(gene)
            gene_expression_over_time[gene] = {}

            # now let's calculate the average expression for this gene across samples
            # per timepoint
            for tp in tps:
                tp_data = perturbed_data[
                    perturbed_data.obs[ObservationColumns.TIMEPOINT.value] == tp
                ]
                gene_expression_over_time[gene][tp] = tp_data[:, gene_col_idx].X

        return gene_expression_over_time


class GlobalPerturbationGeneExpression(GlobalPerturbationBasedMetrics):
    def _defaults(self):
        return {
            # List of genes to evaluate, if None, will evaluate genes found in perturbation
            "affected_genes": None,
        }

    def _prep_kwargs_for_submetric_eval(self, output_path, dataset, method):
        # We need to modify this so it saves it to eval output path
        return {
            "output_path": output_path,
            "method": method,
        }

    def _submetric_eval(self, output_path, method):
        logging.debug(
            f"Returning the average gene expression for the affected genes across time"
        )
        genes = self.params["affected_genes"]
        logging.debug(f"Evaluating genes: {genes}")

        # now we will get the gene expression for each of the affected genes across time
        # we will return a dictionary of the form:
        # gene -> timepoint -> expression data (list of expression values across cells)
        gene_expression_over_time = {}
        eval_output_path = os.path.join(output_path, self._get_relative_output_path())
        perturbed_data = load_output_file(
            eval_output_path,
            RequiredOutputFiles.PERTURBED_TEST_ANN_DATA_T_TO_T_PLUS_ONE,
        )

        # then let's first un-log normalize the data
        if not is_raw(perturbed_data):
            perturbed_data = undo_log_normalization(perturbed_data)

        tps = sorted(
            list(perturbed_data.obs[ObservationColumns.TIMEPOINT.value].unique())
        )

        test_ann_data = load_test_dataset(output_path)
        if self.perturbation_set.gene_col_name is None:
            gene_names_list = test_ann_data.var_names.tolist()
        else:
            gene_names_list = test_ann_data.var[
                self.perturbation_set.gene_col_name
            ].tolist()

        for gene in genes:
            gene_col_idx = gene_names_list.index(gene)
            gene_expression_over_time[gene] = {}

            # now let's calculate the average expression for this gene across samples
            # per timepoint
            for tp in tps:
                tp_data = perturbed_data[
                    perturbed_data.obs[ObservationColumns.TIMEPOINT.value] == tp
                ]
                gene_expression_over_time[gene][tp] = tp_data[:, gene_col_idx].X

        return gene_expression_over_time
