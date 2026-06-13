"""
Measures average gene expression over time for perturbation analyses.
"""
from scTimeBench.metrics.perturbation.base import PerturbationBasedMetrics
from scTimeBench.shared.utils import load_output_file
from scTimeBench.shared.constants import RequiredOutputFiles, ObservationColumns

import logging
import json
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

        # now we will get the average gene expression for each of the perturbed genes across time
        # we will return a dictionary of the form:
        # gene -> timepoint -> average expression
        gene_expression_over_time = {}
        eval_output_path = os.path.join(output_path, self._get_relative_output_path())
        perturbed_data = load_output_file(
            eval_output_path, RequiredOutputFiles.PERTURBED_TEST_ANN_DATA
        )
        tps = sorted(
            list(perturbed_data.obs[ObservationColumns.TIMEPOINT.value].unique())
        )

        for gene in genes:
            if self.perturbation_set.gene_col_name is None:
                gene_col_idx = perturbed_data.var_names.get_loc(gene)
            else:
                gene_col_idx = perturbed_data.var[
                    self.perturbation_set.gene_col_name
                ].get_loc(gene)

            gene_expression_over_time[gene] = {}

            # now let's calculate the average expression for this gene across samples
            # per timepoint
            for tp in tps:
                tp_data = perturbed_data[
                    perturbed_data.obs[ObservationColumns.TIMEPOINT.value] == tp
                ]
                gene_expression_over_time[gene][tp] = tp_data[:, gene_col_idx].X.mean()

        logging.debug(f"Gene expression over time: {gene_expression_over_time}")
        eval = json.dumps(gene_expression_over_time, sort_keys=True)

        self.db_manager.insert_eval(
            method, self.__class__.__name__, self._get_param_encoding(), eval
        )

        return gene_expression_over_time
