import os
import json
import hashlib
import logging

from typing import Dict
from scTimeBench.shared.utils import (
    is_raw,
    undo_log_normalization,
    log_normalize_to_counts,
)
from scTimeBench.shared.constants import ObservationColumns
import scanpy as sc


class PerturbationSet:
    """
    This defines the PerturbationSet class, which is used to store a set of perturbations.
    i.e. this has a list of perturbations that will be applied
    to the data at each timepoint.

    The yaml should look like this:
    filter_cell_type: ...
    filter_tp: ...
    gene_col_name: ...
    perturbations:
        - timepoint_idx: ...
          knockin_genes: [...]
          knockout_genes: [...]
        ...

    Which we should convert to a dictionary of the form:
    {
        "filter_cell_type": ...,
        "filter_tp": ...,
        "gene_col_name": ...,
        "perturbations": [
            "timepoint_idx": {
            "knockin_genes": [...],
            "knockout_genes": [...],
            },
            ...
        ]
    }
    so we can easily access the perturbations for each timepoint.
    """

    def __init__(self, perturbation_config: Dict):
        self.perturbation_config = perturbation_config
        logging.debug(f"Perturbation config: {self.perturbation_config}")

        self.perturbations = {}
        self.filter_cell_type = perturbation_config.get("filter_cell_type", None)
        self.filter_tp_idx = perturbation_config.get("filter_tp_idx", 0)
        self.gene_col_name = perturbation_config.get("gene_col_name", None)

        for perturbation in self.perturbation_config.get("perturbations", []):
            timepoint_idx = perturbation["timepoint_idx"]
            self.perturbations[timepoint_idx] = {
                "knockin_genes": perturbation.get("knockin_genes", []),
                "knockout_genes": perturbation.get("knockout_genes", []),
            }

    def apply_intial_filter(self, ann_data) -> sc.AnnData:
        """
        Apply the initial filter to the data if specified, and then return
        the filtered data, and the list of all timepoints in the data (after filtering).
        """
        all_tps = sorted(ann_data.obs[ObservationColumns.TIMEPOINT.value].unique())

        assert self.filter_tp_idx < len(
            all_tps
        ), f"Filter timepoint index {self.filter_tp_idx} is out of range for the number of timepoints {len(all_tps)} in the data."

        # filter for all timepoints after the specified timepoint (inclusive)
        filter_tp = all_tps[self.filter_tp_idx]
        tps = [tp for tp in all_tps if tp >= filter_tp]
        print(
            f"Filtering for timepoints {tps} (filtering for timepoint index {self.filter_tp_idx} which corresponds to timepoint {filter_tp})."
        )

        ann_data = ann_data[
            ann_data.obs[ObservationColumns.TIMEPOINT.value] == filter_tp
        ]

        if self.filter_cell_type is not None:
            # for the filtered timepoint, only select the cell type specified
            assert (
                self.filter_cell_type
                in ann_data.obs[ObservationColumns.CELL_TYPE.value].unique()
            ), f"Filter cell type {self.filter_cell_type} not found in the data."
            ann_data = ann_data[
                ann_data.obs[ObservationColumns.CELL_TYPE.value]
                == self.filter_cell_type
            ]

        print(
            f"After filtering for timepoint {filter_tp} and cell type {self.filter_cell_type}, we have {ann_data.n_obs} cells."
        )
        return ann_data, tps

    def apply_perturbation(self, ann_data, timepoint_idx):
        if timepoint_idx not in self.perturbations:
            print(f"No perturbation specified for timepoint {timepoint_idx}.")
            return ann_data

        perturb = self.perturbations[timepoint_idx]
        knockout_genes = perturb["knockout_genes"]
        knockin_genes = perturb["knockin_genes"]
        if len(knockout_genes) == 0 and len(knockin_genes) == 0:
            print(
                f"Knockout and knockin genes are both empty for timepoint {timepoint_idx}, skipping perturbation."
            )
            return ann_data

        if self.gene_col_name is None:
            gene_names = list(ann_data.var_names)
        else:
            gene_names = list(ann_data.var[self.gene_col_name])

        gene_to_index = {gene: idx for idx, gene in enumerate(gene_names)}

        # first let's un-log normalize the data if it's log-normalized
        # and then apply the perturbation, and then log-normalize it back
        is_log_normalized = not is_raw(ann_data)
        if is_log_normalized:
            ann_data = undo_log_normalization(ann_data)

        # track average knockout change per gene for debugging
        for gene in knockout_genes:
            if gene not in gene_to_index:
                # error out
                raise ValueError(
                    f"Knockout gene {gene} not found in the data. Double check your gene_col_name and gene names used."
                )
            total_expression = ann_data.X[:, gene_to_index[gene]].sum() / ann_data.n_obs
            logging.debug(
                f"Setting knockout gene {gene} with average expression {total_expression}"
            )
            ann_data.X[:, gene_to_index[gene]] = 0

        # for the knockin gene, we take the highest count of all genes
        # in the selected cells and set the knockin gene to that value,
        # to simulate a strong overexpression
        highest_gex = ann_data.X.max()
        logging.debug(
            f"Setting knockin genes {knockin_genes} to have expression value {highest_gex} in the perturbed cells."
        )
        for gene in knockin_genes:
            if gene not in gene_to_index:
                raise ValueError(
                    f"Knockin gene {gene} not found in the data. Double check your gene_col_name and gene names used."
                )
            total_expression = ann_data.X[:, gene_to_index[gene]].sum() / ann_data.n_obs
            logging.debug(
                f"Setting knockin gene {gene} with average expression {total_expression}, to {highest_gex}"
            )
            ann_data.X[:, gene_to_index[gene]] = highest_gex

        if is_log_normalized:
            ann_data = log_normalize_to_counts(ann_data)

        return ann_data

    def encode(self):
        """
        Encode the perturbation set into a unique hash string that will be saved
        under <method>/perturbations/<perturbation_set_hash>.yaml.
        """
        unique_string = json.dumps(self.perturbation_config, sort_keys=True)
        return hashlib.sha256(unique_string.encode()).hexdigest()

    def perturbation_path(self):
        return os.path.join("perturbations", self.encode())

    def get_genes(self):
        """
        Get the list of all genes that are perturbed in this perturbation set.
        """
        genes = set()
        for perturb in self.perturbations.values():
            genes.update(perturb["knockin_genes"])
            genes.update(perturb["knockout_genes"])
        return list(genes)
