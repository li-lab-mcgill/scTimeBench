import os
import yaml
import json
import hashlib
import logging

from typing import List, Dict
from scTimeBench.shared.utils import (
    is_raw,
    undo_log_normalization,
)


class PerturbationSet:
    """
    This defines the PerturbationSet class, which is used to store a set of perturbations.
    i.e. this has a list of perturbations that will be applied
    to the data at each timepoint.

    The yaml should look like this:
    perturbations:
        - timepoint_idx: ...
          knockin_genes: [...]
          knockout_genes: [...]
          gene_col_name: ...
        ...

    Which we should convert to a dictionary of the form:
    {
        timepoint_idx: {
        "knockin_genes": [...],
        "knockout_genes": [...],
        "gene_col_name": ...
        },
        ...
    }
    so we can easily access the perturbations for each timepoint.
    """

    def __init__(self, perturbations: List[Dict]):
        self.perturbations = {}
        for perturbation in perturbations:
            timepoint_idx = perturbation["timepoint_idx"]
            self.perturbations[timepoint_idx] = {
                "knockin_genes": perturbation.get("knockin_genes", []),
                "knockout_genes": perturbation.get("knockout_genes", []),
                "gene_col_name": perturbation.get("gene_col_name", None),
            }

    def apply_perturbation(self, ann_data, timepoint_idx):
        assert (
            timepoint_idx in self.perturbations
        ), f"Timepoint index {timepoint_idx} not found in perturbations."
        perturb = self.perturbations[timepoint_idx]

        if perturb["gene_col_name"] is None:
            gene_names = list(ann_data.var_names)
        else:
            gene_names = list(ann_data.var[perturb["gene_col_name"]])

        gene_to_index = {gene: idx for idx, gene in enumerate(gene_names)}

        # first let's un-log normalize the data if it's log-normalized
        # and then apply the perturbation, and then log-normalize it back
        is_log_normalized = not is_raw(ann_data)
        if is_log_normalized:
            ann_data = undo_log_normalization(ann_data)

        # track average knockout change per gene for debugging
        knockout_genes = perturb["knockout_genes"]
        knockin_genes = perturb["knockin_genes"]
        for gene in knockout_genes:
            if gene not in gene_to_index:
                continue
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
                continue
            total_expression = ann_data.X[:, gene_to_index[gene]].sum() / ann_data.n_obs
            logging.debug(
                f"Setting knockin gene {gene} with average expression {total_expression}, to {highest_gex}"
            )
            ann_data.X[:, gene_to_index[gene]] = highest_gex

        return ann_data

    def encode(self):
        """
        Encode the perturbation set into a unique hash string that will be saved
        under <method>/perturbations/<perturbation_set_hash>.yaml.
        """
        unique_string = json.dumps(self.perturbations, sort_keys=True)
        return hashlib.sha256(unique_string.encode()).hexdigest()

    def save_file(self, output_dir):
        """
        Save the configuration yaml to the output path
        """
        output_path = os.path.join(output_dir, "perturbation.yaml")
        with open(output_path, "w") as f:
            yaml.dump(self.perturbations, f)

    def perturbation_path(self):
        return os.path.join("perturbations", self.encode())
