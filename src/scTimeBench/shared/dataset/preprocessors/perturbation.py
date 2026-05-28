from scTimeBench.shared.dataset.base import BaseDatasetPreprocessor
from scTimeBench.shared.constants import ObservationColumns
from scTimeBench.shared.utils import is_raw

import scanpy as sc

import os
import logging


class PerturbationPreprocessor(BaseDatasetPreprocessor):
    """
    Perturbation preprocessor for trajectory inference metrics.

    This preprocessor will take the original dataset and create a perturbed version of it,
    specified by the cell type, timepoint, and genes to perturb.

    The perturbation will be applied to the test set, while the train set will remain unchanged.
    """

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.splits = True
        self.params = kwargs

    def _parameters(self):
        """
        Return preprocessor-specific parameters.

        The dataset dict will look like the following:
        - name: PerturbationPreprocessor
          cell_type: ...
          timepoint_idx: ...
          knockout_genes: [gene1, gene2, ...]
          knockin_genes: [gene1, gene2, ...]

        Cell type by default is the most common cell type at the first timepoint,
        timepoint by default is the first timepoint.

        knockout_genes and knockin_genes are lists of genes to perturb.
        By default, they are empty lists, meaning no perturbation will be applied.
        """
        return {
            "cell_type": self.params.get("cell_type"),
            "timepoint_idx": self.params.get("timepoint_idx", 0),
            "knockout_genes": self.params.get("knockout_genes", []),
            "knockin_genes": self.params.get("knockin_genes", []),
            "gene_col_name": self.params.get(
                "gene_col_name"
            ),  # optional column name in var to use for gene names, if not specified will use var_names
        }

    def preprocess(self, ann_data, **kwargs):
        """
        Preprocess the dataset by applying perturbations to the test set.
        """

        train_data = ann_data.copy()

        test_data = ann_data.copy()

        timepoints = sorted(test_data.obs[ObservationColumns.TIMEPOINT.value].unique())
        if not timepoints:
            raise ValueError("Cannot apply perturbation to an empty dataset.")

        # get the first timepoint if not specified
        timepoint_idx = self._parameters()["timepoint_idx"]
        if timepoint_idx >= len(timepoints):
            raise ValueError(
                f"Requested perturbation timepoint {timepoint_idx} is not available for the dataset of {len(timepoints)} timepoints. Indexing starts at 0."
            )

        timepoint = timepoints[timepoint_idx]

        # get the most common cell type at the first timepoint if not specified
        cell_type = self._parameters()["cell_type"]
        if cell_type is None:
            first_tp_cell_types = test_data.obs[
                test_data.obs[ObservationColumns.TIMEPOINT.value] == timepoint
            ][ObservationColumns.CELL_TYPE.value]
            cell_type = first_tp_cell_types.value_counts().idxmax()

        selected_mask = (
            test_data.obs[ObservationColumns.TIMEPOINT.value] == timepoint
        ) & (test_data.obs[ObservationColumns.CELL_TYPE.value] == cell_type)

        logging.debug(
            f"Applying perturbation to cell type {cell_type!r} at timepoint {timepoint!r} (index {timepoint_idx}) with {selected_mask.sum()} cells selected."
        )
        if not selected_mask.any():
            raise ValueError(
                f"No cells matched the requested perturbation target: cell_type={cell_type!r}, timepoint={timepoint!r}."
            )

        # let's first filter the test data to only include the selected cells, and then apply the perturbation to those cells
        test_data = test_data[selected_mask].copy()

        if self._parameters()["gene_col_name"] is None:
            gene_names = list(test_data.var_names)
        else:
            gene_names = list(test_data.var[self._parameters()["gene_col_name"]])

        logging.debug(f"Gene names in the dataset: {gene_names[:10]}")
        # let's also show the top n expressed genes
        if self.params.get("top_n_genes_plot_path") is not None:
            logging.getLogger("matplotlib").setLevel(logging.WARNING)
            ax = sc.pl.highest_expr_genes(
                test_data,
                gene_symbols=self._parameters()["gene_col_name"],
                show=False,
            )
            os.makedirs(
                os.path.dirname(self.params["top_n_genes_plot_path"]),
                exist_ok=True,
            )
            ax.get_figure().savefig(self.params["top_n_genes_plot_path"])

        if self.params.get("genes_output_path") is not None:
            os.makedirs(
                os.path.dirname(self.params["genes_output_path"]), exist_ok=True
            )
            with open(self.params["genes_output_path"], "w") as f:
                f.write("\n".join(gene_names))

        gene_to_index = {gene: idx for idx, gene in enumerate(gene_names)}

        knockout_genes = self._parameters()["knockout_genes"]
        knockin_genes = self._parameters()["knockin_genes"]

        # make sure that the specified genes are in the dataset, and are mutually exclusive
        for gene in knockout_genes + knockin_genes:
            if gene not in gene_to_index:
                logging.warning(
                    f"Specified perturbation gene {gene!r} not found in the dataset. It will be ignored."
                )

        if set(knockout_genes) & set(knockin_genes):
            raise ValueError(
                "The same gene cannot be both knocked out and knocked in. Please check your configuration."
            )

        # Important: apply this to the raw data! Fail if raw data is not available
        # since we want to make sure the perturbation is applied before any normalization.
        if not is_raw(test_data) and test_data.raw is None:
            raise ValueError(
                "Data appears to not provide any raw counts data, making perturbation analyses incorrect. "
                "Please provide proper raw counts."
            )

        if not is_raw(test_data):
            logging.debug(
                "Data appears to be normalized. Using raw data for perturbation."
            )
            test_data = test_data.raw.to_adata()

        # track average knockout change per gene for debugging
        for gene in knockout_genes:
            if gene not in gene_to_index:
                continue
            total_expression = (
                test_data.X[:, gene_to_index[gene]].sum() / test_data.n_obs
            )
            logging.debug(
                f"Setting knockout gene {gene} with average expression {total_expression}"
            )
            test_data.X[:, gene_to_index[gene]] = 0

        # for the knockin gene, we take the highest count of all genes
        # in the selected cells and set the knockin gene to that value,
        # to simulate a strong overexpression
        highest_gex = test_data.X.max()
        logging.debug(
            f"Setting knockin genes {knockin_genes} to have expression value {highest_gex} in the perturbed cells."
        )
        for gene in knockin_genes:
            if gene not in gene_to_index:
                continue
            total_expression = (
                test_data.X[:, gene_to_index[gene]].sum() / test_data.n_obs
            )
            logging.debug(
                f"Setting knockin gene {gene} with average expression {total_expression}, to {highest_gex}"
            )
            test_data.X[:, gene_to_index[gene]] = highest_gex

        # finally, save out to the metadata all the timepoints which exist that are after
        # the perturbation timepoint, so that we can use this for evaluation later on
        future_timepoints = sorted([tp for tp in timepoints if tp >= timepoint])
        test_data.uns[ObservationColumns.FUTURE_TIMEPOINTS.value] = future_timepoints

        logging.debug(
            f"Perturbation applied successfully. Test data has: {test_data.n_obs} cells, {test_data.n_vars} genes."
        )
        return train_data, test_data
