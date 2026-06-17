from scTimeBench.metrics.meta.base import MetaMetric

import pandas as pd

import os


class GRN:
    def __init__(self, grn_path):
        if not os.path.exists(grn_path):
            raise FileNotFoundError(f"GRN file not found at {grn_path}")

        # check the format, if it's tsv, read it as a dataframe
        if grn_path.endswith(".tsv"):
            self.grn_df = pd.read_csv(grn_path, sep="\t")
            self.tf_col = "TF"
            self.gene_col = "Target"
            self.regulation = "Regulation"

    def get_genes_from_tf(self, tf):
        """
        Get the transcription factor (TF) and gene pairs from the GRN dataframe,
        given a specific TF.

        Returns:
            List of genes regulated by the given TF, and direction if available.
        """
        if tf not in self.grn_df[self.tf_col].values:
            return []

        tf_genes = self.grn_df[self.grn_df[self.tf_col] == tf][
            [self.gene_col, self.regulation]
        ]
        return tf_genes.to_dict("records")

    def get_tfs_from_gene(self, gene):
        """
        Get the transcription factor (TF) and gene pairs from the GRN dataframe,
        given a specific gene.

        Returns:
            List of TFs that regulate the given gene, and direction if available.
        """
        if gene not in self.grn_df[self.gene_col].values:
            return []

        gene_tfs = self.grn_df[self.grn_df[self.gene_col] == gene][
            [self.tf_col, self.regulation]
        ]
        return gene_tfs.to_dict("records")


class MetaGRN(MetaMetric):
    """
    Meta submetric for GRN analyses.
    """

    def _defaults(self):
        """The default parameters for grn-meta-based metrics."""
        return {
            "grn_path": "grn_data/resources/trrust_rawdata.human.tsv",
        }

    def _submetric_eval(self, output_path, dataset, method):
        """
        Run the submetric evaluation for GRN analyses.

        Args:
            output_path (str): The path to the output directory.
            dataset (Dataset): The dataset object.
            method (Method): The method object.
        """

        self.grn = GRN(grn_path=self.params.grn_path)

        # now we need to first choose one of the more important genes
        # i.e. a highly variable gene, and then figure out either:
        # 1) the genes that regulate it
        # 2) the genes that it regulates
        # 3) a random gene that isn't related to the gene
        # And we choose 5 genes from either category
        # to up and down regulate and calculate the perturbation

        # To do this, we will do t to t + 1 for all the cells
        # and perturb it in this way instead
