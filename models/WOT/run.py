"""
Waddington-OT (WOT) runner script.

This script trains and evaluates Waddington-OT on an AnnData dataset.
It follows the BaseModel runner structure used across the project.
"""

import os

import numpy as np

from crispy_fishstick.model_utils.model_runner import main
from crispy_fishstick.model_utils.ot_model_runner import BaseOTModel
from crispy_fishstick.shared.constants import ObservationColumns

import anndata
import wot


class WOT(BaseOTModel):
    def _prepare_generate(self, test_ann_data):
        metadata = self.config.get("model", {}).get("metadata", {})
        epsilon = metadata.get("epsilon", 0.05)
        lambda1 = metadata.get("lambda1", 1.0)
        lambda2 = metadata.get("lambda2", 50.0)
        self.ot_model = wot.ot.OTModel(
            test_ann_data,
            day_field=ObservationColumns.TIMEPOINT.value,
            epsilon=epsilon,
            lambda1=lambda1,
            lambda2=lambda2,
        )

    def get_transport_plan(self, ann_data: anndata.AnnData, source_tp, target_tp):
        """
        Solve a single transport problem for transition source_tp -> target_tp.
        """

        def get_transport_map(source_tp: str, target_tp: str, ot_model) -> np.ndarray:
            problems_dir = os.path.join(self.config["output_path"], "problems")
            cache_file = os.path.join(problems_dir, f"{source_tp}_{target_tp}.h5ad")
            if os.path.exists(cache_file):
                transport_map = anndata.read_h5ad(cache_file)
                return transport_map
            else:
                os.makedirs(problems_dir, exist_ok=True)
                tp_anno = ot_model.compute_transport_map(source_tp, target_tp)
                tp_anno.write_h5ad(cache_file)
                return tp_anno

        print(f"Computing transport map from {source_tp} to {target_tp}...")
        tp_anno = get_transport_map(source_tp, target_tp, self.ot_model)
        return tp_anno.X


if __name__ == "__main__":
    main(WOT)
