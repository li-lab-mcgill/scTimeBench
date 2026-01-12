"""
ExampleRandomSampler script.

We use this script to train the ExampleRandomSampler model on a dataset.
Where we simply memorize a random sample from each time point.
"""

# let's add the model_utils path
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from model_utils.parser import main, BaseModel
from shared.constants import ObservationColumns
import random
import numpy as np
import scanpy as sc


class ExampleRandomSampler(BaseModel):
    def train(self, ann_data):
        # Implement training logic here
        self.timepoints = sorted(
            ann_data.obs[ObservationColumns.TIMEPOINT.value].unique()
        )
        self.samples = {}
        for tp in self.timepoints:
            tp_data = ann_data[ann_data.obs[ObservationColumns.TIMEPOINT.value] == tp]
            random_sample = tp_data.X[random.randint(0, tp_data.n_obs - 1)].toarray()
            self.samples[tp] = random_sample

        print(self.samples)

    def generate(self, test_ann_data, expected_output_path):
        """
        Generation logic with interpolation.
        Returns an AnnData object containing the generated samples.
        """
        generated_list = []

        # Efficiently get the timepoint column
        tp_column = ObservationColumns.TIMEPOINT.value
        timepoints = test_ann_data.obs[tp_column]

        for tp in timepoints:
            if tp in self.timepoints:
                generated_list.append(self.samples[tp])
            else:
                # Linear Interpolation Logic
                lower_tps = [t for t in self.timepoints if t < tp]
                upper_tps = [t for t in self.timepoints if t > tp]

                if lower_tps and upper_tps:
                    l_tp, u_tp = max(lower_tps), min(upper_tps)
                    weight = (tp - l_tp) / (u_tp - l_tp)
                    interpolated = (1 - weight) * self.samples[
                        l_tp
                    ] + weight * self.samples[u_tp]
                    generated_list.append(interpolated)
                elif lower_tps:
                    generated_list.append(self.samples[max(lower_tps)])
                elif upper_tps:
                    generated_list.append(self.samples[min(upper_tps)])

        # 2. Construct the new AnnData object
        # We copy the .obs and .var to keep cell_ids and gene names intact
        new_adata = sc.AnnData(
            X=np.vstack(generated_list),
            obs=test_ann_data.obs.copy(),
            var=test_ann_data.var.copy(),
        )

        new_adata.write_h5ad(
            expected_output_path,
        )


if __name__ == "__main__":
    main(ExampleRandomSampler)
