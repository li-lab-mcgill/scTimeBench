"""
Embedding-based metrics.
"""
from crispy_fishstick.metrics.base import skip_metric
from crispy_fishstick.metrics.embeddings.base import EmbeddingMetrics
from crispy_fishstick.shared.constants import RequiredOutputFiles, ObservationColumns
from crispy_fishstick.shared.utils import load_test_dataset, load_output_file
from crispy_fishstick.trajectory_infer.base import (
    TrajectoryInferenceMethodFactory,
    BaseTrajectoryInferMethod,
)
from crispy_fishstick.trajectory_infer.classifier import Classifier
from crispy_fishstick.trajectory_infer.kNN import kNN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score

import logging
import json
import numpy as np


class TrajectoryEmbeddingMetrics(EmbeddingMetrics):
    def _setup_model_output_requirements(self):
        # ** NOTE: must define the following attributes **
        # where we define the output embedding name
        # as well as the required features and outputs
        self.required_outputs = [
            RequiredOutputFiles.EMBEDDING,
            RequiredOutputFiles.NEXT_TIMEPOINT_EMBEDDING,
        ]


# now let's create the other metrics such as the gini index or classification entropy, etc.
class ClassificationEntropy(TrajectoryEmbeddingMetrics):
    def _setup_trajectory_inference_model(self):
        # by default we use the classifier trajectory inference model
        logging.debug(
            "Setting up trajectory inference model for classification entropy."
        )

        self.trajectory_infer_model: BaseTrajectoryInferMethod = (
            TrajectoryInferenceMethodFactory().get_trajectory_infer_method(
                self.metric_config.get(
                    "trajectory_infer_model", {"name": Classifier.__name__}
                )
            )
        )
        self.params["trajectory_infer_model"] = str(self.trajectory_infer_model)

    def _embedding_eval(self, output_path, dataset):
        test_ann_data = self.trajectory_infer_model._prep_data(output_path)
        traj_infer_path = self.trajectory_infer_model._get_traj_infer_path(output_path)

        # grab the probabilities needed
        probas_and_labels, _ = self.trajectory_infer_model.train_and_predict(
            output_path, test_ann_data, traj_infer_path
        )
        test_probas, _ = probas_and_labels
        next_tp_probas, _, _ = self.trajectory_infer_model.predict_next_tp(
            output_path, test_ann_data, traj_infer_path
        )

        logging.debug(f"Test probabilities: {test_probas}")
        entropy = -np.sum(
            test_probas * np.log(test_probas + 1e-10), axis=1
        )  # avoid log(0)
        logging.debug(f"Average classification entropy: {np.mean(entropy)}")

        predicted_entropy = -np.sum(
            next_tp_probas * np.log(next_tp_probas + 1e-10), axis=1
        )  # avoid log(0)
        logging.debug(
            f"Average predicted classification entropy: {np.mean(predicted_entropy)}"
        )

        num_classes = test_probas.shape[1]
        normalized_entropy = entropy / np.log(num_classes)

        return json.dumps(
            {
                "avg_entropy": np.mean(entropy).item(),
                "std_entropy": np.std(entropy).item(),
                "avg_normalized_entropy": np.mean(normalized_entropy).item(),
                "std_normalized_entropy": np.std(normalized_entropy).item(),
                "num_classes": num_classes,
                "pred_tp_avg_entropy": np.mean(predicted_entropy).item(),
                "pred_tp_avg_normalized_entropy": np.mean(
                    predicted_entropy / np.log(num_classes)
                ).item(),
                "pred_tp_std_entropy": np.std(predicted_entropy).item(),
            }
        )


# now let's create the other metrics such as the gini index or classification entropy, etc.
class EmbeddingGiniIndex(TrajectoryEmbeddingMetrics):
    def _setup_trajectory_inference_model(self):
        # by default we use the classifier trajectory inference model
        logging.debug(
            "Setting up trajectory inference model for classification entropy."
        )

        self.trajectory_infer_model: kNN = (
            TrajectoryInferenceMethodFactory().get_trajectory_infer_method(
                self.metric_config.get("trajectory_infer_model", {"name": kNN.__name__})
            )
        )
        if not isinstance(self.trajectory_infer_model, kNN):
            logging.warning("Gini Index only supports kNN trajectory inference model.")
        self.params["trajectory_infer_model"] = str(self.trajectory_infer_model)

    def _embedding_eval(self, output_path, dataset):
        if not isinstance(self.trajectory_infer_model, kNN):
            logging.warning(
                "Skipping Gini Index evaluation since trajectory inference model is not kNN."
            )
            return

        # Load embeddings from output files
        embeddings = load_output_file(output_path, RequiredOutputFiles.EMBEDDING)
        next_timepoint_embeddings = load_output_file(
            output_path, RequiredOutputFiles.NEXT_TIMEPOINT_EMBEDDING
        )

        # Load test dataset
        test_ann_data = load_test_dataset(output_path)
        cell_types = test_ann_data.obs[ObservationColumns.CELL_TYPE.value].to_numpy()
        timepoints = test_ann_data.obs[ObservationColumns.TIMEPOINT.value]

        # filter next timepoint embeddings to only include the valid timepoints
        valid_timepoints = np.where(timepoints < timepoints.max())[0]
        next_timepoint_embeddings = next_timepoint_embeddings[valid_timepoints]

        knn_graph: NearestNeighbors = self.trajectory_infer_model.get_kNN_graph(
            output_path
        )

        # now let's compute the gini index based on the knn graph
        def calc_gini(indices):
            gini_indices = []
            for neighbor_indices in indices:
                neighbor_cell_types = cell_types[neighbor_indices]
                _, counts = np.unique(neighbor_cell_types, return_counts=True)
                proportions = counts / counts.sum()
                gini_index = 1 - np.sum(proportions**2)
                gini_indices.append(gini_index)
            return np.array(gini_indices)

        _, indices = knn_graph.kneighbors(embeddings)
        gini_indices = calc_gini(indices)

        _, pred_indices = knn_graph.kneighbors(next_timepoint_embeddings)
        gini_pred_indices = calc_gini(pred_indices)

        return json.dumps(
            {
                "avg_gini_index": np.mean(gini_indices).item(),
                "std_gini_index": np.std(gini_indices).item(),
                "pred_tp_avg_gini_index": np.mean(gini_pred_indices).item(),
                "pred_tp_std_gini_index": np.std(gini_pred_indices).item(),
            }
        )


@skip_metric
class ClassifierMetrics(TrajectoryEmbeddingMetrics):
    def _defaults(self):
        return {"f1_average": "weighted", "k_folds": 3}

    def _setup_trajectory_inference_model(self):
        # by default we use the classifier trajectory inference model
        logging.debug(
            "Setting up trajectory inference model for classification entropy."
        )

        self.trajectory_infer_model: BaseTrajectoryInferMethod = (
            TrajectoryInferenceMethodFactory().get_trajectory_infer_method(
                self.metric_config.get(
                    "trajectory_infer_model", {"name": Classifier.__name__}
                )
            )
        )
        self.params["trajectory_infer_model"] = str(self.trajectory_infer_model)

    def _embedding_eval(self, output_path, dataset):
        output = self._classifier_metrics_eval(output_path)
        if self.trajectory_infer_model.uses_gene_expr():
            self.db_manager.insert_dataset_metric(
                dataset, self.__class__.__name__, self._get_param_encoding(), output
            )
            return
        return output

    def _classifier_metrics_eval(self, output_path):
        if self.k_folds == 1:
            (
                pred_probs_and_mapping,
                true_labels,
            ) = self.trajectory_infer_model.train_and_predict(output_path)
            probs, idx_to_cell_map = pred_probs_and_mapping

            # now let's get the predicted labels
            pred_labels = np.array(
                [idx_to_cell_map[np.argmax(proba)] for proba in probs]
            )

            # TODO: future, add more metrics such as precision, recall, f1-score, etc.
            return json.dumps(
                {
                    "accuracy": np.mean(true_labels == pred_labels).item(),
                    "f1_score": f1_score(
                        true_labels, pred_labels, average=self.f1_average
                    ),
                }
            )
        else:
            results = self.trajectory_infer_model.train_and_predict_k_fold_cv(
                output_path, self.k_folds
            )

            k_fold_accuracy = 0.0
            k_fold_f1 = 0.0

            for fold_idx, (pred_probs_and_mapping, true_labels) in enumerate(results):
                probs, idx_to_cell_map = pred_probs_and_mapping

                # now let's get the predicted labels
                pred_labels = np.array(
                    [idx_to_cell_map[np.argmax(proba)] for proba in probs]
                )

                fold_accuracy = np.mean(true_labels == pred_labels).item()
                k_fold_accuracy += fold_accuracy
                fold_f1 = f1_score(true_labels, pred_labels, average=self.f1_average)
                k_fold_f1 += fold_f1

                logging.debug(
                    f"Fold {fold_idx + 1}/{self.k_folds} - Accuracy: {fold_accuracy}, F1 Score: {fold_f1}"
                )

            k_fold_accuracy /= self.k_folds
            k_fold_f1 /= self.k_folds

            return json.dumps(
                {
                    "k_fold_accuracy": k_fold_accuracy,
                    "k_fold_f1_score": k_fold_f1,
                    "k": self.k_folds,
                }
            )
