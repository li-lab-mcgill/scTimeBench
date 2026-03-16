"""
Embedding-based metrics.
"""
from scTimeBench.metrics.base import skip_metric
from scTimeBench.metrics.embeddings.base import EmbeddingMetrics
from scTimeBench.shared.constants import RequiredOutputFiles
from scTimeBench.trajectory_infer.base import (
    TrajectoryInferenceMethodFactory,
    BaseTrajectoryInferMethod,
)
from scTimeBench.trajectory_infer.classifier import Classifier
from sklearn.metrics import f1_score, classification_report

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
                    "trajectory_infer_model",
                    {"name": Classifier.__name__, "model_classifier": True},
                )
            )
        )
        self.params["trajectory_infer_model"] = str(self.trajectory_infer_model)

    def _embedding_eval(self, output_path, dataset):
        # grab the probabilities needed
        probas_and_labels, _ = self.trajectory_infer_model.train_and_predict(
            output_path
        )
        test_probas, _ = probas_and_labels
        next_tp_probas, _ = self.trajectory_infer_model.predict_next_tp(output_path)

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
                self.metric_config.get("trajectory_infer_model", {})
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
                logging.debug(
                    f"Classification Report for Fold {fold_idx + 1}:\n{classification_report(true_labels, pred_labels)}"
                )

            k_fold_accuracy /= self.k_folds
            k_fold_f1 /= self.k_folds

            return json.dumps(
                {
                    "k_fold_accuracy": k_fold_accuracy,
                    "k_fold_f1_score": k_fold_f1,
                }
            )
