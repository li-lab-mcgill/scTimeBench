"""
Embedding-based trajectory metrics (classification entropy, classifier metrics).
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
    def _setup_method_output_requirements(self):
        self.required_outputs = [
            RequiredOutputFiles.EMBEDDING,
            RequiredOutputFiles.NEXT_TIMEPOINT_EMBEDDING,
            RequiredOutputFiles.NEXT_TIMEPOINT_GENE_EXPRESSION,
        ]


def _classification_entropy_results(test_probas, next_tp_probas) -> dict:
    entropy = -np.sum(test_probas * np.log(test_probas + 1e-10), axis=1)
    predicted_entropy = -np.sum(next_tp_probas * np.log(next_tp_probas + 1e-10), axis=1)
    num_classes = test_probas.shape[1]
    normalized_entropy = entropy / np.log(num_classes)

    return {
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


class _ClassificationEntropyBase(TrajectoryEmbeddingMetrics):
    def _setup_trajectory_inference_model(self):
        logging.debug(
            "Setting up trajectory inference model for %s.",
            self.__class__.__name__,
        )
        traj_config = dict(
            self.metric_config.get(
                "trajectory_infer_model", {"name": Classifier.__name__}
            )
        )
        traj_config["embedding_classifier"] = self._uses_embedding_classifier()
        self.trajectory_infer_model: BaseTrajectoryInferMethod = (
            TrajectoryInferenceMethodFactory().get_trajectory_infer_method(traj_config)
        )
        self.params["trajectory_infer_model"] = str(self.trajectory_infer_model)

    def _uses_embedding_classifier(self) -> bool:
        raise NotImplementedError


class ClassificationEntropy(_ClassificationEntropyBase):
    def _uses_embedding_classifier(self) -> bool:
        return True

    def _embedding_eval(self, output_path, dataset):
        probas_and_labels, _ = self.trajectory_infer_model.train_and_predict(
            output_path
        )
        test_probas, _ = probas_and_labels
        next_tp_probas, _ = self.trajectory_infer_model.predict_next_tp(output_path)

        logging.debug(f"Test probabilities: {test_probas}")
        results = _classification_entropy_results(test_probas, next_tp_probas)
        logging.debug("Average classification entropy: %s", results["avg_entropy"])
        logging.debug(
            "Average predicted classification entropy: %s",
            results["pred_tp_avg_entropy"],
        )
        return json.dumps(results)


class ClassificationEntropyGEx(_ClassificationEntropyBase):
    def _uses_embedding_classifier(self) -> bool:
        return False

    def _setup_method_output_requirements(self):
        self.required_outputs = [
            RequiredOutputFiles.NEXT_TIMEPOINT_GENE_EXPRESSION,
        ]

    def _embedding_eval(self, output_path, dataset):
        probas_and_labels, _ = self.trajectory_infer_model.train_and_predict(
            output_path
        )
        test_probas, _ = probas_and_labels
        next_tp_probas, _ = self.trajectory_infer_model.predict_next_tp(output_path)

        results = _classification_entropy_results(test_probas, next_tp_probas)
        logging.debug("Average GEx classification entropy: %s", results["avg_entropy"])
        logging.debug(
            "Average predicted GEx classification entropy: %s",
            results["pred_tp_avg_entropy"],
        )
        return json.dumps(results)


@skip_metric
class ClassifierMetrics(TrajectoryEmbeddingMetrics):
    def _defaults(self):
        return {"f1_average": "weighted", "k_folds": 3}

    def _setup_trajectory_inference_model(self):
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

            pred_labels = np.array(
                [idx_to_cell_map[np.argmax(proba)] for proba in probs]
            )

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
