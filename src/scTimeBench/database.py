"""
Database manager using sqlite3.

This module provides a simple interface to interact with an SQLite database,
including the setup of tables for storing:
1. Paths to processed datasets.
2. Paths to method checkpoints.
3. Paths to method predictions.
4. Metric results.
"""

import sqlite3
from scTimeBench.config import Config
from pathlib import Path
import csv

from scTimeBench.metrics.method_manager import MethodManager
from scTimeBench.shared.dataset.base import (
    BaseDataset,
    DATASET_PREPROCESSOR_REGISTRY,
    DATASET_REGISTRY,
)
import json
import yaml


class DatabaseManager:
    def __init__(self, config: Config):
        self.conn = sqlite3.connect(config.database_path)
        self._create_tables()
        self.table_names = [
            "method_outputs",
            "datasets",
            "metrics",
            "evals",
            "dataset_metrics",
        ]

    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS method_outputs (
                id INTEGER PRIMARY KEY,
                name TEXT,
                dataset_id INTEGER,
                metadata TEXT,
                path TEXT,
                UNIQUE(name, dataset_id, metadata, path)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY,
                name TEXT,
                dataset_dict TEXT,
                dataset_preprocessors TEXT,
                path TEXT,
                UNIQUE(name, dataset_dict, dataset_preprocessors)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY,
                name TEXT,
                parameters TEXT,
                UNIQUE(name, parameters)
            )
        """
        )
        # does not need to be unique, can have multiple evals for same method_output and metric
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS evals (
                id INTEGER PRIMARY KEY,
                method_output_id INTEGER,
                metric_id INTEGER,
                result TEXT
            )
        """
        )

        # finally, let's have a table for the dataset metrics, such as different classifiers on dataset 1
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS dataset_metrics (
                id INTEGER PRIMARY KEY,
                dataset_id INTEGER,
                metric_id INTEGER,
                result TEXT
            )
        """
        )
        self.conn.commit()

    def get_dataset_id(self, method: MethodManager):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id FROM datasets
            WHERE name = ? AND dataset_dict = ? AND dataset_preprocessors = ?
        """,
            (
                method.dataset.get_name(),
                method.dataset.encode_dataset_dict(),
                method.dataset.encode_preprocessors(),
            ),
        )
        result = cursor.fetchone()
        return result[0] if result else None

    def insert_dataset(self, dataset: BaseDataset):
        # first insert into datasets table if not already there, and get its id
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR IGNORE INTO datasets (name, dataset_dict, dataset_preprocessors, path)
            VALUES (?, ?, ?, ?)
        """,
            (
                dataset.get_name(),
                dataset.encode_dataset_dict(),
                dataset.encode_preprocessors(),
                dataset.get_dataset_dir(),
            ),
        )
        self.conn.commit()

    def insert_method_output(self, method: MethodManager, output_path: str):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO method_outputs (name, dataset_id, metadata, path)
            VALUES (?, ?, ?, ?)
        """,
            (
                method._get_name(),
                self.get_dataset_id(method),
                method._encode_metadata(),
                output_path,
            ),
        )
        self.conn.commit()

    def get_method_output_path(self, method: MethodManager):
        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT path FROM method_outputs
            WHERE name = ? AND dataset_id = ? AND metadata = ?
        """,
            (
                method._get_name(),
                self.get_dataset_id(method),
                method._encode_metadata(),
            ),
        )
        result = cursor.fetchone()
        return result[0] if result else None

    def close(self):
        self.conn.close()

    def _encode_dataset_from_config(self, dataset_config: dict):
        dataset_name = dataset_config.get("name")
        if dataset_name not in DATASET_REGISTRY:
            return None, None

        preprocessors = dataset_config.get("data_preprocessing_steps", [])
        try:
            dataset_preprocessor_instances = [
                DATASET_PREPROCESSOR_REGISTRY[dataset_preprocessor["name"]](
                    dataset_config,
                    **{k: v for k, v in dataset_preprocessor.items() if k != "name"},
                )
                for dataset_preprocessor in preprocessors
            ]
        except KeyError:
            return None, None

        dataset_instance: BaseDataset = DATASET_REGISTRY[dataset_name](
            dataset_config,
            dataset_preprocessor_instances,
            "",
        )
        return (
            dataset_instance.encode_dataset_dict(),
            dataset_instance.encode_preprocessors(),
        )

    def get_dataset_tag_from_id(self, dataset_id):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT name, dataset_dict, dataset_preprocessors FROM datasets
            WHERE id = ?
        """,
            (dataset_id,),
        )
        result = cursor.fetchone()
        if not result:
            return None

        # now let's look for the dataset tag
        name, dataset_dict, dataset_preprocessors = result

        shared_path = Path(__file__).resolve().parent / "shared" / "dataset"
        dataset_files = [
            shared_path / "default_datasets.yaml",
            shared_path / "optional_datasets.yaml",
        ]

        for dataset_file in dataset_files:
            if not dataset_file.exists():
                continue

            with open(dataset_file, "r") as f:
                config = yaml.safe_load(f) or {}

            for dataset in config.get("datasets", []):
                (
                    encoded_dataset_dict,
                    encoded_dataset_preprocessors,
                ) = self._encode_dataset_from_config(dataset)

                if (
                    encoded_dataset_dict is None
                    or encoded_dataset_preprocessors is None
                ):
                    continue

                if (
                    dataset.get("name") == name
                    and encoded_dataset_dict == dataset_dict
                    and encoded_dataset_preprocessors == dataset_preprocessors
                ):
                    return dataset.get("tag")

        parsed_preprocessors = json.loads(dataset_preprocessors)
        preprocessor_names = [
            preprocessor_item.get("name")
            for preprocessor_item in parsed_preprocessors
            if isinstance(preprocessor_item, dict) and preprocessor_item.get("name")
        ]

        if len(preprocessor_names) == 0:
            return name

        return f"{name}-{'-'.join(preprocessor_names)}"

    def print_all(self):
        cursor = self.conn.cursor()
        for table in self.table_names:
            print("-" * 100)
            print(f"Contents of table: {table}")
            if table == "evals":
                # add the method name, dataset name, and metric name for easier reading
                cursor.execute(
                    """
                    SELECT method_outputs.name, datasets.id, metrics.name, evals.result
                    FROM evals
                    JOIN method_outputs ON evals.method_output_id = method_outputs.id
                    JOIN metrics ON evals.metric_id = metrics.id
                    JOIN datasets ON method_outputs.dataset_id = datasets.id
                """
                )
            else:
                cursor.execute(f"SELECT * FROM {table}")

            rows = cursor.fetchall()
            for row in rows:
                if table == "datasets":
                    # call get_dataset_tag_from_id to get the tag as well
                    dataset_id, dataset_dict, dataset_preprocessors, path = (
                        row[0],
                        row[2],
                        row[3],
                        row[4],
                    )
                    dataset_tag = self.get_dataset_tag_from_id(dataset_id)
                    print(
                        f"({dataset_id}, {dataset_tag}, {dataset_dict}, {dataset_preprocessors}, {path})"
                    )
                elif table == "evals":
                    # now print out evals as normal except do the dataset tag instead
                    method_name, dataset_id, metric_name, result = row
                    dataset_tag = self.get_dataset_tag_from_id(dataset_id)
                    print(f"({method_name}, {metric_name}, {dataset_tag}, {result})")
                else:
                    print(row)
        self.conn.commit()

    def return_all(self):
        cursor = self.conn.cursor()
        outputs = ""
        for table in self.table_names:
            outputs += f"Contents of table: {table}\n"
            cursor.execute(f"SELECT * FROM {table}")
            rows = cursor.fetchall()
            for row in rows:
                outputs += f"{row}\n"
        self.conn.commit()
        return outputs

    def graph_sim_to_csv(self, output_csv_path):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT metrics.name, method_outputs.name, datasets.id, datasets.name, evals.result
            FROM evals
            JOIN metrics ON evals.metric_id = metrics.id
            JOIN method_outputs ON evals.method_output_id = method_outputs.id
            JOIN datasets ON method_outputs.dataset_id = datasets.id
            WHERE metrics.name IN ('GraphClassificationReport', 'JaccardSimilarity')
        """
        )
        rows = cursor.fetchall()

        csvfile = open(output_csv_path, "w", newline="")
        writer = csv.writer(csvfile)

        writer.writerow(
            [
                "method",
                "dataset",
                "step_setting",
                "metric",
                "time_type",
                "result",
            ]
        )

        seen_threshold_rows = set()
        metrics_to_retrieve = ["GraphClassificationReport", "JaccardSimilarity"]

        for metric_name, method_name, dataset_id, dataset_name, result_json in rows:
            if metric_name not in metrics_to_retrieve:
                continue

            parsed = json.loads(result_json)

            step_setting = parsed.get("criteria")
            threshold = parsed.get("threshold")
            eval_payload = parsed.get("eval")

            dataset_tag = self.get_dataset_tag_from_id(dataset_id)
            time_type = "Pseudotime" if "pseudo" in dataset_tag.lower() else "Real Time"

            threshold_key = (time_type, dataset_name, method_name, step_setting)
            if threshold_key not in seen_threshold_rows and threshold is not None:
                writer.writerow(
                    [
                        method_name,
                        dataset_name,
                        step_setting,
                        "threshold",
                        time_type,
                        threshold,
                    ]
                )
                seen_threshold_rows.add(threshold_key)

            if metric_name == "GraphClassificationReport":
                if isinstance(eval_payload, str):
                    eval_payload = json.loads(eval_payload)

                to_retrieve = ["f1", "auc_prc"]
                for key in to_retrieve:
                    writer.writerow(
                        [
                            method_name,
                            dataset_name,
                            step_setting,
                            key.upper(),
                            time_type,
                            eval_payload.get(key),
                        ]
                    )

            else:
                writer.writerow(
                    [
                        method_name,
                        dataset_name,
                        step_setting,
                        metric_name,
                        time_type,
                        eval_payload,
                    ]
                )

        csvfile.close()
        print(f"Graph similarity results saved to {output_csv_path}")

    # ** METRIC RELATED FUNCTIONS **
    def has_metric(self, name: str, parameters: str) -> bool:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id FROM metrics
            WHERE name = ? AND parameters = ?
        """,
            (name, parameters),
        )
        result = cursor.fetchone()
        return result is not None

    def insert_metric(self, name: str, parameters: str):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR IGNORE INTO metrics (name, parameters)
            VALUES (?, ?)
        """,
            (name, parameters),
        )
        self.conn.commit()

    # ** EVAL RELATED FUNCTIONS **
    def has_eval(
        self, method: MethodManager, metric_name: str, metric_params: str
    ) -> bool:
        cursor = self.conn.cursor()

        # first we get the method id
        cursor.execute(
            """
            SELECT id FROM method_outputs
            WHERE name = ? AND dataset_id = ? AND metadata = ?
        """,
            (
                method._get_name(),
                self.get_dataset_id(method),
                method._encode_metadata(),
            ),
        )
        method_output_row = cursor.fetchone()
        if method_output_row is None:
            return False
        method_output_id = method_output_row[0]

        # then we get the metric id
        cursor.execute(
            """
            SELECT id FROM metrics
            WHERE name = ? AND parameters = ?
        """,
            (metric_name, metric_params),
        )
        metric_row = cursor.fetchone()
        if metric_row is None:
            return False
        metric_id = metric_row[0]

        # finally we check for the eval
        cursor.execute(
            """
            SELECT id FROM evals
            WHERE method_output_id = ? AND metric_id = ?
        """,
            (method_output_id, metric_id),
        )
        eval_row = cursor.fetchone()
        return eval_row is not None

    def insert_eval(
        self, method: MethodManager, metric_name: str, metric_params: str, result
    ):
        cursor = self.conn.cursor()

        # first we get the method id
        cursor.execute(
            """
            SELECT id FROM method_outputs
            WHERE name = ? AND dataset_id = ? AND metadata = ?
        """,
            (
                method._get_name(),
                self.get_dataset_id(method),
                method._encode_metadata(),
            ),
        )
        method_output_row = cursor.fetchone()
        if method_output_row is None:
            raise ValueError("Method output not found in database.")
        method_output_id = method_output_row[0]

        # then we get the metric id
        cursor.execute(
            """
            SELECT id FROM metrics
            WHERE name = ? AND parameters = ?
        """,
            (metric_name, metric_params),
        )
        metric_row = cursor.fetchone()
        if metric_row is None:
            raise ValueError("Metric not found in database.")
        metric_id = metric_row[0]

        # finally we insert the eval
        cursor.execute(
            """
            INSERT INTO evals (method_output_id, metric_id, result)
            VALUES (?, ?, ?)
        """,
            (method_output_id, metric_id, result),
        )
        self.conn.commit()

    def get_evals_per_metric(self, metric_name: str, metric_params: str):
        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT id FROM metrics
            WHERE name = ? AND parameters = ?
        """,
            (metric_name, metric_params),
        )
        metric_row = cursor.fetchone()
        if metric_row is None:
            raise ValueError("Metric not found in database.")
        metric_id = metric_row[0]

        # finally we get the evals
        cursor.execute(
            """
            SELECT method_output_id, result FROM evals
            WHERE metric_id = ?
        """,
            (metric_id,),
        )
        results = cursor.fetchall()

        outputs = []
        # then, let's fetch all the methods and their parameters as well
        # so we can nicely print this out
        for row in results:
            method_id = row[0]
            cursor.execute(
                """
                SELECT name, datasets.name, datasets.dataset_dict, datasets.dataset_preprocessors, metadata FROM method_outputs
                JOIN datasets ON method_outputs.dataset_id = datasets.id
                WHERE method_outputs.id = ?
            """,
                (method_id,),
            )
            method_row = cursor.fetchone()

            outputs.append(
                {
                    "method_name": method_row[0],
                    "dataset_name": method_row[1],
                    "dataset_dict": json.loads(method_row[2]),
                    "dataset_preprocessors": json.loads(method_row[3]),
                    "metadata": json.loads(method_row[4]),
                    "result": row[1],
                }
            )

        return outputs

    def get_evals_per_method(self, method: MethodManager):
        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT id FROM method_outputs
            WHERE name = ? AND dataset_id = ? AND metadata = ?
        """,
            (
                method._get_name(),
                self.get_dataset_id(method),
                method._encode_metadata(),
            ),
        )
        method_output_row = cursor.fetchone()
        if method_output_row is None:
            raise ValueError("method output not found in database.")
        method_output_id = method_output_row[0]

        # finally we get the evals
        cursor.execute(
            """
            SELECT metric_id, result FROM evals
            WHERE method_output_id = ?
        """,
            (method_output_id,),
        )
        results = cursor.fetchall()

        outputs = []
        # then, let's fetch all the methods and their parameters as well
        # so we can nicely print this out
        for row in results:
            metric_id = row[0]
            cursor.execute(
                """
                SELECT name, parameters FROM metrics
                WHERE id = ?
            """,
                (metric_id,),
            )
            metric_row = cursor.fetchone()

            outputs.append(
                {
                    "metric_name": metric_row[0],
                    "metric_parameters": metric_row[1],
                    "result": row[1],
                }
            )

        return outputs

    # ** DATASET METRICS **
    def insert_dataset_metric(
        self, dataset: BaseDataset, metric_name, metric_params, result
    ):
        cursor = self.conn.cursor()

        # first we get the metric id
        cursor.execute(
            """
            SELECT id FROM metrics
            WHERE name = ? AND parameters = ?
        """,
            (metric_name, metric_params),
        )
        metric_row = cursor.fetchone()
        if metric_row is None:
            raise ValueError("Metric not found in database.")
        metric_id = metric_row[0]

        # then we get the dataset id
        cursor.execute(
            """
            SELECT id FROM datasets
            WHERE name = ? AND dataset_dict = ? AND dataset_preprocessors = ?
        """,
            (
                dataset.get_name(),
                dataset.encode_dataset_dict(),
                dataset.encode_preprocessors(),
            ),
        )
        dataset_row = cursor.fetchone()
        if dataset_row is None:
            raise ValueError("Dataset not found in database.")
        dataset_id = dataset_row[0]

        # finally we insert the dataset metric
        cursor.execute(
            """
            INSERT INTO dataset_metrics (dataset_id, metric_id, result)
            VALUES (?, ?, ?)
        """,
            (dataset_id, metric_id, result),
        )
        self.conn.commit()

    # ** CLEAR TABLES **
    def clear_tables(self):
        cursor = self.conn.cursor()
        for table in self.table_names:
            cursor.execute(f"DELETE FROM {table}")
        self.conn.commit()
