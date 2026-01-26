"""
Database manager using sqlite3.

This module provides a simple interface to interact with an SQLite database,
including the setup of tables for storing:
1. Paths to processed datasets.
2. Paths to model checkpoints.
3. Paths to model predictions.
4. Metric results.
"""

import sqlite3
from crispy_fishstick.config import Config

from crispy_fishstick.metrics.model_manager import ModelManager
import json


class DatabaseManager:
    def __init__(self, config: Config):
        self.conn = sqlite3.connect(config.database_path)
        self._create_tables()
        self.table_names = ["model_outputs", "metrics", "evals"]

    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS model_outputs (
                id INTEGER PRIMARY KEY,
                name TEXT,
                dataset_name TEXT,
                dataset_dict TEXT,
                dataset_filters TEXT,
                metadata TEXT,
                path TEXT,
                UNIQUE(name, dataset_name, dataset_dict, dataset_filters, metadata, path)
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
        # does not need to be unique, can have multiple evals for same model_output and metric
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS evals (
                id INTEGER PRIMARY KEY,
                model_output_id INTEGER,
                metric_id INTEGER,
                result TEXT
            )
        """
        )
        self.conn.commit()

    def insert_model_output(self, model: ModelManager, output_path: str):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO model_outputs (name, dataset_name, dataset_dict, dataset_filters, metadata, path)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                model._get_name(),
                model.dataset.get_name(),
                model.dataset.encode_dataset_dict(),
                model.dataset.encode_filters(),
                model._encode_metadata(),
                output_path,
            ),
        )
        self.conn.commit()

    def get_model_output_path(self, model: ModelManager):
        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT path FROM model_outputs
            WHERE name = ? AND dataset_name = ? AND dataset_dict = ? AND dataset_filters = ? AND metadata = ?
        """,
            (
                model._get_name(),
                model.dataset.get_name(),
                model.dataset.encode_dataset_dict(),
                model.dataset.encode_filters(),
                model._encode_metadata(),
            ),
        )
        result = cursor.fetchone()
        return result[0] if result else None

    def close(self):
        self.conn.close()

    def print_all(self):
        cursor = self.conn.cursor()
        for table in self.table_names:
            print("-" * 100)
            print(f"Contents of table: {table}")
            if table == "evals":
                # add the model name, dataset name, and metric name for easier reading
                cursor.execute(
                    """
                    SELECT evals.id, evals.model_output_id, model_outputs.name, model_outputs.dataset_name, evals.metric_id, metrics.name, evals.result
                    FROM evals
                    JOIN model_outputs ON evals.model_output_id = model_outputs.id
                    JOIN metrics ON evals.metric_id = metrics.id
                """
                )
            else:
                cursor.execute(f"SELECT * FROM {table}")

            rows = cursor.fetchall()
            for row in rows:
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
        self, model: ModelManager, metric_name: str, metric_params: str
    ) -> bool:
        cursor = self.conn.cursor()

        # first we get the model id
        cursor.execute(
            """
            SELECT id FROM model_outputs
            WHERE name = ? AND dataset_name = ? AND dataset_dict = ? AND dataset_filters = ? AND metadata = ?
        """,
            (
                model._get_name(),
                model.dataset.get_name(),
                model.dataset.encode_dataset_dict(),
                model.dataset.encode_filters(),
                model._encode_metadata(),
            ),
        )
        model_output_row = cursor.fetchone()
        if model_output_row is None:
            return False
        model_output_id = model_output_row[0]

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
            WHERE model_output_id = ? AND metric_id = ?
        """,
            (model_output_id, metric_id),
        )
        eval_row = cursor.fetchone()
        return eval_row is not None

    def insert_eval(
        self, model: ModelManager, metric_name: str, metric_params: str, result: float
    ):
        cursor = self.conn.cursor()

        # first we get the model id
        cursor.execute(
            """
            SELECT id FROM model_outputs
            WHERE name = ? AND dataset_name = ? AND dataset_dict = ? AND dataset_filters = ? AND metadata = ?
        """,
            (
                model._get_name(),
                model.dataset.get_name(),
                model.dataset.encode_dataset_dict(),
                model.dataset.encode_filters(),
                model._encode_metadata(),
            ),
        )
        model_output_row = cursor.fetchone()
        if model_output_row is None:
            raise ValueError("Model output not found in database.")
        model_output_id = model_output_row[0]

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
            INSERT INTO evals (model_output_id, metric_id, result)
            VALUES (?, ?, ?)
        """,
            (model_output_id, metric_id, result),
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
            SELECT model_output_id, result FROM evals
            WHERE metric_id = ?
        """,
            (metric_id,),
        )
        results = cursor.fetchall()

        outputs = []
        # then, let's fetch all the models and their parameters as well
        # so we can nicely print this out
        for row in results:
            model_id = row[0]
            cursor.execute(
                """
                SELECT name, dataset_name, dataset_dict, dataset_filters, metadata FROM model_outputs
                WHERE id = ?
            """,
                (model_id,),
            )
            model_row = cursor.fetchone()

            outputs.append(
                {
                    "model_name": model_row[0],
                    "dataset_name": model_row[1],
                    "dataset_dict": json.loads(model_row[2]),
                    "dataset_filters": json.loads(model_row[3]),
                    "metadata": json.loads(model_row[4]),
                    "result": row[1],
                }
            )

        return outputs

    def get_evals_per_model(self, model: ModelManager):
        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT id FROM model_outputs
            WHERE name = ? AND dataset_name = ? AND dataset_dict = ? AND dataset_filters = ? AND metadata = ?
        """,
            (
                model._get_name(),
                model.dataset.get_name(),
                model.dataset.encode_dataset_dict(),
                model.dataset.encode_filters(),
                model._encode_metadata(),
            ),
        )
        model_output_row = cursor.fetchone()
        if model_output_row is None:
            raise ValueError("Model output not found in database.")
        model_output_id = model_output_row[0]

        # finally we get the evals
        cursor.execute(
            """
            SELECT metric_id, result FROM evals
            WHERE model_output_id = ?
        """,
            (model_output_id,),
        )
        results = cursor.fetchall()

        outputs = []
        # then, let's fetch all the models and their parameters as well
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

    # ** CLEAR TABLES **
    def clear_tables(self):
        cursor = self.conn.cursor()
        for table in self.table_names:
            cursor.execute(f"DELETE FROM {table}")
        self.conn.commit()
