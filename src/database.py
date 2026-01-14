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
from config import Config

from metrics.model_manager import ModelManager


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
                result REAL
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
            cursor.execute(f"SELECT * FROM {table}")
            rows = cursor.fetchall()
            for row in rows:
                print(row)
        self.conn.commit()

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
