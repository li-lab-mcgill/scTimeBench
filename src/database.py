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
        for table in ["model_outputs"]:
            print("-" * 100)
            print(f"Contents of table: {table}")
            cursor.execute(f"SELECT * FROM {table}")
            rows = cursor.fetchall()
            for row in rows:
                print(row)
        self.conn.commit()
