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

import json


class DatabaseManager:
    def __init__(self, config: Config):
        self.conn = sqlite3.connect(config.database_path)
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_datasets (
                id INTEGER PRIMARY KEY,
                dataset_name TEXT,
                filters TEXT,
                path TEXT
            )
        """
        )
        # TODO: Create other tables for model checkpoints, predictions, and metric results
        self.conn.commit()

    def encode_filters(self, filters):
        """
        Encode the list of dataset filters into a JSON string.
        """
        return json.dumps([type(f).__name__ for f in filters])

    def insert_processed_dataset(self, dataset_name, filters, path):
        cursor = self.conn.cursor()

        # filters will be given as a list of dataset filter class objects
        # turn these into a string representation
        filters = self.encode_filters(filters)

        cursor.execute(
            """
            INSERT INTO processed_datasets (dataset_name, filters, path)
            VALUES (?, ?, ?)
        """,
            (dataset_name, filters, path),
        )
        self.conn.commit()

    def get_processed_dataset_path(self, dataset_name, filters):
        cursor = self.conn.cursor()

        # filters will be given as a list of dataset filter class objects
        # turn these into a string representation
        filters = self.encode_filters(filters)

        cursor.execute(
            """
            SELECT path FROM processed_datasets
            WHERE dataset_name = ? AND filters = ?
        """,
            (dataset_name, filters),
        )
        result = cursor.fetchone()
        return result[0] if result else None

    def close(self):
        self.conn.close()

    def print_all(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM processed_datasets")
        rows = cursor.fetchall()
        for row in rows:
            print(row)
