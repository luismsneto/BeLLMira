import logging
import os
import threading
import pandas as pd
import datetime
from threading import Lock

logger = logging.getLogger(__name__)


class CSVWriter:
    _lock = Lock()

    @staticmethod
    def write(df: pd.DataFrame, path: str, all_columns: list[str]):
        with CSVWriter._lock:
            if not os.path.isfile(path):
                df.to_csv(path, mode='w', header=True, index=False)
                return

            with open(path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()

            if not first_line:
                # File exists but is empty
                df.to_csv(path, mode='w', header=True, index=False)
                return

            existing_header = first_line.split(",")

            if set(all_columns) == set(existing_header):
                # Columns unchanged — efficient append, no header
                df.to_csv(path, mode='a', header=False, index=False)
            else:
                # New columns appeared — reread, merge new row, rewrite once with expanded header
                df_existing = pd.read_csv(path)
                combined = pd.concat([df_existing, df], ignore_index=True)
                combined.to_csv(path, mode='w', header=True, index=False)


class ParquetWriter:
    @staticmethod
    def write(df: pd.DataFrame, path: str):
        if os.path.exists(path):
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_parquet(path, index=False)


class DeltaWriter:
    @staticmethod
    def write(df: pd.DataFrame, path: str):
        from delta import DeltaTable
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        if os.path.exists(path):
            delta_table = DeltaTable.forPath(spark, path)
            spark_df = spark.createDataFrame(df)
            spark_df.write.format("delta").mode("append").save(path)
        else:
            spark_df = spark.createDataFrame(df)
            spark_df.write.format("delta").mode("overwrite").save(path)


class FileWriter:
    @staticmethod
    def write_row(row: dict, file_path: str, columns: list[str] = None, file_type: str = None):
        ext = os.path.splitext(file_path)[1].lower().lstrip('.')
        ftype = file_type or ext

        # Infer existing columns
        existing_columns = []
        if columns is not None:
            existing_columns = list(columns)
        elif ftype == 'csv' and os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                existing_columns = first_line.split(',') if first_line else []
        elif ftype == 'parquet' and os.path.exists(file_path):
            existing_df = pd.read_parquet(file_path)
            existing_columns = list(existing_df.columns)

        # Merge all columns
        all_columns = list(existing_columns)
        for key in row:
            if key not in all_columns:
                all_columns.append(key)

        df_row = pd.DataFrame([row], columns=all_columns)

        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)

        if ftype == 'csv':
            CSVWriter.write(df_row, file_path, all_columns)
        elif ftype == 'parquet':
            ParquetWriter.write(df_row, file_path)
        elif ftype == 'delta':
            DeltaWriter.write(df_row, file_path)
        else:
            raise ValueError(f"Unsupported file type: {ftype}")


class ExperimentLogger:
    """
    Logger to record experiment metrics into a single file.
    """
    def __init__(self, file_path: str, columns: list[str] = None, file_type: str = None):
        self.file_path = file_path
        self.columns = columns
        self.file_type = file_type or os.path.splitext(file_path)[1].lower().lstrip('.')

    def log(self, row: dict):
        try:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row.setdefault("Datetime", now)
            FileWriter.write_row(row, self.file_path, self.columns, self.file_type)
        except Exception as e:
            logger.error("Failed to log row to %s: %s", self.file_path, e, exc_info=True)
