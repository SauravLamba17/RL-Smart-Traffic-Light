"""
Metrics logger for tracking training progress and benchmark evaluation.
Exports performance data directly to CSV.
"""
import os
import csv
from typing import Dict, List, Any


class MetricsLogger:
    """
    Logs episode metrics to memory and disk.
    """

    def __init__(self, log_filepath: str | None = None):
        self.log_filepath = log_filepath
        self.records: List[Dict[str, Any]] = []

    def log(self, metrics: Dict[str, Any]) -> None:
        """Appends a new metrics record."""
        self.records.append(metrics)

    def save_to_csv(self, filepath: str | None = None) -> str:
        """Writes logged metrics to a CSV file."""
        target_path = filepath or self.log_filepath or "logs/training_log.csv"
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

        if not self.records:
            return target_path

        fieldnames = list(self.records[0].keys())
        with open(target_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.records:
                writer.writerow(record)

        return target_path

    def get_summary(self, last_n: int = 100) -> Dict[str, float]:
        """Calculates mean summary metrics over the last N records."""
        if not self.records:
            return {}

        subset = self.records[-last_n:]
        summary = {}
        for key in subset[0].keys():
            if isinstance(subset[0][key], (int, float)):
                summary[f"mean_{key}"] = sum(r[key] for r in subset) / len(subset)
        return summary
