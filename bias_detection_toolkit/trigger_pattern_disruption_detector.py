"""
Module: trigger_pattern_disruption_detector.py

Detects patterns of sudden behavior disruption in time-series data potentially triggered by
external symbolic or emotional stimuli (triggers).

Author: Edenilson Brandl
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import List, Dict

class TriggerPatternDisruptionDetector:
    def __init__(self, anomaly_sensitivity=0.15):
        self.anomaly_sensitivity = anomaly_sensitivity

    def detect_disruptions(self, df: pd.DataFrame, target_col: str, time_col: str, context_cols: List[str]) -> Dict:
        df_sorted = df.sort_values(time_col).reset_index(drop=True)
        rolling_mean = df_sorted[target_col].rolling(window=5, center=True).mean()
        residuals = df_sorted[target_col] - rolling_mean

        df_sorted['residual'] = residuals
        model = IsolationForest(contamination=self.anomaly_sensitivity, random_state=42)
        anomaly_labels = model.fit_predict(residuals.fillna(0).values.reshape(-1, 1))
        df_sorted["anomaly"] = anomaly_labels

        # Possíveis gatilhos contextuais
        trigger_counts = {}
        anomaly_indices = df_sorted[df_sorted["anomaly"] == -1].index

        for col in context_cols:
            val_counts = df_sorted.loc[anomaly_indices, col].value_counts()
            for val, count in val_counts.items():
                if count > 1:
                    key = f"{col}:{val}"
                    trigger_counts[key] = trigger_counts.get(key, 0) + count

        return {
            "anomalies": df_sorted[df_sorted["anomaly"] == -1][[time_col, target_col] + context_cols],
            "potential_triggers": trigger_counts,
            "full_df": df_sorted
        }

# Exemplo de uso
if __name__ == "__main__":
    np.random.seed(42)
    time = pd.date_range("2023-01-01", periods=100)
    behavior = np.random.normal(5, 0.2, 100)
    # Um comportamento agressivo repentino
    behavior[48:52] = np.random.normal(10, 0.1, 4)

    context = ['none'] * 100
    context[48:52] = ['paddle'] * 4  # Gatilho simbólico

    df = pd.DataFrame({
        'timestamp': time,
        'aggression_level': behavior,
        'kitchen_object': context
    })

    detector = TriggerPatternDisruptionDetector()
    result = detector.detect_disruptions(
        df, target_col='aggression_level', time_col='timestamp', context_cols=['kitchen_object']
    )

    import pprint
    pprint.pprint(result["potential_triggers"])
