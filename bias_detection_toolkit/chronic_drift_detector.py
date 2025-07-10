"""
Module: chronic_drift_detector.py

Detects slow, cumulative, and non-obvious drift in data over time,
often caused by long-term stress, fatigue or adaptive emotional exhaustion.

Author: Edenilson Brandl
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean
from typing import List

class ChronicDriftDetector:
    def __init__(self, window_size=10, drift_threshold=0.3):
        self.window_size = window_size
        self.drift_threshold = drift_threshold

    def detect_drift(self, df: pd.DataFrame, features: List[str], time_col: str) -> List[dict]:
        df_sorted = df.sort_values(time_col).reset_index(drop=True)
        drift_results = []
        for i in range(self.window_size, len(df_sorted) - self.window_size):
            past_window = df_sorted.loc[i - self.window_size:i - 1, features]
            future_window = df_sorted.loc[i:i + self.window_size - 1, features]

            past_mean = past_window.mean().values
            future_mean = future_window.mean().values

            drift = euclidean(past_mean, future_mean)

            if drift > self.drift_threshold:
                drift_results.append({
                    'index': i,
                    'time': df_sorted.loc[i, time_col],
                    'drift_score': round(drift, 4)
                })

        return drift_results

# Exemplo de uso
if __name__ == "__main__":
    np.random.seed(42)
    # Simula dados com leve mudança de humor ao longo do tempo
    time = pd.date_range(start="2020-01-01", periods=100)
    mood = np.concatenate([
        np.random.normal(5, 0.5, 50),
        np.random.normal(3.5, 0.4, 50)  # Fadiga ao longo do tempo
    ])
    engagement = np.concatenate([
        np.random.normal(6, 0.3, 50),
        np.random.normal(4.2, 0.3, 50)
    ])

    df = pd.DataFrame({
        'date': time,
        'mood': mood,
        'engagement': engagement
    })

    detector = ChronicDriftDetector(window_size=10, drift_threshold=0.5)
    results = detector.detect_drift(df, features=['mood', 'engagement'], time_col='date')

    import pprint
    pprint.pprint(results)
