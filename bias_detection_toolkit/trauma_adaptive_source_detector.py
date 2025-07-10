"""
Module: trauma_adaptive_source_detector.py

Detects datasets where valid data may have originated from emotionally adaptive,
traumatic or defense-based behaviors, potentially masking true causal roots.

Author: Edenilson Brandl
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis, skew

class TraumaAdaptiveSourceDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.isolation_model = IsolationForest(contamination=0.1, random_state=42)

    def analyze_behavior_distortion(self, df: pd.DataFrame, features: list) -> dict:
        scaled = self.scaler.fit_transform(df[features])
        pcs = self.pca.fit_transform(scaled)

        df_pca = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
        scores = self.isolation_model.fit_predict(df_pca)
        df_result = df.copy()
        df_result["adaptive_flag"] = scores

        # estatísticas gerais
        summary = {
            "skewness": {col: round(skew(df[col]), 4) for col in features},
            "kurtosis": {col: round(kurtosis(df[col]), 4) for col in features},
            "adaptive_data_percent": round((df_result["adaptive_flag"] == -1).sum() / len(df) * 100, 2),
            "suspected_points": df_result[df_result["adaptive_flag"] == -1]
        }

        return summary

# Exemplo de uso
if __name__ == "__main__":
    np.random.seed(42)
    normal = np.random.normal(loc=10, scale=2, size=(80, 2))
    trauma_behavior = np.random.normal(loc=15, scale=0.1, size=(20, 2))  # Comportamento rígido adaptativo

    data = np.vstack([normal, trauma_behavior])
    df = pd.DataFrame(data, columns=['response_time', 'pressure_signal'])

    detector = TraumaAdaptiveSourceDetector()
    results = detector.analyze_behavior_distortion(df, ['response_time', 'pressure_signal'])

    import pprint
    pprint.pprint(results)
