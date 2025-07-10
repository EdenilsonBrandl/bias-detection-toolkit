"""
Module: spurious_correlation_detector.py

Detects spurious correlations between variable pairs by testing for latent or common influencers
that explain both and invalidate their direct association.

Author: Edenilson Brandl
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from typing import List, Tuple

class SpuriousCorrelationDetector:
    def __init__(self, corr_threshold=0.5, pval_threshold=0.05):
        self.corr_threshold = corr_threshold
        self.pval_threshold = pval_threshold

    def detect_spurious_pairs(self, df: pd.DataFrame, variables: List[str]) -> List[dict]:
        n = len(variables)
        results = []

        for i in range(n):
            for j in range(i+1, n):
                x = variables[i]
                y = variables[j]

                corr_xy, pval_xy = pearsonr(df[x], df[y])
                if abs(corr_xy) < self.corr_threshold or pval_xy > self.pval_threshold:
                    continue  # Not strong enough correlation

                for z in variables:
                    if z in [x, y]:
                        continue
                    # regress x ~ z and y ~ z
                    zx_model = LinearRegression().fit(df[[z]], df[x])
                    zy_model = LinearRegression().fit(df[[z]], df[y])
                    x_pred = zx_model.predict(df[[z]])
                    y_pred = zy_model.predict(df[[z]])

                    corr_pred, _ = pearsonr(x_pred, y_pred)
                    if abs(corr_pred) > 0.8 and abs(corr_pred) >= abs(corr_xy) * 0.9:
                        results.append({
                            'var1': x,
                            'var2': y,
                            'original_corr': round(corr_xy, 4),
                            'common_explainer': z,
                            'explained_corr': round(corr_pred, 4),
                            'spurious': True
                        })
                        break  # found spurious explanation

        return results

# Exemplo de uso
if __name__ == "__main__":
    np.random.seed(42)
    size = 100
    city_size = np.random.normal(10000, 2000, size)
    accidents = city_size * 0.003 + np.random.normal(0, 10, size)
    priests = city_size * 0.0002 + np.random.normal(0, 1, size)

    df = pd.DataFrame({
        'city_size': city_size,
        'accidents': accidents,
        'priests': priests
    })

    detector = SpuriousCorrelationDetector()
    results = detector.detect_spurious_pairs(df, ['city_size', 'accidents', 'priests'])

    import pprint
    pprint.pprint(results)
