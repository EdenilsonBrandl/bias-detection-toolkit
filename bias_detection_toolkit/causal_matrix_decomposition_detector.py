"""
Module: causal_matrix_decomposition_detector.py

Detects cases where the same output value is caused by structurally different sets of input variables
within a dataset.

Author: Edenilson Brandl
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class CausalMatrixDecompositionDetector:
    def __init__(self, n_clusters=2, variance_threshold=0.9):
        self.n_clusters = n_clusters
        self.variance_threshold = variance_threshold

    def analyze_output_variability(self, df: pd.DataFrame, input_cols: list, output_col: str) -> dict:
        results = {}
        grouped = df.groupby(output_col)

        for value, group in grouped:
            subset = group[input_cols]
            if len(subset) < self.n_clusters * 2:
                continue  # skip small groups

            pca = PCA()
            transformed = pca.fit_transform(subset)
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

            reduced_dims = np.argmax(cumulative_variance >= self.variance_threshold) + 1
            reduced_data = transformed[:, :reduced_dims]

            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            labels = kmeans.fit_predict(reduced_data)
            group_result = {
                'group_size': len(subset),
                'original_output': value,
                'distinct_causal_clusters': len(np.unique(labels)),
                'cluster_labels': labels.tolist(),
                'reduced_dimensions_used': reduced_dims
            }

            if group_result['distinct_causal_clusters'] > 1:
                results[value] = group_result

        return results

# Exemplo de uso
if __name__ == "__main__":
    # Criando um dataset com múltiplas origens para um mesmo output
    df = pd.DataFrame({
        'x1': [1, 1, 2, 2, 5, 5, 10, 10],
        'x2': [2, 2, 3, 3, 0, 0, 5, 5],
        'x3': [0, 0, 0, 0, 7, 7, 10, 10],
        'output': [3, 3, 5, 5, 5, 5, 20, 20]
    })

    detector = CausalMatrixDecompositionDetector()
    results = detector.analyze_output_variability(df, input_cols=['x1', 'x2', 'x3'], output_col='output')

    import pprint
    pprint.pprint(results)
