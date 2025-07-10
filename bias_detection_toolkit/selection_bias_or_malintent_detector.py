"""
SelectionBiasOrMalintentDetector

Detecta seleção de dados enviesada ou com intenção maliciosa no processo de geração do conjunto de dados.
"""

import pandas as pd

class SelectionBiasOrMalintentDetector:
    def __init__(self, data, group_col, metric_col):
        """
        data: DataFrame com os dados
        group_col: coluna usada para agrupar os dados
        metric_col: coluna métrica para verificar desvios entre os grupos
        """
        self.data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        self.group_col = group_col
        self.metric_col = metric_col

    def analyze(self):
        bias_report = []
        group_means = self.data.groupby(self.group_col)[self.metric_col].mean()
        global_mean = self.data[self.metric_col].mean()
        for group, mean in group_means.items():
            deviation = abs(mean - global_mean) / global_mean
            if deviation > 0.5:  # grande desvio indica possível viés de seleção
                bias_report.append({
                    "group": group,
                    "mean": mean,
                    "deviation_from_global": deviation,
                    "selection_bias_or_malintent": True
                })
        return bias_report
