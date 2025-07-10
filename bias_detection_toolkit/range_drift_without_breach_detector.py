"""
RangeDriftWithoutBreachDetector

Detecta alterações nos dados que permanecem dentro dos limites aceitáveis, mas indicam uma possível mudança de comportamento ou contexto.
"""

import pandas as pd

class RangeDriftWithoutBreachDetector:
    def __init__(self, data, monitored_col, min_limit, max_limit):
        """
        data: DataFrame com os dados
        monitored_col: coluna a ser monitorada
        min_limit: limite mínimo aceitável
        max_limit: limite máximo aceitável
        """
        self.data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        self.monitored_col = monitored_col
        self.min_limit = min_limit
        self.max_limit = max_limit

    def analyze(self):
        drift_report = []
        within_range = self.data[(self.data[self.monitored_col] >= self.min_limit) & 
                                 (self.data[self.monitored_col] <= self.max_limit)]
        mean_value = within_range[self.monitored_col].mean()
        if mean_value > (self.max_limit - self.min_limit) * 0.8 + self.min_limit:
            drift_report.append({
                "mean_value": mean_value,
                "status": "Drift detected within range",
                "action_recommended": True
            })
        return drift_report
