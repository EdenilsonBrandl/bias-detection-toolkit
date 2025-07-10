"""
ExplainableOutlierWithHiddenCauseDetector

Detecta outliers que parecem legítimos mas cuja causa raiz está oculta no conjunto de dados.
"""

import pandas as pd

class ExplainableOutlierWithHiddenCauseDetector:
    def __init__(self, data, key_columns):
        """
        data: DataFrame com os dados
        key_columns: lista de colunas que são críticas para o resultado
        """
        self.data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        self.key_columns = key_columns

    def analyze(self):
        outlier_report = []
        for col in self.key_columns:
            q1 = self.data[col].quantile(0.25)
            q3 = self.data[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]
            for _, row in outliers.iterrows():
                outlier_report.append({
                    "column": col,
                    "value": row[col],
                    "hidden_cause_suspected": True
                })
        return outlier_report
