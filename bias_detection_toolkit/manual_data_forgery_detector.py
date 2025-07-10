"""
ManualDataForgeryDetector

Detecta adulteração manual de dados que se comportam como outliers disfarçados dentro do grupo.
"""

import pandas as pd

class ManualDataForgeryDetector:
    def __init__(self, data, key_columns):
        """
        data: DataFrame com os dados
        key_columns: lista de colunas críticas para verificar padrões suspeitos
        """
        self.data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        self.key_columns = key_columns

    def analyze(self):
        forgery_suspects = []
        for col in self.key_columns:
            freq_table = self.data[col].value_counts(normalize=True)
            highly_repeated_values = freq_table[freq_table > 0.9].index.tolist()
            if highly_repeated_values:
                forgery_suspects.append({
                    "column": col,
                    "repeated_values": highly_repeated_values,
                    "manual_forgery_suspected": True
                })
        return forgery_suspects
