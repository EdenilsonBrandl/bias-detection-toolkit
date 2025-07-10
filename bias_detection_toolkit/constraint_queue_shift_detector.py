"""
ConstraintQueueShiftDetector

Detecta variáveis que funcionam como gargalos sucessivos, onde ao resolver uma restrição outra variável surge como bloqueio principal (teoria das restrições dinâmica).
"""

import pandas as pd

class ConstraintQueueShiftDetector:
    def __init__(self, data, constraint_cols):
        """
        data: DataFrame com os dados
        constraint_cols: lista de colunas que podem atuar como restrições no sistema
        """
        self.data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        self.constraint_cols = constraint_cols

    def analyze(self):
        constraint_report = []
        variances = self.data[self.constraint_cols].var().sort_values(ascending=False)
        for idx, (col, var) in enumerate(variances.items()):
            constraint_report.append({
                "rank": idx + 1,
                "constraint_variable": col,
                "variance": var,
                "potential_new_bottleneck": idx > 0  # Marca como novo gargalo se não for o 1º
            })
        return constraint_report
