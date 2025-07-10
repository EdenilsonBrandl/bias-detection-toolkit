"""
SyntheticDataFormulaDetector

Detecta dados sintéticos gerados a partir de fórmulas ou pesos intencionais para simular dados reais.
"""

import pandas as pd
import numpy as np

class SyntheticDataFormulaDetector:
    def __init__(self, data, columns):
        """
        data: DataFrame com os dados
        columns: lista de colunas para verificar relações lineares/fórmulas suspeitas
        """
        self.data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        self.columns = columns

    def analyze(self):
        formula_suspects = []
        corr_matrix = self.data[self.columns].corr().abs()
        for i, col1 in enumerate(self.columns):
            for col2 in self.columns[i+1:]:
                if corr_matrix.loc[col1, col2] > 0.95:
                    formula_suspects.append({
                        "columns": (col1, col2),
                        "correlation": corr_matrix.loc[col1, col2],
                        "synthetic_pattern_suspected": True
                    })
        return formula_suspects
