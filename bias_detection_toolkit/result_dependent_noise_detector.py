"""
ResultDependentNoiseDetector

Detecta ruídos nos dados que são subprodutos do resultado e não erros de amostragem ou análise.
"""

import pandas as pd

class ResultDependentNoiseDetector:
    def __init__(self, data, target_col, noise_col):
        """
        data: DataFrame com os dados
        target_col: coluna de resultado
        noise_col: coluna suspeita de conter ruído dependente do resultado
        """
        self.data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        self.target_col = target_col
        self.noise_col = noise_col

    def analyze(self):
        noise_report = []
        grouped = self.data.groupby(self.target_col)[self.noise_col].std()
        for target_value, std_dev in grouped.items():
            if std_dev > self.data[self.noise_col].std():
                noise_report.append({
                    "target_value": target_value,
                    "std_dev": std_dev,
                    "result_dependent_noise_suspected": True
                })
        return noise_report
