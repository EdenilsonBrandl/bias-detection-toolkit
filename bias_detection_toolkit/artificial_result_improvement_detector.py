"""
ArtificialResultImprovementDetector

Detecta melhorias artificiais nos resultados dos dados que não possuem causa raiz válida (e.g., devido a mudanças externas como políticas temporárias ou crédito emergencial).
"""

import pandas as pd

class ArtificialResultImprovementDetector:
    def __init__(self, data, result_col, date_col):
        """
        data: DataFrame com os dados
        result_col: coluna de resultados (e.g., pagamentos, aprovações)
        date_col: coluna de datas para analisar mudanças ao longo do tempo
        """
        self.data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        self.result_col = result_col
        self.date_col = date_col

    def analyze(self):
        improvement_report = []
        self.data[self.date_col] = pd.to_datetime(self.data[self.date_col])
        monthly_means = self.data.groupby(self.data[self.date_col].dt.to_period('M'))[self.result_col].mean()
        rolling_mean = monthly_means.rolling(window=3).mean()

        for period, value in monthly_means.iteritems():
            if rolling_mean.loc[period] and abs(value - rolling_mean.loc[period]) > (rolling_mean.std() * 1.5):
                improvement_report.append({
                    "period": str(period),
                    "value": value,
                    "artificial_improvement_suspected": True
                })

        return improvement_report
