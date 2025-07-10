"""
MaskedFeedbackBiasDetector

Detecta vieses ocultos em feedbacks onde as respostas foram intencionalmente suavizadas ou mascaradas.
"""

import pandas as pd

class MaskedFeedbackBiasDetector:
    def __init__(self, data, feedback_col):
        """
        data: DataFrame com os dados
        feedback_col: coluna que contém os feedbacks numéricos ou categóricos
        """
        self.data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        self.feedback_col = feedback_col

    def analyze(self):
        bias_report = []
        value_counts = self.data[self.feedback_col].value_counts(normalize=True)
        # Detecta concentrações suspeitas no valor mais alto ou médio
        for val, proportion in value_counts.items():
            if proportion > 0.8:
                bias_report.append({
                    "feedback_value": val,
                    "proportion": proportion,
                    "masked_bias_suspected": True
                })
        return bias_report
