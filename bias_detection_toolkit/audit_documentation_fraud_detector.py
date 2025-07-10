"""
AuditDocumentationFraudDetector

Detecta possíveis fraudes em documentos criados para validação de processos (e.g., documentação perfeita demais ou forjada para auditorias).
"""

import pandas as pd

class AuditDocumentationFraudDetector:
    def __init__(self, data, doc_quality_col, timestamp_col):
        """
        data: DataFrame com os dados
        doc_quality_col: coluna que avalia a qualidade/documentação
        timestamp_col: coluna com datas/hora de criação ou modificação
        """
        self.data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        self.doc_quality_col = doc_quality_col
        self.timestamp_col = timestamp_col

    def analyze(self):
        fraud_suspects = []
        sudden_spikes = self.data[self.doc_quality_col].diff().abs() > (self.data[self.doc_quality_col].std() * 2)
        for idx, spike in sudden_spikes.items():
            if spike:
                fraud_suspects.append({
                    "index": idx,
                    "timestamp": self.data.loc[idx, self.timestamp_col],
                    "doc_quality": self.data.loc[idx, self.doc_quality_col],
                    "fraud_pattern_suspected": True
                })
        return fraud_suspects
