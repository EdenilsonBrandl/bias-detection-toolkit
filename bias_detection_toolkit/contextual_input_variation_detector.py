"""
ContextualInputVariationDetector

Analisa variações no contexto da geração dos dados, detectando situações onde os dados foram manipulados
ou têm distribuição diferenciada por motivos externos (exemplo: provas com dificuldades diferenciadas para alunos específicos).
"""

import pandas as pd
import numpy as np

class ContextualInputVariationDetector:
    def __init__(self, data, context_col, score_col, group_col=None):
        """
        data: DataFrame com os dados
        context_col: coluna que indica o contexto da geração (ex: número da prova, lote, tipo)
        score_col: coluna com os valores a serem analisados (ex: nota, resultado)
        group_col: opcional, coluna para agrupar (ex: aluno, turma)
        """
        self.data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        self.context_col = context_col
        self.score_col = score_col
        self.group_col = group_col

    def analyze(self):
        """
        Identifica variações suspeitas no contexto, como desvios muito maiores que a média, 
        assimetrias ou manipulações direcionadas.

        Retorna um relatório com contextos que apresentaram variações incomuns.
        """
        report = []

        # Se houver agrupamento (ex: por aluno), calcula estatísticas dentro do grupo
        if self.group_col:
            grouped = self.data.groupby([self.context_col, self.group_col])[self.score_col].mean().reset_index()
            context_stats = grouped.groupby(self.context_col)[self.score_col].agg(['mean','std']).reset_index()
        else:
            context_stats = self.data.groupby(self.context_col)[self.score_col].agg(['mean','std']).reset_index()

        # Calcula média geral e desvio padrão dos scores para comparação
        global_mean = self.data[self.score_col].mean()
        global_std = self.data[self.score_col].std()

        for _, row in context_stats.iterrows():
            mean = row['mean']
            std = row['std']

            # Detecta se a média ou o desvio do contexto se afastam muito da média global
            if (mean > global_mean + 2*global_std) or (mean < global_mean - 2*global_std):
                report.append({
                    "context": row[self.context_col],
                    "mean": mean,
                    "std": std,
                    "variation_type": "mean deviation",
                    "suspected": True
                })
            elif std > global_std * 1.5:
                report.append({
                    "context": row[self.context_col],
                    "mean": mean,
                    "std": std,
                    "variation_type": "high std deviation",
                    "suspected": True
                })

        return report
