"""
DownstreamVariableShadowingDetector

Detecta variáveis internas que parecem causar um resultado, mas na verdade são subprodutos de outra variável raiz que gera o efeito.
"""

import pandas as pd
import numpy as np

class DownstreamVariableShadowingDetector:
    def __init__(self, data, target_variable, variable_group):
        """
        data: pandas DataFrame com os dados
        target_variable: string, nome da variável de resultado que queremos entender
        variable_group: lista de strings, variáveis do grupo para analisar shadowing
        """
        self.data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        self.target_variable = target_variable
        self.variable_group = variable_group

    def analyze(self):
        """
        Análise básica para detectar variáveis que "seguem" outras no grupo,
        mas não geram o resultado diretamente.

        Retorna:
          shadowing_report: lista de dicionários com variáveis que parecem shadowing.
        """
        shadowing_report = []

        # Correlações entre variáveis do grupo e com a variável alvo
        corr_matrix = self.data[self.variable_group + [self.target_variable]].corr()

        # Correl com target
        target_corr = corr_matrix[self.target_variable]

        # Para cada variável do grupo, verifica se a correlação com a target é baixa
        # mas a correlação com outra variável do grupo é alta
        for var in self.variable_group:
            corr_with_target = target_corr[var]
            # verifica outras variáveis do grupo com alta correlação com essa var
            high_corr_vars = corr_matrix[var][self.variable_group].drop(var).abs()
            high_corr_vars = high_corr_vars[high_corr_vars > 0.8]

            if abs(corr_with_target) < 0.3 and not high_corr_vars.empty:
                shadowing_report.append({
                    "variable": var,
                    "corr_with_target": corr_with_target,
                    "high_corr_with": list(high_corr_vars.index),
                    "shadowing_suspected": True
                })

        return shadowing_report
