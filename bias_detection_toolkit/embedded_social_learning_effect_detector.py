"""
EmbeddedSocialLearningEffectDetector

Detecta mudanças comportamentais nos dados causadas por aprendizado social ou influência externa não explícita nos registros.
"""

import pandas as pd

class EmbeddedSocialLearningEffectDetector:
    def __init__(self, data, behavior_col, group_col):
        """
        data: DataFrame com os dados
        behavior_col: coluna de comportamento observável
        group_col: coluna que identifica grupos sociais
        """
        self.data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        self.behavior_col = behavior_col
        self.group_col = group_col

    def analyze(self):
        learning_effects = []
        group_means = self.data.groupby(self.group_col)[self.behavior_col].mean()
        global_mean = self.data[self.behavior_col].mean()
        for group, mean in group_means.items():
            if abs(mean - global_mean) > global_mean * 0.3:
                learning_effects.append({
                    "group": group,
                    "behavior_mean": mean,
                    "deviation_from_global": abs(mean - global_mean),
                    "embedded_social_learning_detected": True
                })
        return learning_effects
