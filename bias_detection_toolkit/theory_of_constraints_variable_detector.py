import pandas as pd
import numpy as np

class TheoryOfConstraintsVariableDetector:
    def __init__(self):
        pass

    def detect_constraints(self, data: pd.DataFrame, variables: list, threshold: float = 0.1):
        dominance_sequence = []
        detailed_analysis = []

        for idx in data.index:
            row = data.loc[idx, variables]
            max_var = row.idxmax()
            max_val = row.max()
            second_max = row.nlargest(2).iloc[-1]
            if max_val - second_max >= threshold:
                dominance_sequence.append((idx, max_var))
                detailed_analysis.append({
                    'index': idx,
                    'dominant_variable': max_var,
                    'dominant_value': max_val,
                    'second_value': second_max,
                    'difference': max_val - second_max
                })
            else:
                dominance_sequence.append((idx, None))
                detailed_analysis.append({
                    'index': idx,
                    'dominant_variable': None,
                    'dominant_value': max_val,
                    'second_value': second_max,
                    'difference': max_val - second_max
                })

        ordered_vars = []
        for _, var in dominance_sequence:
            if var and (len(ordered_vars) == 0 or ordered_vars[-1] != var):
                ordered_vars.append(var)

        return {
            'constraint_sequence': ordered_vars,
            'detailed_analysis': detailed_analysis
        }