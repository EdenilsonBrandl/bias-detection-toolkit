import pandas as pd

class SocialEngineeringIgnoredDataDetector:
    def __init__(self):
        pass

    def identify_unused_data(self, collected_data: pd.DataFrame, used_columns: list):
        unused_cols = [col for col in collected_data.columns if col not in used_columns]
        return unused_cols

    def flag_potential_impact(self, collected_data: pd.DataFrame, unused_cols: list):
        stats = {}
        for col in unused_cols:
            series = collected_data[col]
            stats[col] = {
                'unique_values': series.nunique(),
                'value_counts': series.value_counts().to_dict(),
                'missing_percentage': series.isna().mean()
            }
        return stats