"""
Module: social_engineering_ignored_data_detector

Detects new data collected during a process that is not used in verifying information or results,
potentially masking important changes or behaviors.

Classes:
    SocialEngineeringIgnoredDataDetector - identifies unused but relevant new data.
"""

import pandas as pd

class SocialEngineeringIgnoredDataDetector:
    def __init__(self):
        pass

    def identify_unused_data(self, collected_data: pd.DataFrame, used_columns: list):
        """
        Identify columns or data fields collected but not used in analysis.

        Parameters:
        - collected_data: pd.DataFrame with all collected data
        - used_columns: list of str, columns actually used in analysis or modeling

        Returns:
        - list of unused columns that may contain important info
        """
        unused_cols = [col for col in collected_data.columns if col not in used_columns]
        return unused_cols

    def flag_potential_impact(self, collected_data: pd.DataFrame, unused_cols: list):
        """
        Perform simple checks on unused data to flag if values differ significantly from used data.

        Parameters:
        - collected_data: pd.DataFrame
        - unused_cols: list of str

        Returns:
        - dict mapping unused columns to basic stats for inspection
        """
        stats = {}
        for col in unused_cols:
            series = collected_data[col]
            stats[col] = {
                'unique_values': series.nunique(),
                'value_counts': series.value_counts().to_dict(),
                'missing_percentage': series.isna().mean()
            }
        return stats

# Example usage
if __name__ == "__main__":
    import pandas as pd

    data = pd.DataFrame({
        'window_size_child': [0.5, 0.6, 0.5, 0.5],
        'window_size_mother': [0.9, 0.85, 0.9, 0.88],
        'child_happy': [True, True, False, True],
        'mother_opinion': ['small windows bad', 'small windows bad', 'small windows good', 'small windows bad']
    })

    used_cols = ['child_happy']
    detector = SocialEngineeringIgnoredDataDetector()
    unused_cols = detector.identify_unused_data(data, used_cols)
    stats = detector.flag_potential_impact(data, unused_cols)

    import pprint
    pprint.pprint({
        'unused_columns': unused_cols,
        'unused_data_stats': stats
    })
