"""
Module: social_engineering_behavior_change_detector

Detects changes in behavior or responses that are influenced by external social or political factors,
which do not appear as root causes in the dataset.

Classes:
    SocialEngineeringBehaviorChangeDetector - detects behavior shifts linked to external influences.
"""

import pandas as pd
import numpy as np

class SocialEngineeringBehaviorChangeDetector:
    def __init__(self):
        pass

    def detect_behavior_shifts(self, data: pd.DataFrame, subject_col: str, behavior_metric: str, event_dates: list):
        """
        Detect shifts in behavior metric around known event dates.

        Parameters:
        - data: pd.DataFrame indexed by date or time with subject_col and behavior_metric
        - subject_col: str, identifies the subject/person
        - behavior_metric: str, column of behavior measure
        - event_dates: list of datetime objects indicating external events

        Returns:
        - dict keyed by subject with detected shifts info
        """
        shifts_report = {}

        for subject, group in data.groupby(subject_col):
            group = group.sort_index()
            shifts_report[subject] = []
            for event_date in event_dates:
                pre_event = group[group.index < event_date][behavior_metric].mean()
                post_event = group[group.index >= event_date][behavior_metric].mean()
                if pd.isna(pre_event) or pd.isna(post_event):
                    continue
                diff = post_event - pre_event
                shifts_report[subject].append({
                    'event_date': event_date,
                    'pre_event_mean': pre_event,
                    'post_event_mean': post_event,
                    'difference': diff,
                    'shift_detected': abs(diff) > 0.1  # Threshold for meaningful shift
                })
        return shifts_report

# Example usage
if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    dates = pd.date_range(start='2023-01-01', periods=10)
    data = pd.DataFrame({
        'date': dates.tolist() * 2,
        'subject': ['person1']*10 + ['person2']*10,
        'behavior_score': [0.5,0.5,0.5,0.5,0.5,0.8,0.9,0.85,0.8,0.8] +
                          [0.3,0.3,0.3,0.4,0.3,0.35,0.3,0.3,0.3,0.3]
    }).set_index('date')

    detector = SocialEngineeringBehaviorChangeDetector()
    event_dates = [pd.Timestamp('2023-01-06')]
    report = detector.detect_behavior_shifts(data, 'subject', 'behavior_score', event_dates)

    import pprint
    pprint.pprint(report)
