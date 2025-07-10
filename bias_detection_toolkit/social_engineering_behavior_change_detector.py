import pandas as pd

class SocialEngineeringBehaviorChangeDetector:
    def __init__(self):
        pass

    def detect_behavior_shifts(self, data: pd.DataFrame, subject_col: str, behavior_metric: str, event_dates: list):
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
                    'shift_detected': abs(diff) > 0.1
                })
        return shifts_report