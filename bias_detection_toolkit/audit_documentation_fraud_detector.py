import pandas as pd

class AuditDocumentationFraudDetector:
    def __init__(self):
        pass

    def detect_perfect_documents(self, metadata: pd.DataFrame, completeness_threshold=0.99):
        perfect_docs = metadata[metadata['completeness'] >= completeness_threshold].index.tolist()
        return perfect_docs

    def detect_artificial_aging(self, metadata: pd.DataFrame, aging_indicators: list):
        flagged_docs = []
        for doc_id, row in metadata.iterrows():
            if all(row[ind] for ind in aging_indicators):
                flagged_docs.append(doc_id)
        return flagged_docs

    def detect_nonconformity_resolution_for_audit(self, records: pd.DataFrame, resolution_col: str, audit_flag_col: str):
        suspicious = records[(records[resolution_col] == True) & (records[audit_flag_col] == True)]
        return suspicious.index.tolist()