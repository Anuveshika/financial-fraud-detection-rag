# app/data_loader.py

import pandas as pd

REQUIRED_COLUMNS = {
    "transaction_id",
    "amount",
    "location",
    "merchant",
    "description",
}

def load_transactions(file_path: str) -> list[str]:
    try:
        # Primary attempt (most stable for Excel CSVs)
        df = pd.read_csv(
            file_path,
            encoding="utf-8-sig",   # ✅ handles BOM
            sep=",",
            engine="c"              # ✅ avoid python parser issues
        )
    except Exception:
        # Fallback (auto-detect delimiter if comma fails)
        df = pd.read_csv(
            file_path,
            encoding="utf-8-sig",
            sep=None,
            engine="python"
        )

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in CSV: {missing}. "
            f"Found columns: {df.columns.tolist()}"
        )

    documents = []
    for _, row in df.iterrows():
        text = (
            f"Transaction ID: {row['transaction_id']}, "
            f"Amount: {row['amount']}, "
            f"Location: {row['location']}, "
            f"Merchant: {row['merchant']}, "
            f"Description: {row['description']}"
        )
        documents.append(text)

    return documents
