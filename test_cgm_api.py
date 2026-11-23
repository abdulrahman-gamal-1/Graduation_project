import os
import pandas as pd
import requests

CSV_PATH = "glucocare_dataset/glucose_readings.csv"
API_BASE_URL = "http://127.0.0.1:8000/api/v1"
TARGET_PATIENT = "p1"
LIMIT_ROWS = 500
INTERNAL_API_KEY = os.getenv("GLUCOCARE_INTERNAL_API_KEY", "")


def main() -> None:
    # Load dataset and filter readings for the target patient
    df = pd.read_csv(CSV_PATH)

    # Detect the glucose column name dynamically (e.g., glucose_mg_dl or value_mg_dl)
    glucose_col = next(
        (col for col in df.columns if "glucose" in col.lower() or "value" in col.lower()),
        None,
    )
    if glucose_col is None:
        raise ValueError(f"Glucose column not found. Available columns: {list(df.columns)}")

    patient_df = df[df["patient_id"] == TARGET_PATIENT].head(LIMIT_ROWS).copy()
    patient_df["timestamp"] = pd.to_datetime(patient_df["timestamp"], errors="coerce")
    patient_df = patient_df.dropna(subset=["timestamp", glucose_col])

    # Build request payload matching the FastAPI schema
    payload = {
        "user_id": TARGET_PATIENT,
        "readings": [
            {
                "timestamp": row["timestamp"].isoformat(),
                "value_mg_dl": float(row[glucose_col]),
            }
            for _, row in patient_df.iterrows()
        ],
        "options": {},
    }

    # Include internal service key header only when configured
    headers = {"X-Internal-API-Key": INTERNAL_API_KEY} if INTERNAL_API_KEY else {}

    # Send request to the summary analytics endpoint
    response = requests.post(
        f"{API_BASE_URL}/analytics/summary",
        json=payload,
        headers=headers,
        timeout=30,
    )
    print("Status Code:", response.status_code)
    print(response.json())


if __name__ == "__main__":
    main()