#!/usr/bin/env python3
"""
Training script for Prophet model with specific parameters.
Modify the CONFIG dictionary below to change parameters.
"""
import numpy as np
# Ensure Prophet sees np.float_ in NumPy 2.x environments
np.float_ = np.float64
from prophet import Prophet
from google.cloud import bigquery, storage
import pandas as pd
import pickle
import io
import json
import os
from datetime import datetime

# ===== CONFIGURATION =====
# Modify these parameters as needed
CONFIG = {
    "output_column": "call_duration",
    "timestamp_column": "call_start_time_est",
    "input_columns": ["dnis", "customer_speaking_duration"],
    "begin_date": "2025-08-01",
    "end_date": "2025-09-01",
    "interval_width": 0.9,
    "scaling_factor": 2.2,
    "project_id": "dev-poc-429118",
    "bq_table": "dev-poc-429118.aa_genai.call-duration-aphw-aug-sep",
    "bucket_name": "aphw-prophet-models",
    "model_blob_name": f"dynamic_models/model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
}
# =========================

def train_prophet_model(config):
    """Train a Prophet model with the given configuration."""
    print("🚀 Starting Prophet model training...")
    print(f"📋 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Initialize clients
    print("\n🔧 Initializing BigQuery and Storage clients...")
    bq_client = bigquery.Client(project=config["project_id"])
    storage_client = storage.Client()
    
    # Build query
    print(f"\n📊 Querying BigQuery table: {config['bq_table']}")
    input_cols_str = ", " + ", ".join(config["input_columns"]) if config["input_columns"] else ""
    query = f"""
    SELECT {config['timestamp_column']}, {config['output_column']}{input_cols_str}
    FROM `{config['bq_table']}`
    WHERE {config['timestamp_column']} BETWEEN '{config['begin_date']}' AND '{config['end_date']}'
    """
    print(f"📝 Query: {query}")
    
    # Execute query
    print("⏳ Executing query...")
    df = bq_client.query(query).to_dataframe()
    print(f"✅ Fetched {len(df)} rows")
    
    if df.empty:
        raise ValueError("No data returned from BigQuery query!")
    
    print(f"\n📋 Data columns: {list(df.columns)}")
    print(f"📊 Data shape: {df.shape}")
    print(f"📊 First few rows:\n{df.head()}")
    
    # Rename for Prophet
    print(f"\n🔄 Renaming columns for Prophet...")
    df = df.rename(columns={
        config["timestamp_column"]: "ds",
        config["output_column"]: "y"
    })
    
    # Convert timestamp
    print("🔄 Converting timestamp column...")
    df["ds"] = pd.to_datetime(df["ds"], utc=False)
    # Ensure Prophet gets tz-naive timestamps
    try:
        if getattr(df["ds"].dt, "tz", None) is not None:
            try:
                df["ds"] = df["ds"].dt.tz_convert(None)
            except Exception:
                df["ds"] = df["ds"].dt.tz_localize(None)
    except Exception:
        df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    
    print(f"✅ Timestamp conversion complete")
    print(f"📊 Date range: {df['ds'].min()} to {df['ds'].max()}")
    
    # Create Prophet model
    print(f"\n🤖 Creating Prophet model...")
    print(f"   interval_width: {config['interval_width']}")
    print(f"   daily_seasonality: True")
    model = Prophet(daily_seasonality=True, interval_width=config["interval_width"])
    
    # Add regressors
    if config["input_columns"]:
        print(f"\n➕ Adding regressors: {config['input_columns']}")
        for col in config["input_columns"]:
            model.add_regressor(col)
            print(f"   ✅ Added regressor: {col}")
    
    # Fit model
    print(f"\n🏋️ Training model...")
    model.fit(df)
    print("✅ Model training complete!")
    
    # Forecast on training timestamps
    print(f"\n🔮 Generating forecast on training data...")
    regressor_cols = config["input_columns"] if config["input_columns"] else []
    pred_df = df[["ds"] + regressor_cols] if regressor_cols else df[["ds"]]
    forecast = model.predict(pred_df)
    print("✅ Forecast complete!")
    
    # Inflate upper bound for anomaly detection
    print(f"\n📈 Adjusting upper bound with scaling factor: {config['scaling_factor']}")
    forecast["yhat_upper"] = forecast["yhat"] + config["scaling_factor"] * (forecast["yhat_upper"] - forecast["yhat"])
    print("✅ Upper bound adjusted!")
    
    # Serialize model
    print(f"\n💾 Serializing model...")
    model_bytes = io.BytesIO()
    pickle.dump(model, model_bytes)
    model_bytes.seek(0)
    print(f"✅ Model serialized ({len(model_bytes.getvalue())} bytes)")
    
    # Upload to GCS
    print(f"\n☁️ Uploading model to GCS...")
    print(f"   Bucket: {config['bucket_name']}")
    print(f"   Blob: {config['model_blob_name']}")
    bucket = storage_client.bucket(config["bucket_name"])
    blob = bucket.blob(config["model_blob_name"])
    blob.upload_from_file(model_bytes)
    print(f"✅ Model uploaded successfully!")
    print(f"🔗 GCS path: gs://{config['bucket_name']}/{config['model_blob_name']}")
    
    # Save model info to JSON file
    model_info = {
        "model_blob_name": config["model_blob_name"],
        "gcs_path": f"gs://{config['bucket_name']}/{config['model_blob_name']}",
        "bucket_name": config["bucket_name"],
        "timestamp_column": config["timestamp_column"],
        "output_column": config["output_column"],
        "input_columns": config["input_columns"],
        "trained_at": datetime.now().isoformat(),
        "training_config": {
            "begin_date": config["begin_date"],
            "end_date": config["end_date"],
            "interval_width": config["interval_width"],
            "scaling_factor": config["scaling_factor"]
        }
    }
    
    json_file = os.path.join(os.path.dirname(__file__), "model_info.json")
    print(f"\n💾 Saving model info to {json_file}...")
    with open(json_file, "w") as f:
        json.dump(model_info, f, indent=2)
    print(f"✅ Model info saved!")
    
    return {
        "model": model,
        "forecast": forecast,
        "gcs_path": f"gs://{config['bucket_name']}/{config['model_blob_name']}",
        "model_blob_name": config["model_blob_name"],
        "model_info": model_info
    }

if __name__ == "__main__":
    try:
        result = train_prophet_model(CONFIG)
        print("\n🎉✨ Training completed successfully! ✨🎉")
        print(f"\n📋 Summary:")
        print(f"   Model blob: {result['model_blob_name']}")
        print(f"   GCS path: {result['gcs_path']}")
        print(f"   Model info saved to: tests/model_info.json")
        print(f"\n💡 Next steps:")
        print(f"   1. Run inference_server.py - it will automatically load the model info")
        print(f"   2. The inference server will read from model_info.json")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

