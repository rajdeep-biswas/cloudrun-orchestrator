#!/usr/bin/env python3
"""
Inference server for Prophet model.
Automatically loads model info from model_info.json (created by train_model.py).
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from google.cloud import storage
import pandas as pd
import pickle
import io
import json
import os

# ===== CONFIGURATION =====
# Load model info from JSON file (created by train_model.py)
MODEL_INFO_FILE = os.path.join(os.path.dirname(__file__), "model_info.json")

def load_model_info():
    """Load model configuration from JSON file."""
    if not os.path.exists(MODEL_INFO_FILE):
        raise FileNotFoundError(
            f"Model info file not found: {MODEL_INFO_FILE}\n"
            f"Please run train_model.py first to generate the model and create this file."
        )
    
    with open(MODEL_INFO_FILE, "r") as f:
        model_info = json.load(f)
    
    return model_info

try:
    MODEL_INFO = load_model_info()
    CONFIG = {
        "bucket_name": MODEL_INFO["bucket_name"],
        "model_blob_name": MODEL_INFO["model_blob_name"],
        "timestamp_column": MODEL_INFO["timestamp_column"],
        "output_column": MODEL_INFO["output_column"],
        "input_columns": MODEL_INFO["input_columns"]
    }
    print(f"✅ Loaded model info from {MODEL_INFO_FILE}")
except FileNotFoundError as e:
    print(f"❌ {e}")
    exit(1)
except KeyError as e:
    print(f"❌ Missing key in model_info.json: {e}")
    exit(1)
# =========================

app = FastAPI(title="Prophet Inference Server")

# Initialize storage client
_storage = storage.Client()
_model = None

def _load_model():
    """Load the Prophet model from GCS."""
    global _model
    if _model is not None:
        return _model
    
    print(f"📥 Loading model from GCS...")
    print(f"   Bucket: {CONFIG['bucket_name']}")
    print(f"   Blob: {CONFIG['model_blob_name']}")
    
    bucket = _storage.bucket(CONFIG["bucket_name"])
    blob = bucket.blob(CONFIG["model_blob_name"])
    data = blob.download_as_bytes()
    _model = pickle.loads(data)
    
    print(f"✅ Model loaded successfully!")
    return _model

@app.get("/")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "prophet-inference",
        "config": {
            "bucket": CONFIG["bucket_name"],
            "model": CONFIG["model_blob_name"],
            "timestamp_column": CONFIG["timestamp_column"],
            "output_column": CONFIG["output_column"],
            "input_columns": CONFIG["input_columns"]
        },
        "model_info": MODEL_INFO.get("training_config", {})
    }

@app.post("/predict")
async def predict(request: Request):
    """
    Prediction endpoint.
    Expects JSON payload with 'instances' array.
    Each instance should have:
    - timestamp_column (call_start_time_est)
    - input_columns (dnis, customer_speaking_duration)
    - Any other fields (e.g., call_id) will be preserved in response
    """
    try:
        payload = await request.json()
        instances = payload.get("instances")
        
        if not instances:
            return JSONResponse(
                {"error": "`instances` array required for prediction"},
                status_code=400
            )
        
        print(f"📥 Received {len(instances)} instances for prediction")
        
        # Convert to DataFrame
        df = pd.DataFrame(instances)
        if df.empty:
            return {"predictions": [], "message": "No instances provided"}
        
        print(f"📊 DataFrame shape: {df.shape}")
        print(f"📋 DataFrame columns: {list(df.columns)}")
        
        # Validate required columns
        if CONFIG["timestamp_column"] not in df.columns:
            return JSONResponse(
                {"error": f"`{CONFIG['timestamp_column']}` field missing in instances"},
                status_code=400
            )
        
        # Check for regressors
        regressors = CONFIG["input_columns"] or []
        missing_regressors = [col for col in regressors if col not in df.columns]
        if missing_regressors:
            return JSONResponse(
                {"error": f"Missing regressor fields: {missing_regressors}"},
                status_code=400
            )
        
        # Preserve original columns (e.g., call_id)
        original_payload_columns = [
            col for col in df.columns 
            if col not in regressors + [CONFIG["timestamp_column"]]
        ]
        print(f"📋 Preserving original columns: {original_payload_columns}")
        
        # Prepare DataFrame for prediction
        df = df.rename(columns={CONFIG["timestamp_column"]: "ds"})
        df["ds"] = pd.to_datetime(df["ds"])
        
        # Load model (cached after first load)
        model = _load_model()
        
        # Prepare prediction DataFrame
        pred_df = df[["ds"] + regressors] if regressors else df[["ds"]]
        print(f"🔮 Making predictions...")
        print(f"   Prediction DataFrame shape: {pred_df.shape}")
        print(f"   Prediction DataFrame columns: {list(pred_df.columns)}")
        
        # Generate forecast
        forecast = model.predict(pred_df)
        print(f"✅ Forecast generated!")
        
        # Extract results
        result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        
        # Calculate anomaly flag if output_column (actual value) is present
        if CONFIG["output_column"] in df.columns:
            actual_values = df[CONFIG["output_column"]].reset_index(drop=True)
            # Anomaly: actual value is outside the prediction interval
            result["is_anomaly"] = (actual_values < result["yhat_lower"]) | (actual_values > result["yhat_upper"])
            print(f"✅ Calculated is_anomaly based on {CONFIG['output_column']}")
        else:
            # If no actual values, set is_anomaly to False
            result["is_anomaly"] = False
            print(f"⚠️ No {CONFIG['output_column']} in input, setting is_anomaly to False")
        
        # Convert datetime to ISO format string
        result["ds"] = result["ds"].apply(lambda x: x.isoformat() if pd.notna(x) else None)
        
        # Merge back original columns
        if original_payload_columns:
            result[original_payload_columns] = df[original_payload_columns].reset_index(drop=True)
        
        print(f"📊 Returning {len(result)} predictions")
        return {"predictions": result.to_dict(orient="records")}
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Prophet Inference Server...")
    print(f"📋 Configuration (loaded from {MODEL_INFO_FILE}):")
    for key, value in CONFIG.items():
        print(f"   {key}: {value}")
    print(f"\n📊 Model Info:")
    print(f"   Trained at: {MODEL_INFO.get('trained_at', 'N/A')}")
    print(f"   GCS path: {MODEL_INFO.get('gcs_path', 'N/A')}")
    print(f"\n🌐 Server will start on http://localhost:8080")
    print(f"📡 Health check: http://localhost:8080/")
    print(f"🔮 Predict endpoint: http://localhost:8080/predict")
    print()
    uvicorn.run(app, host="0.0.0.0", port=8080)

