import logging
import httpx
import pandas as pd
import asyncio
from datetime import datetime, date
from fastapi import FastAPI, Request
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from google.cloud import bigquery
from google.auth import default
from google.auth.transport.requests import Request as AuthRequest

app = FastAPI(title="Prophet Inference")
router = APIRouter()

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)
logger.propagate = True

def log_info(message: str):
    logger.info(message)

def _to_json_serializable(value):
    """Convert value to JSON-serializable type."""
    if pd.isna(value):
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        # Format with milliseconds (microseconds truncated to milliseconds)
        dt = pd.Timestamp(value) if not isinstance(value, pd.Timestamp) else value
        # Format: YYYY-MM-DDTHH:MM:SS.mmm (3 digits for milliseconds)
        return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]  # Truncate microseconds to milliseconds
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    # For other types, convert to string
    return str(value)

# === Config ===
PROJECT_ID = "dev-poc-429118"
MODEL_REGISTRY_TABLE = "dev-poc-429118.aa_genai.prophet_model_registry"

@router.get("/")
def health_check():
    log_info("🦄✨ Inference service health check — service is up!")
    return {"status": "ok", "service": "inference"}

@router.post("/inference")
async def inference(request: Request):
    """
    Inference endpoint that:
    1) Accepts run_id, begin_date, end_date
    2) Fetches model config from prophet_model_registry
    3) Queries actual data from BigQuery
    4) Transforms data into instances format
    5) POSTs to endpoint_predict_url
    6) Returns predictions
    """
    try:
        log_info("🔮 Starting /inference — fetching model config and running predictions...")
        payload = await request.json()
        
        # Validate required keys
        required_keys = ["run_id", "begin_date", "end_date"]
        missing = [k for k in required_keys if k not in payload]
        if missing:
            log_info(f"❌ Missing required keys: {missing}")
            return JSONResponse({"error": f"Missing keys: {missing}"}, status_code=400)
        
        run_id = payload["run_id"]
        begin_date = payload["begin_date"]
        end_date = payload["end_date"]
        
        # Step 1: Fetch model config from registry
        bq_client = bigquery.Client(project=PROJECT_ID)
        query = f"""
        SELECT 
            bq_table,
            output_column,
            input_columns,
            timestamp_column,
            endpoint_predict_url
        FROM `{MODEL_REGISTRY_TABLE}`
        WHERE run_id = @run_id
        LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("run_id", "STRING", run_id)
            ]
        )
        query_job = bq_client.query(query, job_config=job_config)
        results = query_job.result()
        rows = list(results)
        
        if not rows:
            log_info(f"❌ No model found for run_id: {run_id}")
            return JSONResponse({"error": f"No model found for run_id: {run_id}"}, status_code=404)
        
        model_config = dict(rows[0])
        bq_table = model_config["bq_table"]
        output_column = model_config["output_column"]
        input_columns_csv = model_config["input_columns"]
        timestamp_column = model_config["timestamp_column"]
        endpoint_predict_url = model_config["endpoint_predict_url"]
        input_columns_list = [col.strip() for col in input_columns_csv.split(",")] if input_columns_csv else []
        log_info(f"✅ Step 1: Model config loaded — bq_table={bq_table}, input_columns={input_columns_list}, endpoint={endpoint_predict_url}")
        
        # Step 2: Query actual data from BigQuery
        data_query = f"""
        SELECT *
        FROM `{bq_table}`
        WHERE {timestamp_column} BETWEEN @begin_date AND @end_date
        """
        data_job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("begin_date", "STRING", begin_date),
                bigquery.ScalarQueryParameter("end_date", "STRING", end_date)
            ]
        )
        data_query_job = bq_client.query(data_query, job_config=data_job_config)
        df = data_query_job.to_dataframe()
        
        if df.empty:
            log_info(f"⚠️ No data found for date range: {begin_date} to {end_date}")
            return JSONResponse({"error": "No data found for the specified date range"}, status_code=404)
        
        # Convert timestamp column to timezone-naive (Prophet expects this)
        if timestamp_column in df.columns:
            df[timestamp_column] = pd.to_datetime(df[timestamp_column], format='ISO8601')
            if df[timestamp_column].dt.tz is not None:
                df[timestamp_column] = df[timestamp_column].dt.tz_convert(None)
        
        log_info(f"✅ Step 2: Fetched {len(df)} rows from {bq_table} ({begin_date} to {end_date})")
        
        # Step 3: Transform data into instances format
        instances = []
        for idx, (_, row) in enumerate(df.iterrows(), start=1):
            instance = {}
            
            # Add all non-input columns as-is (call_id, call_start_time_est, call_duration, etc.)
            for col in df.columns:
                if col not in input_columns_list and col != timestamp_column:
                    value = _to_json_serializable(row[col])
                    if value is not None:
                        instance[col] = value
            
            # Add timestamp column (already converted to timezone-naive above)
            if timestamp_column in df.columns:
                ts_value = _to_json_serializable(row[timestamp_column])
                if ts_value is not None:
                    instance[timestamp_column] = ts_value
            
            # Add input columns with their original names (Prophet model expects these)
            for input_col in input_columns_list:
                if input_col in df.columns:
                    value = _to_json_serializable(row[input_col])
                    if value is not None:
                        instance[input_col] = value
            
            instances.append(instance)
        
        log_info(f"✅ Step 3: Transformed {len(instances)} instances")
        
        # Step 4: POST to endpoint_predict_url with Google auth
        request_payload = {"instances": instances}
        # Use the URL as-is - Vertex AI REST API uses :predict as the method name
        predict_url = endpoint_predict_url
        
        # Get Google auth token with Vertex AI scopes
        log_info("🔐 Getting Google authentication token with Vertex AI scopes...")
        credentials, project = default(scopes=[
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/aiplatform"
        ])
        auth_req = AuthRequest()
        credentials.refresh(auth_req)
        auth_token = credentials.token
        log_info(f"✅ Auth token obtained (length: {len(auth_token)}, project: {project})")
        
        import json as json_lib
        payload_str = json_lib.dumps(request_payload)
        payload_size_mb = len(payload_str.encode('utf-8')) / (1024 * 1024)
        payload_size_bytes = len(payload_str.encode('utf-8'))
        
        log_info(f"✅ Step 4: Preparing POST request")
        log_info(f"   🔗 URL: {predict_url}")
        log_info(f"   📦 Payload: {len(instances)} instances")
        log_info(f"   📊 Payload size: {payload_size_bytes:,} bytes ({payload_size_mb:.2f} MB)")
        log_info(f"   ⏱️ Timeout: 300s total, 30s connect, 300s read, 300s write")
        log_info(f"   📋 Headers: Content-Type=application/json, Authorization=Bearer (token length: {len(auth_token)})")
        log_info(f"   🔑 OAuth Scopes: cloud-platform, aiplatform")
        
        # Sample first instance for debugging
        if instances:
            log_info(f"   📝 Sample instance keys: {list(instances[0].keys())}")
            log_info(f"   📝 Sample instance (first 500 chars): {json_lib.dumps(instances[0])[:500]}")
        
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=30.0, read=300.0, write=300.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        ) as client:
            try:
                log_info("⏳ Sending POST request...")
                response = await client.post(
                    predict_url,
                    json=request_payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {auth_token}"
                    },
                    follow_redirects=True
                )
                log_info(f"📥 Response received!")
                log_info(f"   Status Code: {response.status_code}")
                log_info(f"   Response Headers: {dict(response.headers)}")
                log_info(f"   Response Content Length: {len(response.content)} bytes")
                log_info(f"   Response Content (first 1000 chars): {response.text[:1000]}")
                
                response.raise_for_status()
                predictions = response.json()
                log_info(f"✅ Step 4: Received predictions (status {response.status_code})")
                
            except httpx.HTTPStatusError as e:
                log_info(f"❌ HTTP Status Error: {e.response.status_code}")
                log_info(f"   Response Headers: {dict(e.response.headers)}")
                log_info(f"   Response Content Length: {len(e.response.content)} bytes")
                log_info(f"   Response Text (full): {e.response.text}")
                log_info(f"   Request URL: {e.request.url if hasattr(e, 'request') else 'N/A'}")
                log_info(f"   Request Method: {e.request.method if hasattr(e, 'request') else 'N/A'}")
                raise
                
            except httpx.ReadError as e:
                log_info(f"❌ ReadError occurred:")
                log_info(f"   Error Type: {type(e).__name__}")
                log_info(f"   Error Message: {str(e)}")
                log_info(f"   Error Repr: {repr(e)}")
                log_info(f"   Error Args: {e.args if hasattr(e, 'args') else 'N/A'}")
                if hasattr(e, 'request'):
                    log_info(f"   Request URL: {e.request.url}")
                    log_info(f"   Request Method: {e.request.method}")
                    log_info(f"   Request Headers: {dict(e.request.headers) if hasattr(e.request, 'headers') else 'N/A'}")
                # Try to get any response if available
                if hasattr(e, 'response'):
                    log_info(f"   Response Status: {e.response.status_code if hasattr(e.response, 'status_code') else 'N/A'}")
                    log_info(f"   Response Text: {e.response.text if hasattr(e.response, 'text') else 'N/A'}")
                import traceback
                log_info(f"   Full Traceback:\n{traceback.format_exc()}")
                raise
                
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                log_info(f"❌ Connection/Timeout Error:")
                log_info(f"   Error Type: {type(e).__name__}")
                log_info(f"   Error Message: {str(e)}")
                log_info(f"   Error Repr: {repr(e)}")
                if hasattr(e, 'request'):
                    log_info(f"   Request URL: {e.request.url}")
                import traceback
                log_info(f"   Full Traceback:\n{traceback.format_exc()}")
                raise
        
        log_info("🎉 Inference completed successfully!")
        return {
            "status": "success",
            "run_id": run_id,
            "num_instances": len(instances),
            "predictions": predictions
        }
        
    except httpx.ReadError as e:
        log_info(f"❌ ReadError during inference (connection issue):")
        log_info(f"   Error Type: {type(e).__name__}")
        log_info(f"   Error Message: {str(e)}")
        log_info(f"   Error Details: {repr(e)}")
        log_info(f"   💡 This usually means:")
        log_info(f"      - Connection was closed unexpectedly")
        log_info(f"      - Network timeout or interruption")
        log_info(f"      - Server closed the connection")
        log_info(f"      - Payload might be too large")
        log_info(f"   💡 Try checking:")
        log_info(f"      - Endpoint URL correctness")
        log_info(f"      - Network connectivity")
        log_info(f"      - Payload size (may be too large)")
        log_info(f"      - Vertex AI endpoint availability")
        import traceback
        log_info(f"   Traceback:\n{traceback.format_exc()}")
        return JSONResponse(
            {"error": f"Connection error when calling prediction endpoint: {str(e)}. This may be due to network issues, timeout, or the endpoint being unavailable."},
            status_code=500
        )
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        log_info(f"❌ Connection/Timeout error: {str(e)}")
        return JSONResponse(
            {"error": f"Connection/timeout error: {str(e)}"},
            status_code=500
        )
    except httpx.HTTPStatusError as e:
        log_info(f"❌ HTTP error {e.response.status_code}: {e.response.text[:200]}")
        return JSONResponse(
            {"error": f"HTTP error from prediction endpoint: {str(e)}", "status_code": e.response.status_code},
            status_code=500
        )
    except Exception as e:
        log_info(f"❌ Error during /inference: {type(e).__name__} - {str(e)}")
        import traceback
        log_info(f"Traceback: {traceback.format_exc()}")
        return JSONResponse({"error": str(e)}, status_code=500)

app.include_router(router)
