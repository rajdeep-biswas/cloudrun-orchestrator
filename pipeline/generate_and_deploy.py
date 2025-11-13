import os
import logging
import tempfile
import zipfile
from datetime import datetime
import uuid
from fastapi import FastAPI, Request
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from jinja2 import Environment, FileSystemLoader
from google.cloud import bigquery
from google.cloud import storage
from google.cloud import aiplatform
from google.cloud.aiplatform import models
from google.cloud.devtools import cloudbuild_v1

from google.oauth2 import service_account
import google.auth
import os


app = FastAPI(title="Prophet Orchestrator")
router = APIRouter()

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# Log through Uvicorn's error logger so messages appear in console
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)
logger.propagate = True

def log_info(message: str):
    logger.info(message)

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.normpath(os.path.join(_MODULE_DIR, "..", "templates"))
env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))

# === Config ===
PROJECT_ID = "dev-poc-429118"
REGION = "us-central1"
GCS_TEMP_BUCKET = "prophet-temp-bucket"  # pre-created bucket for temporary workdirs
MODEL_REGISTRY_TABLE = "dev-poc-429118.aa_genai.prophet_model_registry"
TIMESTAMP_COLUMN = "call_start_time_est"
PROJECT_ID = "dev-poc-429118"
BQ_TABLE = "dev-poc-429118.aa_genai.call-duration-aphw-aug-sep"
BUCKET_NAME = "aphw-prophet-models"

# Override default credentials with proper scoped service account
key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
if key_path:
    credentials = service_account.Credentials.from_service_account_file(
        key_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    # Force refresh to ensure we get an access token
    import google.auth.transport.requests
    credentials.refresh(google.auth.transport.requests.Request())
    log_info("✅ Loaded service account credentials with access token.")
else:
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    log_info("✅ Using default application credentials.")

# ===== Internal helpers (non-endpoint) =====
def _generate_artifacts_from_payload(payload: dict):
    """
    Internal utility that performs the same work as /generate and returns:
    - workdir, generated_files, gcs_source_uri, model_blob_name
    """
    output_col = payload["output_column"]
    input_cols = payload["input_columns"]
    timestamp_col = payload["timestamp_column"]
    begin_date = payload["begin_date"]
    end_date = payload["end_date"]
    interval_width = payload["interval_width"]
    scaling_factor = payload["scaling_factor"]
    project_id = payload["project_id"]
    bq_table = payload["bq_table"]
    bucket_name = payload["bucket_name"]

    # Create timestamped working directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workdir = tempfile.mkdtemp(prefix=f"build_{timestamp}_")
    model_blob_name = f"dynamic_models/model_{timestamp}.pkl"
    log_info(f"📂 [helper] Working directory created: {workdir} ✅")

    # --- Render train.py ---
    train_template = env.get_template("train_template.py.j2")
    rendered_train = train_template.render(
        input_columns=input_cols,
        output_column=output_col,
        timestamp_column=timestamp_col,
        begin_date=begin_date,
        end_date=end_date,
        interval_width=interval_width,
        scaling_factor=scaling_factor,
        project_id=project_id,
        bq_table=bq_table,
        bucket_name=bucket_name,
        model_blob_name=model_blob_name,
    )
    train_path = os.path.join(workdir, "train.py")
    with open(train_path, "w") as f:
        f.write(rendered_train)
    log_info("📝 [helper] Rendered and wrote train.py ✅")

    # --- Copy Dockerfile ---
    dockerfile_template_path = os.path.join(TEMPLATE_DIR, "dockerfile_template.j2")
    dockerfile_target = os.path.join(workdir, "Dockerfile")
    with open(dockerfile_template_path) as src, open(dockerfile_target, "w") as dst:
        dst.write(src.read())
    log_info("📦 [helper] Dockerfile prepared for training image ✅")

    # --- Copy training requirements ---
    train_requirements_path = os.path.join(TEMPLATE_DIR, "requirements_train.txt")
    target_requirements = os.path.join(workdir, "requirements.txt")
    with open(train_requirements_path) as src, open(target_requirements, "w") as dst:
        dst.write(src.read())
    log_info("📚 [helper] Training requirements.txt written ✅")

    # --- Zip workdir ---
    zip_filename = f"/tmp/workdir_{timestamp}.zip"
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(workdir):
            for file in files:
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, workdir)
                zipf.write(full_path, arcname=relative_path)
    log_info(f"🗜️ [helper] Zipped workdir to {zip_filename} ✅")

    log_info("🔐 [helper] Credentials:")
    log_info(f"Type: {type(credentials)}")
    log_info(f"Service Account Email: {credentials.service_account_email}")
    # log_info(f"Token: {credentials.token}")

    # --- Upload to GCS ---
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(GCS_TEMP_BUCKET)
    gcs_object = f"workdir_{timestamp}.zip"
    blob = bucket.blob(gcs_object)
    blob.upload_from_filename(zip_filename)
    gcs_source_uri = f"gs://{GCS_TEMP_BUCKET}/{gcs_object}"
    log_info(f"☁️ [helper] Uploaded zip to GCS: {gcs_source_uri} ✅")

    return {
        "workdir": workdir,
        "generated_files": ["train.py", "Dockerfile", "requirements.txt"],
        "gcs_source_uri": gcs_source_uri,
        "model_blob_name": model_blob_name,
        "timestamp": timestamp,
    }

def _store_run_metadata(payload: dict, gen_artifacts: dict, *, gcs_infer_source_uri: str, train_image_uri: str,
                        inference_image_uri: str, model_display_name: str, model_id: str,
                        endpoint_display_name: str, endpoint_id: str, run_id: str):
    """
    Persist the orchestrated run metadata into BigQuery.
    """
    bq_client = bigquery.Client(project=PROJECT_ID)
    created_at = datetime.utcnow().isoformat() + "Z"
    train_workdir = gen_artifacts["workdir"]
    train_gcs_source_uri = gen_artifacts["gcs_source_uri"]
    model_blob_name = gen_artifacts["model_blob_name"]
    
    # Convert input_columns list to CSV string
    input_cols_list = payload.get("input_columns", [])
    input_columns_csv = ",".join(input_cols_list) if isinstance(input_cols_list, list) else str(input_cols_list)

    row = {
        "run_id": run_id,
        "project_id": payload.get("project_id", PROJECT_ID),
        "region": REGION,
        "created_at": created_at,
        "created_by": payload.get("created_by"),
        "configuration_name": payload.get("configuration_name"),
        "model_description": payload.get("model_description"),
        "status": "success",
        "error": None,
        "bq_table": payload.get("bq_table"),
        "output_column": payload.get("output_column"),
        "input_columns": input_columns_csv,  # CSV string
        "timestamp_column": payload.get("timestamp_column"),
        "begin_date": payload.get("begin_date"),
        "end_date": payload.get("end_date"),
        "interval_width": payload.get("interval_width"),
        "scaling_factor": payload.get("scaling_factor"),
        "train_workdir": train_workdir,
        "train_gcs_source_uri": train_gcs_source_uri,
        "model_blob_bucket": payload.get("bucket_name"),
        "model_blob_name": model_blob_name,
        "infer_gcs_source_uri": gcs_infer_source_uri,
        "train_image_uri": train_image_uri,
        "inference_image_uri": inference_image_uri,
        "model_display_name": model_display_name,
        "model_id": model_id,
        "endpoint_display_name": endpoint_display_name,
        "endpoint_id": endpoint_id,
        "endpoint_region": REGION,
        "endpoint_predict_url": f"https://{REGION}-aiplatform.googleapis.com/v1/{endpoint_id}:predict",
        "health_route": "/",
        "predict_route": "/predict",
    }

    errors = bq_client.insert_rows_json(MODEL_REGISTRY_TABLE, [row])
    if errors:
        raise RuntimeError(f"BigQuery insert errors: {errors}")
    log_info("📊 Metadata stored in BigQuery ✅")

# ================== API ==================
@router.get("/")
def health_check():
    log_info("🦄✨ Health check hit — service is up!")
    return {"status": "ok", "phase": "full-pipeline"}

@router.post("/generate_and_deploy")
async def generate_and_deploy(request: Request):
    """
    Single-call endpoint that:
    1) Generates training artifacts and uploads source to GCS
    2) Builds, trains, and deploys the model using Cloud Build and Vertex AI
    Returns combined metadata, removing any need to pass outputs between endpoints.
    """
    try:
        log_info("🦄 Starting /generate_and_deploy — full pipeline kickoff...")
        payload = await request.json()
        log_info("📥 Received payload for full pipeline ✅")

        payload["timestamp_column"] = TIMESTAMP_COLUMN
        payload["project_id"] = PROJECT_ID
        payload["bq_table"] = BQ_TABLE
        payload["bucket_name"] = BUCKET_NAME

        # Validate using the superset required for generation (build uses same fields later)
        required_keys = [
            "output_column",
            "input_columns",
            "timestamp_column",
            "begin_date",
            "end_date",
            "interval_width",
            "scaling_factor",
            "project_id",
            "bq_table",
            "bucket_name",
            "configuration_name",
            "model_description",
        ]
        missing = [k for k in required_keys if k not in payload]
        if missing:
            log_info(f"❌ Missing required keys: {missing}")
            return JSONResponse({"error": f"Missing keys: {missing}"}, status_code=400)

        # Phase 1 (internal): generate artifacts
        gen = _generate_artifacts_from_payload(payload)
        gcs_source_uri = gen["gcs_source_uri"]
        model_blob_name = gen["model_blob_name"]

        # Gather other fields for Phase 2
        output_col = payload["output_column"]
        timestamp_col = payload["timestamp_column"]
        input_cols = payload["input_columns"]
        begin_date = payload["begin_date"]
        end_date = payload["end_date"]
        project_id = payload["project_id"]
        bq_table = payload["bq_table"]
        bucket_name = payload["bucket_name"]

        # Phase 2 (inline): build and deploy
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = f"us-central1-docker.pkg.dev/{PROJECT_ID}/prophet/prophet_train:{timestamp}"
        inference_image_name = f"us-central1-docker.pkg.dev/{PROJECT_ID}/prophet/prophet_infer:{timestamp}"
        log_info(f"🏷️ Image tags set — train: {image_name}, infer: {inference_image_name} ✅")

        client = cloudbuild_v1.services.cloud_build.CloudBuildClient()
        if not gcs_source_uri.startswith("gs://"):
            raise ValueError("Invalid gcs_source_uri, must start with gs://")
        source_path = gcs_source_uri[len("gs://"):]
        build_source_bucket, object_name = source_path.split("/", 1)
        log_info(f"📦 Build source bucket: {build_source_bucket}, object: {object_name}")

        build = {
            "source": {"storage_source": {"bucket": build_source_bucket, "object": object_name}},
            "steps": [
                {
                    "name": "python:3.12",
                    "entrypoint": "bash",
                    "args": ["-lc", "pip install --no-cache-dir -r requirements.txt && python train.py"]
                },
                {"name": "gcr.io/cloud-builders/docker", "args": ["build", "-t", image_name, "."]},
                {"name": "gcr.io/cloud-builders/docker", "args": ["push", image_name]},
            ],
            "images": [image_name],
        }

        operation = client.create_build(project_id=PROJECT_ID, build=build)
        log_info(f"🧱 Cloud Build: training build started ⏳ name={operation.operation.name}")
        operation.result()
        log_info("🧱 Cloud Build: training build finished ✅")

        # Prepare inference source
        infer_workdir = tempfile.mkdtemp(prefix=f"infer_{timestamp}_")
        inference_template = env.get_template("inference_app_template.py.j2")
        rendered_infer_app = inference_template.render(
            project_id=project_id,
            bq_table=bq_table,
            bucket_name=bucket_name,
            model_blob_name=model_blob_name,
            timestamp_column=timestamp_col,
            output_column=output_col,
            input_columns=input_cols,
        )
        infer_app_path = os.path.join(infer_workdir, "inference_app.py")
        with open(infer_app_path, "w") as f:
            f.write(rendered_infer_app)
        log_info("📝 Rendered inference_app.py ✅")

        inference_dockerfile_template = os.path.join(TEMPLATE_DIR, "inference_dockerfile_template.j2")
        with open(inference_dockerfile_template) as src, open(os.path.join(infer_workdir, "Dockerfile"), "w") as dst:
            dst.write(src.read())
        log_info("📦 Dockerfile prepared for inference image ✅")

        inference_requirements_path = os.path.join(TEMPLATE_DIR, "requirements_inference.txt")
        with open(inference_requirements_path) as src, open(os.path.join(infer_workdir, "requirements.txt"), "w") as dst:
            dst.write(src.read())
        log_info("📚 Inference requirements.txt written ✅")

        infer_zip_filename = f"/tmp/infer_{timestamp}.zip"
        with zipfile.ZipFile(infer_zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(infer_workdir):
                for file in files:
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, infer_workdir)
                    zipf.write(full_path, arcname=relative_path)
        infer_bucket = storage.Client().bucket(GCS_TEMP_BUCKET)
        infer_gcs_object = f"infer_{timestamp}.zip"
        infer_blob = infer_bucket.blob(infer_gcs_object)
        infer_blob.upload_from_filename(infer_zip_filename)
        gcs_infer_source_uri = f"gs://{GCS_TEMP_BUCKET}/{infer_gcs_object}"
        log_info(f"☁️ Uploaded inference source to: {gcs_infer_source_uri} ✅")

        infer_source_path = gcs_infer_source_uri[len("gs://"):]
        infer_bucket_name, infer_object_name = infer_source_path.split("/", 1)
        log_info(f"📦 Inference source bucket: {infer_bucket_name}, object: {infer_object_name}")
        infer_build = {
            "source": {"storage_source": {"bucket": infer_bucket_name, "object": infer_object_name}},
            "steps": [
                {"name": "gcr.io/cloud-builders/docker", "args": ["build", "-t", inference_image_name, "."]},
                {"name": "gcr.io/cloud-builders/docker", "args": ["push", inference_image_name]},
            ],
            "images": [inference_image_name],
        }
        infer_operation = client.create_build(project_id=PROJECT_ID, build=infer_build)
        log_info("🧱 Cloud Build: inference image build started ⏳")
        infer_operation.result()
        log_info("🧱 Cloud Build: inference image build finished ✅")

        # Initialize Vertex AI
        aiplatform.init(project=PROJECT_ID, location=REGION)
        
        # Create Vertex AI Model
        model_display_name = f"prophet_model_{timestamp}"
        
        model = models.Model.upload(
            display_name=model_display_name,
            serving_container_image_uri=inference_image_name,
            serving_container_ports=[8080],
            serving_container_health_route="/",
            serving_container_predict_route="/predict",
        )
        model.wait()
        model_id = model.resource_name
        log_info(f"🧠 Vertex Model uploaded ✅ Model ID: {model_id}")

        # Create Endpoint
        endpoint_display_name = f"prophet_endpoint_{timestamp}"
        
        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_display_name,
        )
        endpoint.wait()
        endpoint_id = endpoint.resource_name
        log_info(f"📍 Endpoint created: {endpoint_id} ✅")

        # Deploy Model to Endpoint
        endpoint.deploy(
            model=model,
            deployed_model_display_name=f"prophet_deploy_{timestamp}",
            traffic_percentage=100,
            machine_type="n1-standard-2",  # Default machine type, adjust as needed
        )
        endpoint.wait()
        log_info("🚀 Model deployed to endpoint ✅")

        run_id = str(uuid.uuid4())

        log_info("🦄🎉 /generate_and_deploy completed successfully!")
        _store_run_metadata(
            payload,
            gen_artifacts=gen,
            gcs_infer_source_uri=gcs_infer_source_uri,
            train_image_uri=image_name,
            inference_image_uri=inference_image_name,
            model_display_name=model_display_name,
            model_id=model_id,
            endpoint_display_name=endpoint_display_name,
            endpoint_id=endpoint_id,
            run_id=run_id
        )
        return {
            "status": "success",
            "artifacts": {
                "workdir": gen["workdir"],
                "generated_files": gen["generated_files"],
                "gcs_source_uri": gcs_source_uri,
                "model_blob_name": model_blob_name,
            },
            "deployment": {
                "model_id": model_id,
                "endpoint_id": endpoint_id,
                "train_image_name": image_name,
                "inference_image_name": inference_image_name,
            },
            "run_id": run_id,
            "message": "Artifacts generated and model deployed automatically",
        }
    except Exception as e:
        log_info(f"🔥 Error during /generate_and_deploy: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

app.include_router(router)
