import os
import logging
import tempfile
import subprocess
import zipfile
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from jinja2 import Environment, FileSystemLoader
from google.cloud import bigquery
from google.cloud import storage

app = FastAPI(title="Prophet Orchestrator")

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# Log through Uvicorn's error logger so messages appear in console
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)
logger.propagate = True

def log_info(message: str):
    logger.info(message)

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))

# === Config ===
PROJECT_ID = "dev-poc-429118"
REGION = "us-central1"
ARTIFACT_REGISTRY = f"us-central1-docker.pkg.dev/{PROJECT_ID}/prophet"
BQ_TABLE = f"{PROJECT_ID}.prophet_model_registry.model_data"

GCS_TEMP_BUCKET = "prophet-temp-bucket"  # pre-created bucket for temporary workdirs

# ================== Phase 1 ==================
@app.get("/")
def health_check():
    log_info("🦄✨ Health check hit — service is up!")
    return {"status": "ok", "phase": "1+2"}

@app.post("/generate")
async def generate_training_code(request: Request):
    try:
        log_info("🦄 Starting /generate — preparing training artifacts...")
        payload = await request.json()
        log_info("📥 Received payload for training artifact generation ✅")
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
        ]
        missing = [k for k in required_keys if k not in payload]
        if missing:
            log_info(f"❌ Missing required keys: {missing}")
            return JSONResponse({"error": f"Missing keys: {missing}"}, status_code=400)

        # Extract payload
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
        log_info(f"📂 Working directory created: {workdir} ✅")

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
        log_info("📝 Rendered and wrote train.py ✅")

        # --- Copy Dockerfile ---
        dockerfile_template_path = os.path.join(TEMPLATE_DIR, "dockerfile_template.j2")
        dockerfile_target = os.path.join(workdir, "Dockerfile")
        with open(dockerfile_template_path) as src, open(dockerfile_target, "w") as dst:
            dst.write(src.read())
        log_info("📦 Dockerfile prepared for training image ✅")

        # --- Copy training requirements ---
        train_requirements_path = os.path.join(TEMPLATE_DIR, "requirements_train.txt")
        target_requirements = os.path.join(workdir, "requirements.txt")
        with open(train_requirements_path) as src, open(target_requirements, "w") as dst:
            dst.write(src.read())
        log_info("📚 Training requirements.txt written ✅")

        # --- Zip workdir ---
        zip_filename = f"/tmp/workdir_{timestamp}.zip"
        with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(workdir):
                for file in files:
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, workdir)
                    zipf.write(full_path, arcname=relative_path)
        log_info(f"🗜️ Zipped workdir to {zip_filename} ✅")

        # --- Upload to GCS ---
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_TEMP_BUCKET)
        gcs_object = f"workdir_{timestamp}.zip"
        blob = bucket.blob(gcs_object)
        blob.upload_from_filename(zip_filename)
        gcs_source_uri = f"gs://{GCS_TEMP_BUCKET}/{gcs_object}"
        log_info(f"☁️ Uploaded zip to GCS: {gcs_source_uri} ✅")

        log_info("🦄🎉 /generate completed successfully!")
        return {
            "status": "success",
            "workdir": workdir,
            "generated_files": ["train.py", "Dockerfile", "requirements.txt"],
            "gcs_source_uri": gcs_source_uri,
            "model_blob_name": model_blob_name,
            "message": "Training artifacts generated, zipped, and uploaded to GCS automatically"
        }

    except Exception as e:
        log_info(f"🔥 Error during /generate: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# ================== Phase 2 ==================
@app.post("/build_and_deploy")
async def build_and_deploy(request: Request):
    """
    Build Docker image, push to Artifact Registry, create Vertex AI Model & Endpoint,
    store metadata in BigQuery. Fully automated using gcs_source_uri from /generate.
    """
    try:
        log_info("🦄 Starting /build_and_deploy — building, training, and deploying...")
        payload = await request.json()
        log_info("📥 Received payload for build and deploy ✅")
        required_keys = [
            "gcs_source_uri",
            "output_column",
            "timestamp_column",
            "input_columns",
            "begin_date",
            "end_date",
            "project_id",
            "bq_table",
            "bucket_name",
            "model_blob_name",
        ]
        missing = [k for k in required_keys if k not in payload]
        if missing:
            log_info(f"❌ Missing required keys: {missing}")
            return JSONResponse({"error": f"Missing keys: {missing}"}, status_code=400)

        gcs_source_uri = payload["gcs_source_uri"]
        output_col = payload["output_column"]
        timestamp_col = payload["timestamp_column"]
        input_cols = payload["input_columns"]
        begin_date = payload["begin_date"]
        end_date = payload["end_date"]
        project_id = payload["project_id"]
        bq_table = payload["bq_table"]
        bucket_name = payload["bucket_name"]
        model_blob_name = payload["model_blob_name"]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = f"us-central1-docker.pkg.dev/{PROJECT_ID}/prophet/prophet_train:{timestamp}"
        inference_image_name = f"us-central1-docker.pkg.dev/{PROJECT_ID}/prophet/prophet_infer:{timestamp}"
        log_info(f"🏷️ Image tags set — train: {image_name}, infer: {inference_image_name} ✅")

        # --- 1️⃣ Build & push Docker image via Cloud Build ---
        from google.cloud.devtools import cloudbuild_v1
        client = cloudbuild_v1.services.cloud_build.CloudBuildClient()
        # Extract bucket and object from gcs_source_uri
        if not gcs_source_uri.startswith("gs://"):
            raise ValueError("Invalid gcs_source_uri, must start with gs://")
        source_path = gcs_source_uri[len("gs://"):]
        build_source_bucket, object_name = source_path.split("/", 1)
        log_info(f"📦 Build source bucket: {build_source_bucket}, object: {object_name}")

        build = {
            "source": {"storage_source": {"bucket": build_source_bucket, "object": object_name}},
            "steps": [
                # Run training inside Cloud Build with ADC so BigQuery/GCS auth works
                {
                    "name": "python:3.12",
                    "entrypoint": "bash",
                    "args": ["-lc", "pip install --no-cache-dir -r requirements.txt && python train.py"]
                },
                # Optionally build and push the training image for traceability
                {"name": "gcr.io/cloud-builders/docker", "args": ["build", "-t", image_name, "."]},
                {"name": "gcr.io/cloud-builders/docker", "args": ["push", image_name]},
            ],
            "images": [image_name],
        }

        operation = client.create_build(project_id=PROJECT_ID, build=build)
        log_info(f"🧱 Cloud Build: training build started ⏳ name={operation.operation.name}")
        operation.result()  # Wait for build completion
        log_info("🧱 Cloud Build: training build finished ✅")

        # --- 2️⃣ Prepare inference source (render templates) ---
        # Create a temporary workdir for inference server assets
        infer_workdir = tempfile.mkdtemp(prefix=f"infer_{timestamp}_")
        # Render inference_app.py
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

        # Copy inference Dockerfile template
        inference_dockerfile_template = os.path.join(TEMPLATE_DIR, "inference_dockerfile_template.j2")
        with open(inference_dockerfile_template) as src, open(os.path.join(infer_workdir, "Dockerfile"), "w") as dst:
            dst.write(src.read())
        log_info("📦 Dockerfile prepared for inference image ✅")

        # Copy inference requirements
        inference_requirements_path = os.path.join(TEMPLATE_DIR, "requirements_inference.txt")
        with open(inference_requirements_path) as src, open(os.path.join(infer_workdir, "requirements.txt"), "w") as dst:
            dst.write(src.read())
        log_info("📚 Inference requirements.txt written ✅")

        # Zip inference workdir and upload to GCS
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

        # --- 3️⃣ Build & push inference image via Cloud Build ---
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

        # --- 4️⃣ Create Vertex AI Model using inference image ---
        model_display_name = f"prophet_model_{timestamp}"
        subprocess.run([
            "gcloud", "ai", "models", "upload",
            "--region", REGION,
            "--display-name", model_display_name,
            "--container-image-uri", inference_image_name,
            "--container-ports=8080",
            "--container-health-route=/",
            "--container-predict-route=/predict",
            "--project", PROJECT_ID
        ], check=True)
        log_info("🧠 Vertex Model uploaded ✅")

        # --- 3️⃣ Get Model ID ---
        model_id = subprocess.check_output([
            "gcloud", "ai", "models", "list",
            "--region", REGION,
            "--filter", f"display_name:{model_display_name}",
            "--format=value(name)"
        ]).decode().strip()
        log_info(f"🆔 Retrieved Model ID: {model_id} ✅")

        # --- 5️⃣ Create Endpoint ---
        endpoint_display_name = f"prophet_endpoint_{timestamp}"
        endpoint_id = subprocess.check_output([
            "gcloud", "ai", "endpoints", "create",
            "--region", REGION,
            "--display-name", endpoint_display_name,
            "--project", PROJECT_ID,
            "--format=value(name)"
        ]).decode().strip()
        log_info(f"📍 Endpoint created: {endpoint_id} ✅")

        # --- 6️⃣ Deploy Model to Endpoint ---
        subprocess.run([
            "gcloud", "ai", "endpoints", "deploy-model", endpoint_id,
            "--region", REGION,
            "--model", model_id,
            "--display-name", f"prophet_deploy_{timestamp}",
            "--traffic-split=0=100"
        ], check=True)
        log_info("🚀 Model deployed to endpoint ✅")

        """
        # --- 7️⃣ Store Metadata in BigQuery ---
        bq_client = bigquery.Client(project=PROJECT_ID)
        rows_to_insert = [{
            "model_id": model_id,
            "endpoint_url": endpoint_id,
            "output_column": output_col,
            "timestamp_column": timestamp_col,
            "input_columns": input_cols,
            "trained_on_begin_date": begin_date,
            "trained_on_end_date": end_date,
            "train_image_name": image_name,
            "inference_image_name": inference_image_name,
            "gcs_source": gcs_source_uri,
            "model_blob_name": model_blob_name,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }]
        errors = bq_client.insert_rows_json(BQ_TABLE, rows_to_insert)
        if errors:
            raise RuntimeError(f"BigQuery insert errors: {errors}")
        log_info("📊 Metadata stored in BigQuery ✅")
        """

        log_info("🦄🎉 /build_and_deploy completed successfully!")
        return {
            "status": "success",
            "model_id": model_id,
            "endpoint_id": endpoint_id,
            "image_name": image_name,
            "gcs_source": gcs_source_uri,
            "message": "Cloud Build ran, model & endpoint deployed automatically" #, metadata stored automatically"
        }

    except Exception as e:
        log_info(f"🔥 Error during /build_and_deploy: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
