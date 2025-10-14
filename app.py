import os
import tempfile
import subprocess
import zipfile
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from jinja2 import Environment, FileSystemLoader
from google.cloud import bigquery
from google.cloud import storage

app = FastAPI(title="Prophet Orchestrator (Phase 1+2)")

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
    return {"status": "ok", "phase": "1+2"}

@app.post("/generate")
async def generate_training_code(request: Request):
    try:
        payload = await request.json()
        required_keys = [
            "output_column",
            "input_columns",
            "timestamp_column",
            "begin_date",
            "end_date",
            "interval_width",
            "scaling_factor",
        ]
        missing = [k for k in required_keys if k not in payload]
        if missing:
            return JSONResponse({"error": f"Missing keys: {missing}"}, status_code=400)

        # Extract payload
        output_col = payload["output_column"]
        input_cols = payload["input_columns"]
        timestamp_col = payload["timestamp_column"]
        begin_date = payload["begin_date"]
        end_date = payload["end_date"]
        interval_width = payload["interval_width"]
        scaling_factor = payload["scaling_factor"]

        # Create timestamped working directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        workdir = tempfile.mkdtemp(prefix=f"build_{timestamp}_")

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
        )
        train_path = os.path.join(workdir, "train.py")
        with open(train_path, "w") as f:
            f.write(rendered_train)

        # --- Copy Dockerfile ---
        dockerfile_template_path = os.path.join(TEMPLATE_DIR, "dockerfile_template.j2")
        dockerfile_target = os.path.join(workdir, "Dockerfile")
        with open(dockerfile_template_path) as src, open(dockerfile_target, "w") as dst:
            dst.write(src.read())

        # --- Copy training requirements ---
        train_requirements_path = os.path.join(TEMPLATE_DIR, "requirements_train.txt")
        target_requirements = os.path.join(workdir, "requirements.txt")
        with open(train_requirements_path) as src, open(target_requirements, "w") as dst:
            dst.write(src.read())

        # --- Zip workdir ---
        zip_filename = f"/tmp/workdir_{timestamp}.zip"
        with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(workdir):
                for file in files:
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, workdir)
                    zipf.write(full_path, arcname=relative_path)

        # --- Upload to GCS ---
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_TEMP_BUCKET)
        gcs_object = f"workdir_{timestamp}.zip"
        blob = bucket.blob(gcs_object)
        blob.upload_from_filename(zip_filename)
        gcs_source_uri = f"gs://{GCS_TEMP_BUCKET}/{gcs_object}"

        return {
            "status": "success",
            "workdir": workdir,
            "generated_files": ["train.py", "Dockerfile", "requirements.txt"],
            "gcs_source_uri": gcs_source_uri,
            "message": "Training artifacts generated, zipped, and uploaded to GCS automatically"
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ================== Phase 2 ==================
@app.post("/build_and_deploy")
async def build_and_deploy(request: Request):
    """
    Build Docker image, push to Artifact Registry, create Vertex AI Model & Endpoint,
    store metadata in BigQuery. Fully automated using gcs_source_uri from /generate.
    """
    try:
        payload = await request.json()
        required_keys = [
            "gcs_source_uri",
            "output_column",
            "timestamp_column",
            "input_columns",
            "begin_date",
            "end_date"
        ]
        missing = [k for k in required_keys if k not in payload]
        if missing:
            return JSONResponse({"error": f"Missing keys: {missing}"}, status_code=400)

        gcs_source_uri = payload["gcs_source_uri"]
        output_col = payload["output_column"]
        timestamp_col = payload["timestamp_column"]
        input_cols = payload["input_columns"]
        begin_date = payload["begin_date"]
        end_date = payload["end_date"]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = f"us-central1-docker.pkg.dev/{PROJECT_ID}/prophet/prophet_train:{timestamp}"

        # --- 1️⃣ Build & push Docker image via Cloud Build ---
        from google.cloud.devtools import cloudbuild_v1
        client = cloudbuild_v1.services.cloud_build.CloudBuildClient()
        # Extract bucket and object from gcs_source_uri
        if not gcs_source_uri.startswith("gs://"):
            raise ValueError("Invalid gcs_source_uri, must start with gs://")
        _, bucket_name, *object_parts = gcs_source_uri.split("/")
        object_name = "/".join(object_parts)

        build = {
            "source": {"storage_source": {"bucket": bucket_name, "object": object_name}},
            "steps": [
                {"name": "gcr.io/cloud-builders/docker", "args": ["build", "-t", image_name, "."]},
                {"name": "gcr.io/cloud-builders/docker", "args": ["push", image_name]},
            ],
            "images": [image_name],
        }

        operation = client.create_build(project_id=PROJECT_ID, build=build)
        operation.result()  # Wait for build completion

        # --- 2️⃣ Create Vertex AI Model ---
        model_display_name = f"prophet_model_{timestamp}"
        subprocess.run([
            "gcloud", "ai", "models", "upload",
            "--region", REGION,
            "--display-name", model_display_name,
            "--container-image-uri", image_name,
            "--project", PROJECT_ID
        ], check=True)

        # --- 3️⃣ Get Model ID ---
        model_id = subprocess.check_output([
            "gcloud", "ai", "models", "list",
            "--filter", f"display_name:{model_display_name}",
            "--format=value(name)"
        ]).decode().strip()

        # --- 4️⃣ Create Endpoint ---
        endpoint_display_name = f"prophet_endpoint_{timestamp}"
        endpoint_id = subprocess.check_output([
            "gcloud", "ai", "endpoints", "create",
            "--region", REGION,
            "--display-name", endpoint_display_name,
            "--project", PROJECT_ID
        ]).decode().strip()

        # --- 5️⃣ Deploy Model to Endpoint ---
        subprocess.run([
            "gcloud", "ai", "endpoints", "deploy-model", endpoint_id,
            "--model", model_id,
            "--display-name", f"prophet_deploy_{timestamp}",
            "--traffic-split=100"
        ], check=True)

        # --- 6️⃣ Store Metadata in BigQuery ---
        bq_client = bigquery.Client(project=PROJECT_ID)
        rows_to_insert = [{
            "model_id": model_id,
            "endpoint_url": endpoint_id,
            "output_column": output_col,
            "timestamp_column": timestamp_col,
            "input_columns": input_cols,
            "trained_on_begin_date": begin_date,
            "trained_on_end_date": end_date,
            "image_name": image_name,
            "gcs_source": gcs_source_uri,
            "created_at": datetime.utcnow(),
        }]
        errors = bq_client.insert_rows_json(BQ_TABLE, rows_to_insert)
        if errors:
            raise RuntimeError(f"BigQuery insert errors: {errors}")

        return {
            "status": "success",
            "model_id": model_id,
            "endpoint_id": endpoint_id,
            "image_name": image_name,
            "gcs_source": gcs_source_uri,
            "message": "Cloud Build ran, model & endpoint deployed, metadata stored automatically"
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
