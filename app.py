import os
import tempfile
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from jinja2 import Environment, FileSystemLoader

app = FastAPI(title="Prophet Orchestrator (Phase 1 — Template-Based)")

# Initialize Jinja2 environment
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "prophet-orchestrator",
        "phase": 1,
        "mode": "template-based",
    }


@app.post("/generate")
async def generate_training_code(request: Request):
    """
    Generates training artifacts from Jinja2 templates using dynamic parameters.
    """
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

        # Create a timestamped working directory for this build
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        workdir = tempfile.mkdtemp(prefix=f"build_{timestamp}_")
        print(f"📁 Working directory: {workdir}")

        # === 1️⃣ Render train.py ===
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

        with open(os.path.join(workdir, "train.py"), "w") as f:
            f.write(rendered_train)

        # === 2️⃣ Copy Dockerfile ===
        dockerfile_template_path = os.path.join(TEMPLATE_DIR, "dockerfile_template.j2")
        dockerfile_target = os.path.join(workdir, "Dockerfile")

        with open(dockerfile_template_path) as src, open(dockerfile_target, "w") as dst:
            dst.write(src.read())

        # === 3️⃣ Copy training requirements ===
        train_requirements_path = os.path.join(TEMPLATE_DIR, "requirements_train.txt")
        target_requirements = os.path.join(workdir, "requirements.txt")
        if not os.path.exists(train_requirements_path):
            raise FileNotFoundError("Missing templates/requirements_train.txt")
        with open(train_requirements_path) as src, open(target_requirements, "w") as dst:
            dst.write(src.read())

        # Return success
        return {
            "status": "success",
            "message": "Training artifacts generated successfully",
            "workdir": workdir,
            "generated_files": ["train.py", "Dockerfile", "requirements.txt"],
            "notes": "Training requirements were copied from templates/requirements_train.txt",
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
