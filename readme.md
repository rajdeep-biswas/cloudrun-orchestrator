# Run the fastapi applications
## `/generate_and_deploy`
`uvicorn pipeline.generate_and_deploy:app --reload --host 0.0.0.0 --port 8080`
## `/inference`
`uvicorn pipeline.inference:app --reload --host 0.0.0.0 --port 8080`

# Specify google credentials via file instead of python runtime
`export GOOGLE_APPLICATION_CREDENTIALS=dev-poc-429118-8cb81dd6266e.json`  
Use `unset GOOGLE_APPLICATION_CREDENTIALS` if you want to revert to default behavior of using [`cloud auth application-default login`](https://docs.google.com/document/d/1UZHzdlizD4jT-GmGa-Y1F7DGa-k6Who6njT2i_qik4o/edit?tab=t.0).

# Payload for `/generate_and_deploy`
```json
{
  "output_column": "call_duration",
  "model_description": "Trying out call_duration with dnis, csd, and hsrd",
  "configuration_name": "Rajdeep's model",
  "input_columns": ["dnis", "customer_speaking_duration", "hsr_duration"],
  "begin_date": "2025-08-01",
  "end_date": "2025-09-01",
  "interval_width": 0.9,
  "scaling_factor": 2.2
//  "timestamp_column": "call_start_time_est",
//  "project_id": "dev-poc-429118",
//  "bq_table": "dev-poc-429118.aa_genai.call-duration-aphw-aug-sep",
//  "bucket_name": "aphw-prophet-models"
}
```

## Response for `/generate_and_deploy`
```json
{
    "status": "success",
    "artifacts": {
        "workdir": "/var/folders/1t/xkjwxwlx2570z54_8k4bp4yw0000gn/T/build_20251113_173411_34ywivjo",
        "generated_files": [
            "train.py",
            "Dockerfile",
            "requirements.txt"
        ],
        "gcs_source_uri": "gs://prophet-temp-bucket/workdir_20251113_173411.zip",
        "model_blob_name": "dynamic_models/model_20251113_173411.pkl"
    },
    "deployment": {
        "model_id": "projects/893955258323/locations/us-central1/models/4783111975825571840",
        "endpoint_id": "projects/893955258323/locations/us-central1/endpoints/3412856604826009600",
        "train_image_name": "us-central1-docker.pkg.dev/dev-poc-429118/prophet/prophet_train:20251113_173412",
        "inference_image_name": "us-central1-docker.pkg.dev/dev-poc-429118/prophet/prophet_infer:20251113_173412"
    },
    "run_id": "d1dbc6d4-7b76-450f-90bd-2e75566106e2",
    "message": "Artifacts generated and model deployed automatically"
}
```

# Payload for `/inference`
```json
{
    "run_id": "d1dbc6d4-7b76-450f-90bd-2e75566106e2",
    "begin_date": "2025-08-15",
    "end_date": "2025-09-01"
}
```

## Response for /`inference`
```json
{
    "status": "success",
    "run_id": "d1dbc6d4-7b76-450f-90bd-2e75566106e2",
    "num_instances": 3279,
    "predictions": {
        "predictions": [
            {
                "customer_speaking_percentage": 0,
                "call_end_time_est": "2025-08-19T20:33:27.027",
                "is_anomaly": false,
                "yhat": -16.9591722905862,
                "yhat_upper": 66.79254385840875,
                "yhat_lower": -101.5084507396153,
                "ds": "2025-08-18T04:10:38.345000",
                "call_id": "179c1ab916e81572ad61191327766bb6",
                "call_duration": 25.112
            },
            ...
        ]
    }
}
```