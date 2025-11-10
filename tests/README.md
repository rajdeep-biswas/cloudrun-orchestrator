# Prophet Model Testing Scripts

This folder contains standalone scripts for testing Prophet model training and inference with different dependency versions.

## Setup

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **If dependency versions need to be changed:**
   - Edit `requirements.txt` with new versions
   - Reinstall: `pip install -r requirements.txt --upgrade`

## Files

- `requirements.txt` - All dependencies including Prophet versions from templates
- `train_model.py` - Training script with configurable parameters
- `inference_server.py` - Inference endpoint server

## Usage

### 1. Train a Model

Edit the `CONFIG` dictionary in `train_model.py` with your parameters, then run:

```bash
python train_model.py
```

The script will:
- Query BigQuery for training data
- Train a Prophet model
- Upload the model to GCS
- Print the model blob name for use in inference

**Example output:**
```
✅ Model uploaded successfully!
🔗 GCS path: gs://aphw-prophet-models/dynamic_models/model_20250115_143022.pkl
```

### 2. Run Inference Server

The inference server automatically loads model information from `model_info.json` (created by `train_model.py`). No manual configuration needed!

**Start the server:**
```bash
python inference_server.py
```

**Note:** If you see an error about missing `model_info.json`, make sure you've run `train_model.py` first.

3. **Test the endpoint:**
   ```bash
   # Health check
   curl http://localhost:8080/
   
   # Prediction
   curl -X POST http://localhost:8080/predict \
     -H "Content-Type: application/json" \
     -d '{
       "instances": [
         {
           "call_id": "test123",
           "call_start_time_est": "2025-09-14 14:25:37.974",
           "dnis": "1234567890",
           "customer_speaking_duration": 45.5
         }
       ]
     }'
   ```

## Configuration

### Training Parameters (`train_model.py`)

```python
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
    "bucket_name": "aphw-prophet-models"
}
```

### Model Info File (`model_info.json`)

Created automatically by `train_model.py`, contains:
- `model_blob_name` - GCS blob path to the trained model
- `bucket_name` - GCS bucket name
- `timestamp_column`, `output_column`, `input_columns` - Model configuration
- `trained_at` - Training timestamp
- `training_config` - Training parameters used

The inference server automatically reads from this file - no manual configuration needed!

## Troubleshooting

### Dependency Version Issues

1. Edit `requirements.txt` with new versions
2. Reinstall: `pip install -r requirements.txt --upgrade --force-reinstall`
3. Retry training/inference

### Common Issues

- **Prophet import errors**: Check `cmdstanpy` and `prophet` versions are compatible
- **NumPy compatibility**: Ensure `numpy==2.3.2` or compatible version
- **GCS authentication**: Ensure `gcloud auth application-default login` is run
- **BigQuery permissions**: Verify service account has BigQuery read permissions

## Testing Different Versions

To test different dependency versions:

1. Edit `tests/requirements.txt`
2. Reinstall: `pip install -r requirements.txt --upgrade --force-reinstall`
3. Run training: `python train_model.py`
4. If successful, update inference server and test: `python inference_server.py`

Repeat until you find working versions!

