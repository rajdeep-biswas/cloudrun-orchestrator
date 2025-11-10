# Run the fastapi applications
`uvicorn pipeline.generate_and_deploy:app --reload --host 0.0.0.0 --port 8080`
`uvicorn pipeline.inference:app --reload --host 0.0.0.0 --port 8080`
