# Run the fastapi applications
`uvicorn pipeline.generate_and_deploy:app --reload --host 0.0.0.0 --port 8080`
`uvicorn pipeline.inference:app --reload --host 0.0.0.0 --port 8080`

# Specify google credentials via file instead of python runtime
`export GOOGLE_APPLICATION_CREDENTIALS=dev-poc-429118-8cb81dd6266e.json`
Use `unset GOOGLE_APPLICATION_CREDENTIALS` if you want to revert to default behavior.