# Semantic Entropy API

This is a minimal FastAPI wrapper that runs the existing semantic uncertainty pipeline for a single question.

## Run

From the repo root, activate your Python environment (the same one used for the pipeline), then run:

```
uvicorn app:app --reload --port 8000 --app-dir UI/semantic-entropy-api
```

## Request

POST `http://localhost:8000/analyze`

```json
{
  "question": "What is the capital of France?",
  "modelId": "falcon-7b",
  "sampleSize": 10
}
```
