# Emotion Classification API (NLP)

Production-oriented Emotion Classification API built with FastAPI and a fine-tuned Transformer-based model.

This project focuses on end-to-end ML engineering, covering the full lifecycle from data preparation and model training to secure, scalable inference via a REST API.
It intentionally prioritizes real-world constraints, limitations, and trade-offs over benchmark chasing.

---

## Project Goals

- Build a realistic NLP inference service suitable for production environments
- Implement a complete ML pipeline: training → serialization → serving
- Explore practical challenges in emotion classification (label semantics, dataset bias)
- Demonstrate ML engineering skills beyond model accuracy

---

## Key Features

- Transformer-based emotion classification
- End-to-end ML pipeline (training + inference)
- FastAPI-based REST API
- Batch and single-text inference
- Dynamic batching for efficient throughput
- API key authentication
- Rate limiting per client
- Health check endpoint
- Internal runtime metrics
- Fully tested API surface

---

## Quick Start
- Activate virtual environment  
    .\venv\Scripts\Activate.ps1

- Install dependencies  
    pip install -r requirements.txt

- Run the API  
    uvicorn app.main:app --reload

- Run all tests  
    pytest

---

## Supported Emotion Labels

The model predicts one of the following basic emotion classes:
- anger
- disgust
- fear
- joy
- neutral
- sadness
- surprise

### Important note on semantics
Emotion classification is inherently ambiguous.
Certain expressions (e.g. “excited”) often span multiple emotional dimensions (joy, anticipation, fear).
This project uses a single-label classification setup, which reflects common industry datasets but imposes natural limitations on such cases.

---

## API Endpoints

- Authentication
    All inference endpoints require an API key.

    Header:
    X-API-Key: <your-api-key>

- Single Prediction
    `POST /v1/predict-emotion`

    Request:
    {
        "text": "I am very happy today"
    }
    Response:

    {
        "emotion": "joy",
        "confidence": 0.92
    }
- Batch Prediction
    POST /v1/predict-emotion-batch

    Request:
    {
        "texts": [
            "I am happy",
            "I am sad",
            "This is surprising"
        ]
    }
    Response:
    {
        "results": [
            {
                "emotion": "joy",
                "confidence": 0.88
            },
            {
                "emotion": "sadness",
                "confidence": 0.91
            },
            {
                "emotion": "surprise",
                "confidence": 0.86
            }
        ]
    }
- Health Check
    GET /v1/health/live

    Used for container orchestration and monitoring.

    Response:
    {
        "status": "healthy",
        "inference": "ok"
    }
- Metrics (internal)
    GET /v1/internal/metrics

    Returns internal runtime metrics (requests, batches, latency, etc.).

---

## Dynamic Batching
The API uses a DynamicBatcher to group inference requests automatically:

- Adaptive batch size
- Maximum wait time per batch
- Single and batch inference supported
- Optimized for high-throughput workloads
- This design allows efficient scaling under load without changing the API interface.

---

## ML Training Pipeline
The src/ directory contains the complete machine learning training workflow, clearly separated from the inference layer.
This structure reflects real-world ML engineering practices used in production systems.

### Covered responsibilities
- Dataset loading and cleaning
- Label normalization and mapping
- Tokenization and preprocessing
- Fine-tuning a pretrained Transformer model
- Handling model serialization with safetensors
- Managing label / classifier head mismatches
- Generating inference-ready artifacts

The training pipeline intentionally reflects real-world issues, including:
- Dataset label inconsistencies
- Missing or underrepresented classes
- Security constraints in modern ML frameworks (e.g. PyTorch CVEs)

---

## Project Structure

ml-training-pipeline/
├─ app/
│  ├─ main.py
│  ├─ batcher.py
│  ├─ deps.py
│  ├─ security.py
│  ├─ schemas.py
│  ├─ metrics.py
│  ├─ config.py
│  ├─ model.py
│  └─ utils/
│     └─ prediction.py
├─ scripts/
├─ src/
│  ├─ config.py
│  ├─ load_data.py
│  ├─ preprocess.py
│  └─ train_transformer.py
├─ tests/
├─ README.md
└─ requirements.txt

---

## Testing
All endpoints are covered by automated tests:
- Health check
- Metrics endpoint
- Single prediction
- Batch prediction
- Authentication
- Edge cases (empty input)

---

## Technologies Used
- Python 3.11
- FastAPI
- PyTorch
- Hugging Face Transformers
- pandas
- scikit-learn
- pytest
- Uvicorn

---

## Known Limitations
- Single-label emotion classification cannot fully represent mixed emotions
- Emotion boundaries depend heavily on dataset definitions
- Some expressions (e.g. excited, nervous) remain inherently ambiguous
- These limitations are intentional discussion points, not oversights.

---

## Future Improvements
- Multi-label emotion classification
- Hierarchical emotion taxonomy
- Dockerized deployment
- CI/CD pipeline
- Model versioning and rollback
- Prometheus-compatible metrics

---

## Author
Alberto Nadal López

Project developed as part of a professional transition toward a Machine Learning Engineer role, focusing on practical ML system design, production constraints, and honest evaluation of model limitations.
