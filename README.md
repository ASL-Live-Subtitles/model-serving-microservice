# ASL Live Subtitles - Model Serving Microservice

## Overview

The ASL Live Subtitles Model Serving Microservice is a FastAPI-based service designed to provide real-time American Sign Language (ASL) recognition and translation capabilities. This microservice serves as the core ML inference engine for converting hand gestures into text, progressing from letter-level recognition to full sentence generation with AI assistance.

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete deployment instructions.

**Quick deploy:**

```bash
./deploy-to-gcp.sh
```

**Current deployment:**

- URL: http://34.132.238.32:8001
- Swagger UI: http://34.132.238.32:8001/docs

## Quick Start

### Prerequisites

- Python 3.11 or higher
- pip or uv package manager

### Installation

1. **Clone the repository:**

```bash
git clone git@github.com:ASL-Live-Subtitles/model-serving-microservice.git
cd model-serving-microservice
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

### Running the Service

**Start the development server:**

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The service will be available at:

- **API Base URL:** http://localhost:8000
- **Interactive API Documentation:** http://localhost:8000/docs

## API Endpoints

### Core Resources

#### 🤲 Gestures

- `GET /gestures` - List all gesture predictions
- `POST /gestures` - Submit new gesture data for recognition
- `PUT /gestures/{gesture_id}` - Update gesture prediction (NOT IMPLEMENTED)
- `DELETE /gestures/{gesture_id}` - Delete gesture prediction

#### 🤖 Models

- `GET /models` - List all available ML models
- `POST /models` - Register a new ML model
- `PUT /models/{model_id}` - Update model configuration (NOT IMPLEMENTED)
- `DELETE /models/{model_id}` - Remove ML model

#### 🔮 Predictions

- `GET /predictions` - List all prediction requests
- `POST /predictions` - Create batch prediction request
- `PUT /predictions/{prediction_id}` - Update prediction status (NOT IMPLEMENTED)
- `DELETE /predictions/{prediction_id}` - Cancel prediction request

#### 🏥 Health Check

- `GET /` - Root endpoint with service information
- `GET /health` - Health check endpoint

## Testing the API

### Using Swagger UI (Recommended)

1. **Start the server** (see Running the Service section above)
2. **Open your browser** and navigate to http://localhost:8000/docs
3. **Explore the interactive documentation** - you can test all endpoints directly from the browser

![Swagger UI](Swagger%20UI.png)

### Using curl or HTTP clients

**Example: Submit gesture data**

```bash
curl -X POST "http://localhost:8000/gestures" \
  -H "Content-Type: application/json" \
  -d '{
    "landmarks": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
    "user_id": "test_user_123"
  }'
```

**Example: List all gestures**

```bash
curl -X GET "http://localhost:8000/gestures"
```

**Example: Register a new model**

```bash
curl -X POST "http://localhost:8000/models" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "ASL_Letter_Classifier_v1",
    "version": "1.0.0",
    "model_path": "/models/asl_letters.h5",
    "input_shape": [21, 2],
    "output_shape": [26],
    "model_type": "tensorflow"
  }'
```

### API Testing Results

![API Test Results](API%20Test.png)

## Data Models

### GestureInput

```json
{
  "landmarks": [
    [0.1, 0.2],
    [0.3, 0.4]
  ],
  "user_id": "optional_user_id"
}
```

### GestureResult

```json
{
  "id": "gesture_123",
  "predicted_gesture": "A",
  "confidence": 0.95,
  "processing_time_ms": 45.2,
  "model_version": "v1.0.0",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### ModelInput

```json
{
  "name": "ASL_Letter_Classifier",
  "version": "1.0.0",
  "model_path": "/models/classifier.h5",
  "input_shape": [21, 2],
  "output_shape": [26],
  "model_type": "tensorflow"
}
```

## Development

### Project Structure

```
model-serving-microservice/
├── main.py              # FastAPI application and endpoints
├── models/              # Pydantic data models
│   ├── __init__.py
│   └── gesture.py       # Data schemas and validation
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

### Adding New Features

1. **Define new data models** in `models/gesture.py`
2. **Add new endpoints** in `main.py`
3. **Update requirements.txt** if new dependencies are needed
4. **Test your changes** using the Swagger UI or curl commands

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for:

- Google Cloud Compute Engine (VM)
- Google Cloud Run (serverless)
- Local Docker setup
- Management commands
- Troubleshooting

### Local Development

```bash
# Docker
docker-compose up

# Or direct
docker build -t model-serving .
docker run -p 8001:8001 model-serving
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

For questions, issues, or contributions, please visit our GitHub repository or contact the development team.
