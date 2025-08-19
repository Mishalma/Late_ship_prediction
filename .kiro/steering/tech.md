# Technology Stack

## Core Technologies

- **Python 3.11**: Primary programming language
- **FastAPI**: REST API framework with automatic OpenAPI documentation
- **Flask**: Alternative web framework (used in app.py for local development)
- **scikit-learn**: Machine learning library for Random Forest models
- **pandas & NumPy**: Data manipulation and numerical computing
- **Pydantic**: Data validation and schema definition
- **MLflow**: Experiment tracking and model versioning

## Data Processing

- **joblib**: Model serialization and loading
- **RobustScaler**: Feature scaling for numerical data
- **OneHotEncoder**: Categorical feature encoding for nominal variables
- **OrdinalEncoder**: Categorical feature encoding for ordinal variables

## Development & Testing

- **pytest**: Testing framework with FastAPI TestClient integration
- **Uvicorn**: ASGI server for FastAPI applications
- **Docker**: Containerization for deployment

## Common Commands

### Pipeline Execution
```bash
# Run the complete ML pipeline
python run_pipeline.py

# Install dependencies
pip install -r requirements.txt
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_main.py
```

### API Development
```bash
# Start FastAPI development server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Start Flask development server (alternative)
python app.py
```

### Docker Operations
```bash
# Build container
docker build -t shipment-prediction .

# Run container
docker run -p 8000:8000 shipment-prediction
```

## Model Artifacts

All trained models and preprocessors are saved in `models/` directory:
- `late_model.pkl`: Random Forest for 1+ day delays
- `very_late_model.pkl`: Random Forest for 3+ day delays  
- `scaler.pkl`: Feature scaler
- `onehot_encoder.pkl`: Categorical encoder for nominal features
- `ordinal_encoder.pkl`: Categorical encoder for ordinal features