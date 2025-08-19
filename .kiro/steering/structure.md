# Project Structure

## Directory Organization

```
late-shipment-predictions-ml/
├── api/                     # FastAPI application components
│   ├── main.py             # FastAPI app entry point with router registration
│   ├── shipment_schema.py  # Pydantic models for request validation
│   └── __init__.py
├── routers/                # FastAPI route definitions
│   ├── landing.py          # Root endpoint ("/")
│   ├── ping.py            # Health check endpoint ("/ping")
│   ├── predict_late.py    # Late shipment prediction endpoint
│   └── predict_very_late.py # Very late shipment prediction endpoint
├── src/                    # Core ML pipeline modules
│   ├── load_data.py       # Raw data loading
│   ├── clean_data.py      # Data cleaning and validation
│   ├── feature_engineering.py # Feature creation and transformation
│   ├── preprocess_features.py # Encoding, scaling, train/test split
│   ├── train_late_model.py     # Late model training (1+ days)
│   ├── train_very_late_model.py # Very late model training (3+ days)
│   └── logger.py          # Centralized logging configuration
├── data/                  # Data storage with clear separation
│   ├── raw/              # Original datasets
│   ├── unprocessed/      # Cleaned data before encoding
│   ├── preprocessed/     # Final processed features for training
│   └── docs/             # Data documentation
├── models/               # Trained models and preprocessing artifacts
├── tests/                # pytest test suite
├── tuning/               # Hyperparameter tuning scripts with MLflow
├── notebooks/            # Jupyter notebooks for EDA
├── logs/                 # Application and pipeline logs
└── mlruns/              # MLflow experiment tracking
```

## Code Organization Patterns

### Module Structure
- Each `src/` module has a single responsibility (loading, cleaning, training, etc.)
- All modules use the centralized logger from `src/logger.py`
- Consistent function signatures with clear input/output types
- Path handling uses `pathlib.Path` for cross-platform compatibility

### API Structure
- FastAPI app defined in `api/main.py` with router registration
- Individual routers in `routers/` directory for endpoint organization
- Pydantic schemas in `api/shipment_schema.py` for request validation
- Consistent error handling and JSON response formats

### Data Flow
1. Raw data → `src/load_data.py` → DataFrame
2. DataFrame → `src/clean_data.py` → Cleaned DataFrame  
3. Cleaned DataFrame → `src/feature_engineering.py` → Engineered DataFrame
4. Engineered DataFrame → `src/preprocess_features.py` → Train/test splits + encoders
5. Processed data → `src/train_*_model.py` → Trained models

### File Naming Conventions
- Snake_case for Python files and directories
- Descriptive module names indicating functionality
- Model files use `.pkl` extension with descriptive names
- Test files prefixed with `test_`
- Configuration files in root directory

### Logging Strategy
- Unified logging via `src/logger.py` across all modules
- File logging to `logs/pipeline.log` with timestamps
- Console logging for development feedback
- Different log levels (INFO, DEBUG, ERROR) for appropriate detail