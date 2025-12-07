# Recommender System - End-to-End ML Project

A production-ready recommender system built with Matrix Factorization (ALS) and Semantic Similarity Search (Sentence Transformers + FAISS) for product recommendations.

## Project Structure

```
recommender_ml_system/
├── data/
│   ├── raw/                      # Raw JSONL data files
│   └── processed/                # SQLite DB and processed data
├── src/
│   ├── config.py                 # Configuration settings
│   ├── etl/                      # Data loading and database
│   │   ├── database.py           # SQLite database operations
│   │   └── data_loader.py        # JSONL to DB loader
│   ├── models/                   # ML models
│   │   ├── matrix_factorization.py  # ALS collaborative filtering
│   │   └── similarity_search.py     # Semantic search with FAISS
│   ├── api/                      # FastAPI application
│   │   └── app.py                # API endpoints
│   └── utils/                    # Helper utilities
│       └── helpers.py
├── models/                       # Saved model artifacts
├── notebooks/                    # Jupyter notebooks
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Features

### 1. **Matrix Factorization (Collaborative Filtering)**
- Uses Implicit library's ALS algorithm
- Provides item-to-item recommendations ("customers who bought this also bought...")
- Provides personalized user recommendations
- Configurable hyperparameters (factors, regularization, iterations)

### 2. **Semantic Similarity Search**
- Uses Sentence Transformers (all-MiniLM-L6-v2)
- FAISS index for fast similarity search
- Search products by title or description
- Returns semantically similar products

### 3. **REST API**
- `/training` - Train or retrain models
- `/matrix_factor/similar` - Get similar items
- `/matrix_factor/user` - Get personalized recommendations
- `/search` - Semantic search for products
- `/status` - Check system status

### 4. **Database**
- SQLite database for structured data storage
- Tables: `user_item_rating`, `product_catalog`
- Indexed for fast queries

## Setup Instructions

### 1. Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### 2. Installation

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Data Preparation

Place your data files in the `data/raw/` directory:
```
data/raw/
├── Appliances.jsonl/
│   └── Appliances.jsonl          # Reviews data
└── meta_Appliances.jsonl/
    └── meta_Appliances.jsonl     # Product metadata
```

### 4. Load Data into Database

```bash
# From the project root directory
python -m src.etl.data_loader
```

This will:
- Create SQLite database at `data/processed/recommender.db`
- Load reviews into `user_item_rating` table
- Load products into `product_catalog` table
- Print statistics

## Usage

### Starting the API Server

```bash
# From the project root directory
python -m uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

Interactive API documentation: `http://localhost:8000/docs`

### API Endpoints

#### 1. Check Status
```bash
GET http://localhost:8000/status
```

Response:
```json
{
  "database": {
    "n_ratings": 2016674,
    "n_users": 1692877,
    "n_products": 90276,
    "avg_rating": 4.2
  },
  "matrix_factorization": {
    "status": "trained",
    "n_users": 1692877,
    "n_items": 90276,
    "n_factors": 128
  },
  "similarity_search": {
    "status": "trained",
    "n_products": 94327,
    "embedding_dimension": 384
  }
}
```

#### 2. Train Models

**Train Matrix Factorization:**
```bash
POST http://localhost:8000/training
Content-Type: application/json

{
  "model_type": "mf",
  "force_retrain": false
}
```

**Train Similarity Search:**
```bash
POST http://localhost:8000/training
Content-Type: application/json

{
  "model_type": "ss",
  "force_retrain": false
}
```

Response:
```json
{
  "status": "success",
  "message": "Matrix Factorization model trained successfully",
  "model_type": "mf",
  "stats": {
    "status": "trained",
    "n_users": 1692877,
    "n_items": 90276,
    "n_factors": 128
  }
}
```

#### 3. Get Similar Items (Matrix Factorization)

```bash
POST http://localhost:8000/matrix_factor/similar
Content-Type: application/json

{
  "item_id": "B08C9LPCQV",
  "n": 10
}
```

Response:
```json
{
  "recommendations": [
    {
      "item_id": "B08C9LPCQV",
      "score": 0.9999,
      "title": "Sikawai 279816 Dryer Thermal Cut-off Kit...",
      "image_url": "https://m.media-amazon.com/images/..."
    },
    ...
  ]
}
```

#### 4. Get User Recommendations

```bash
POST http://localhost:8000/matrix_factor/user
Content-Type: application/json

{
  "user_id": "AFQLNQNQYFWQZPJQZS6V3NZU4QBQ",
  "n": 10,
  "filter_already_liked": true
}
```

#### 5. Search Products (Semantic Search)

```bash
POST http://localhost:8000/search
Content-Type: application/json

{
  "query": "washing machine",
  "top_k": 10
}
```

Response:
```json
{
  "results": [
    {
      "product_id": "B087Z26VJP",
      "title": "Washer",
      "score": 0.7829,
      "image_url": "https://m.media-amazon.com/images/..."
    },
    ...
  ]
}
```

## Model Configuration

Edit `src/config.py` to customize model parameters:

```python
# Matrix Factorization Config
MF_CONFIG = {
    "factors": 128,           # Number of latent factors
    "regularization": 1.0,    # L2 regularization
    "iterations": 100,        # Training iterations
    "alpha": 10,              # Confidence scaling
    "use_gpu": False,         # Use GPU if available
}

# Similarity Search Config
SS_MODEL_NAME = "all-MiniLM-L6-v2"  # Sentence transformer model
SS_BATCH_SIZE = 256                  # Encoding batch size
```

## Development Workflow

### 1. Data Exploration
Use Jupyter notebooks in the `notebooks/` directory for experimentation.

### 2. Model Training
- Load data into database using ETL pipeline
- Train models via API or directly in Python
- Models are saved automatically to `models/` directory

### 3. Model Serving
- Models are loaded on-demand by the API
- Cached in memory for fast inference
- Can be retrained via API endpoint

### 4. Integration
- API endpoints can be consumed by frontend applications
- Supports both real-time inference and batch processing

## Future Extensions (Full ML Lifecycle)

This project is a foundation for a complete ML lifecycle. Future additions could include:

1. **Model Monitoring**
   - Track prediction latency
   - Monitor recommendation quality metrics
   - A/B testing framework

2. **Model Versioning**
   - MLflow or DVC integration
   - Model registry
   - Experiment tracking

3. **Automated Retraining**
   - Scheduled training jobs
   - Incremental learning
   - Data drift detection

4. **CI/CD Pipeline**
   - Automated testing
   - Model validation
   - Deployment automation

5. **Feature Store**
   - Centralized feature management
   - Real-time feature computation
   - Feature versioning

6. **Orchestration**
   - Apache Airflow for workflow management
   - Scheduled data ingestion
   - Model retraining pipelines

7. **Scalability**
   - Containerization (Docker)
   - Kubernetes deployment
   - Load balancing

## Testing

```bash
# Run tests (when implemented)
pytest tests/
```

## Troubleshooting

### Issue: "Model not trained"
**Solution:** Run the training endpoint first:
```bash
POST /training with {"model_type": "mf"} or {"model_type": "ss"}
```

### Issue: "Item/User not found"
**Solution:** The item or user ID doesn't exist in the training data. Check the database or use valid IDs.

### Issue: Database not found
**Solution:** Run the data loader:
```bash
python -m src.etl.data_loader
```

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

For questions or support, please open an issue in the repository.
