# Amazon Appliances Recommender System
## End-to-End Machine Learning System for Product Recommendations

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready, full-stack machine learning recommender system built on the Amazon Appliances dataset. This project demonstrates a complete ML lifecycle from data ingestion and preprocessing through model training, serving, monitoring, and user interaction tracking.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Data Pipeline](#data-pipeline)
- [ML Models](#ml-models)
- [API Documentation](#api-documentation)
- [Frontend Interfaces](#frontend-interfaces)
- [Configuration](#configuration)
- [Performance](#performance)
- [Documentation](#documentation)
- [Development](#development)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This recommender system combines two powerful recommendation approaches:

1. **Collaborative Filtering** - Matrix Factorization using Alternating Least Squares (ALS) for personalized recommendations based on user behavior
2. **Content-Based Filtering** - Semantic similarity search using Sentence Transformers and FAISS for finding similar products

The system processes **2 million+ user ratings** across **90,000+ products** and provides:
- Real-time product recommendations
- Semantic product search
- User behavior tracking and monitoring
- Admin dashboard for data and model management
- RESTful API with comprehensive endpoints
- Interactive web interfaces

---

## Key Features

### Machine Learning
- **Collaborative Filtering**: ALS-based matrix factorization for personalized recommendations
- **Semantic Search**: Sentence Transformers + FAISS for content-based similarity
- **Dual Recommendation Strategy**: Item-to-item and user-to-item recommendations
- **Model Persistence**: Save/load trained models for efficient serving
- **On-Demand Training**: Retrain models via API with caching support

### Data Management
- **Database-Backed Storage**: SQLite with optimized schema and indexes
- **Bulk Data Operations**: Parallel JSONL parsing with 3-10x performance improvements
- **Foreign Key Constraints**: Referential integrity with CASCADE deletes
- **Admin Operations**: Insert/delete reviews and products via API
- **Data Statistics**: Real-time analytics on ratings, products, and categories

### API & Serving
- **FastAPI Framework**: Modern, async-capable REST API
- **Pydantic Validation**: Strict request/response validation
- **Multiple Endpoints**: Recommendations, search, admin operations, monitoring
- **CORS Support**: Frontend integration enabled
- **Auto-Documentation**: Swagger UI and ReDoc available

### User Tracking & Monitoring
- **Event Tracking**: Page loads, clicks, scrolls, purchases, and visits
- **Quality Metrics**: Good/bad/pending recommendation assessment based on user behavior
- **Observation Storage**: JSON-based tracking with quality signals
- **Admin Dashboard**: Monitor recommendation performance and export analytics

### Frontend
- **User Interface**: Product catalog with search and pagination
- **Product Detail Pages**: With recommendations and purchase tracking
- **Admin Dashboard**: Data management and monitoring tools
- **Session Management**: User ID tracking for personalized experiences

### Performance
- **Fast Inference**: <50ms for item recommendations, <30ms for search
- **Optimized Loading**: 2-5 minutes for 2M+ ratings (3-10x speedup)
- **Scalable Architecture**: Stateless API design for horizontal scaling
- **Memory Efficient**: ~500MB serving footprint per model

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RAW DATA LAYER                              │
│  Appliances.jsonl (2M+ ratings) | meta_Appliances.jsonl (94K prods) │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ETL & PREPROCESSING                            │
│  • Parallel JSONL Parsing (joblib)                                  │
│  • Single Transaction Bulk Insert                                   │
│  • Deferred Index Creation                                          │
│  • SQLite Performance Tuning (3-10x speedup)                        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DATABASE LAYER (SQLite)                        │
│  ┌──────────────────┐          ┌──────────────────────────────┐    │
│  │ product_catalog  │◄─────────┤ user_item_rating             │    │
│  │ (90K products)   │ FK       │ (2M+ ratings)                │    │
│  │ • parent_asin    │          │ • user_id, parent_asin       │    │
│  │ • title, price   │          │ • rating (1-5)               │    │
│  │ • images, desc   │          │ • verified_purchase          │    │
│  └──────────────────┘          └──────────────────────────────┘    │
│  Indexes: user_id, parent_asin, rating                              │
└────────────┬──────────────────────────────┬─────────────────────────┘
             │                              │
             ▼                              ▼
┌──────────────────────────┐    ┌──────────────────────────────┐
│   MATRIX FACTORIZATION   │    │   SIMILARITY SEARCH          │
│   (Collaborative)        │    │   (Content-Based)            │
├──────────────────────────┤    ├──────────────────────────────┤
│ • ALS Algorithm          │    │ • Sentence Transformers      │
│ • 128 latent factors     │    │ • 384-dim embeddings         │
│ • User & Item matrices   │    │ • FAISS IndexFlatIP          │
│ • Implicit feedback      │    │ • Cosine similarity          │
├──────────────────────────┤    ├──────────────────────────────┤
│ Training: 10-15 min      │    │ Training: 5-10 min           │
│ Size: ~200 MB            │    │ Size: ~150 MB                │
│ Inference: <50ms         │    │ Inference: <30ms             │
└────────────┬─────────────┘    └──────────────┬───────────────┘
             │                                  │
             └──────────────┬───────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       FASTAPI APPLICATION                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ Core Predict │  │ Admin APIs   │  │ Tracking & Monitoring    │  │
│  ├──────────────┤  ├──────────────┤  ├──────────────────────────┤  │
│  │ • /similar   │  │ • /reviews   │  │ • /tracking/event        │  │
│  │ • /user      │  │ • /products  │  │ • /monitoring/           │  │
│  │ • /search    │  │ • /stats     │  │   observations           │  │
│  │ • /training  │  │ • Insert/Del │  │ • Quality metrics        │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
└────────────┬──────────────────────────────────────────────────────┬─┘
             │                                                      │
             ▼                                                      ▼
┌─────────────────────────────┐                    ┌────────────────────┐
│      WEB FRONTEND           │                    │   ADMIN PANEL      │
├─────────────────────────────┤                    ├────────────────────┤
│ • login.html (auth)         │                    │ • admin.html       │
│ • landing.html (catalog)    │                    │ • Monitoring tab   │
│ • product.html (detail)     │                    │ • Data mgmt tab    │
│ • Search & pagination       │                    │ • Statistics       │
│ • Scroll/click tracking     │                    │ • CSV export       │
└─────────────────────────────┘                    └────────────────────┘
```

### Design Principles

1. **Separation of Training & Inference**: Expensive model training is decoupled from fast prediction serving
2. **Database-First Approach**: All data flows through structured storage for versioning and audit trails
3. **Stateless API**: Models cached in memory but API remains horizontally scalable
4. **Configuration-Driven**: All hyperparameters centralized in `config.py`
5. **Foreign Key Integrity**: CASCADE deletes prevent orphaned data

---

## Technology Stack

### Core ML & Data Science
| Technology | Version | Purpose |
|------------|---------|---------|
| **NumPy** | 1.24+ | Numerical computing and array operations |
| **Pandas** | 2.0+ | Data manipulation and analysis |
| **SciPy** | 1.10+ | Sparse matrices and scientific computing |
| **Implicit** | 0.7+ | ALS collaborative filtering implementation |
| **Sentence Transformers** | 2.2+ | Text embeddings (all-MiniLM-L6-v2) |
| **FAISS-CPU** | 1.7+ | Fast vector similarity search |
| **scikit-learn** | 1.3+ | ML utilities and metrics |

### Backend & API
| Technology | Version | Purpose |
|------------|---------|---------|
| **FastAPI** | 0.100+ | Modern async web framework |
| **Uvicorn** | 0.23+ | ASGI server with performance optimizations |
| **Pydantic** | 2.0+ | Data validation and serialization |
| **SQLite** | 3.x | Embedded relational database |

### Frontend
| Technology | Purpose |
|------------|---------|
| **HTML5/CSS3** | User interfaces |
| **Vanilla JavaScript** | Interactive features and API calls |
| **Session Storage** | User session management |

### Utilities
| Technology | Version | Purpose |
|------------|---------|---------|
| **tqdm** | 4.65+ | Progress bars for long operations |
| **joblib** | 1.3+ | Parallel processing for data loading |

### Development Tools (Optional)
| Technology | Version | Purpose |
|------------|---------|---------|
| **pytest** | 7.4+ | Testing framework |
| **black** | 23.0+ | Code formatting |
| **flake8** | 6.0+ | Linting |

---

## Project Structure

```
recommender_ml_system/
│
├── data/                           # Data storage (gitignored)
│   ├── raw/                        # Raw JSONL files
│   │   ├── Appliances.jsonl/       # 2M+ user ratings
│   │   └── meta_Appliances.jsonl/  # 94K product metadata
│   └── processed/                  # Processed data
│       ├── appliances_data.db      # SQLite database (500MB)
│       └── observations.json       # User behavior tracking
│
├── src/                            # Source code
│   ├── config.py                   # Centralized configuration
│   ├── etl/                        # Data loading & processing
│   │   └── load_data.py            # Optimized bulk loader (3-10x speedup)
│   ├── models/                     # ML model implementations
│   │   ├── matrix_factorization.py # ALS collaborative filtering
│   │   └── similarity_search.py    # Semantic search with embeddings
│   ├── api/                        # FastAPI application
│   │   ├── app.py                  # Main FastAPI app
│   │   ├── endpoints.py            # Core prediction endpoints
│   │   ├── admin_endpoints.py      # Data lifecycle management
│   │   └── monitoring_endpoints.py # Tracking & observability
│   └── utils/                      # Utilities
│       └── db_helper.py            # Database operations
│
├── models/                         # Saved ML artifacts (gitignored)
│   ├── als_model.npz               # User & item factor matrices
│   ├── mf_mappings.pkl             # ID to index mappings
│   ├── user_item_matrix.npz        # Sparse rating matrix
│   ├── faiss_index.bin             # FAISS vector index
│   ├── embeddings.npy              # Product title embeddings (384-dim)
│   └── product_ids.npy             # Product metadata
│
├── ui/                             # Frontend web interfaces
│   ├── login.html                  # User/Admin authentication
│   ├── landing.html                # Product catalog with search
│   ├── product.html                # Product details + recommendations
│   └── admin.html                  # Admin dashboard (monitoring + data)
│
├── notebooks/                      # Jupyter notebooks (experimentation)
├── tests/                          # Unit tests (pytest)
│
├── load_data.py                    # Script: Load data into database
├── train_models.py                 # Script: Train ML models
├── run_server.py                   # Script: Start API server
│
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git exclusions
│
└── Documentation/
    ├── README.md                   # This file
    ├── QUICKSTART.md               # 5-step setup guide
    ├── PROJECT_SUMMARY.md          # Architecture overview
    ├── DATABASE_SCHEMA.md          # Database documentation
    ├── ADMIN_API_DOCUMENTATION.md  # Admin endpoint guide
    └── PERFORMANCE_OPTIMIZATIONS.md # Performance details
```

---

## Quick Start

### Prerequisites

- **Python 3.8+**
- **8GB+ RAM** (16GB recommended for model training)
- **5GB+ Disk Space** (for data and models)

### Installation & Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd appliances/recommender_ml_system

# 2. Create virtual environment
python -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Prepare raw data
# Place Appliances.jsonl and meta_Appliances.jsonl in data/raw/
# Expected structure:
#   data/raw/Appliances.jsonl/Appliances.jsonl
#   data/raw/meta_Appliances.jsonl/meta_Appliances.jsonl
```

### Data Loading (2-5 minutes)

```bash
# Load 2M+ ratings and 90K+ products into SQLite
python load_data.py

# Output:
# - data/processed/appliances_data.db (500MB)
# - Database with optimized indexes
# - Performance: ~13,500 records/sec
```

### Model Training (15-25 minutes)

```bash
# Train both ML models
python train_models.py

# This trains:
# 1. Matrix Factorization (ALS) - 10-15 min
# 2. Similarity Search (Embeddings) - 5-10 min

# Output:
# - models/als_model.npz
# - models/mf_mappings.pkl
# - models/user_item_matrix.npz
# - models/faiss_index.bin
# - models/embeddings.npy
# - models/product_ids.npy
```

### Start API Server

```bash
# Run development server
python run_server.py

# Server starts at http://localhost:8000
# API Documentation: http://localhost:8000/docs
# Alternative docs: http://localhost:8000/redoc
```

### Access Web Interfaces

Open in browser:
- **User Interface**: http://localhost:8000/ui/login.html
- **Admin Dashboard**: http://localhost:8000/ui/admin.html
- **API Docs (Swagger)**: http://localhost:8000/docs

---

## Data Pipeline

### Data Sources

| Dataset | Records | Description |
|---------|---------|-------------|
| **Appliances.jsonl** | 2,016,674 | User ratings (1-5 scale) with timestamps |
| **meta_Appliances.jsonl** | 94,327 | Product metadata (titles, images, prices) |

**Final Statistics**:
- **Total Ratings**: 2,016,674
- **Unique Users**: 1,692,877
- **Unique Products**: 90,276
- **Average Rating**: 4.23
- **Database Size**: ~500 MB

### Database Schema

#### Table: `product_catalog`

```sql
CREATE TABLE product_catalog (
    parent_asin TEXT PRIMARY KEY,
    title TEXT,
    main_category TEXT,
    average_rating REAL,
    rating_number INTEGER,
    price TEXT,
    store TEXT,
    features TEXT,        -- JSON array
    description TEXT,     -- JSON array
    images TEXT,          -- JSON array
    categories TEXT,      -- JSON array
    details TEXT          -- JSON object
);
```

#### Table: `user_item_rating`

```sql
CREATE TABLE user_item_rating (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    parent_asin TEXT NOT NULL,
    rating REAL NOT NULL,
    timestamp BIGINT,
    verified_purchase INTEGER,
    helpful_vote INTEGER,
    FOREIGN KEY (parent_asin) REFERENCES product_catalog(parent_asin)
        ON DELETE CASCADE,
    UNIQUE(user_id, parent_asin)
);

CREATE INDEX idx_user_rating ON user_item_rating(user_id);
CREATE INDEX idx_item_rating ON user_item_rating(parent_asin);
CREATE INDEX idx_rating_value ON user_item_rating(rating);
```

### Performance Optimizations

The data loading pipeline implements **6 major optimizations** for 3-10x speedup:

1. **Parallel JSONL Parsing**: Uses joblib for multi-core processing
2. **Single Transaction**: All inserts in one atomic transaction
3. **Deferred Index Creation**: Indexes built after bulk insert (40-60% faster)
4. **SQLite Performance Pragmas**:
   ```python
   PRAGMA synchronous = OFF
   PRAGMA journal_mode = MEMORY
   PRAGMA cache_size = 100000
   PRAGMA temp_store = MEMORY
   ```
5. **Batch Insert Optimization**: Efficient executemany() calls
6. **Smart Pre-validation**: Early error detection before bulk insert

**Result**: 13,506 records/sec overall throughput (2-5 minutes for full dataset)

See [PERFORMANCE_OPTIMIZATIONS.md](PERFORMANCE_OPTIMIZATIONS.md) for details.

---

## ML Models

### Model 1: Matrix Factorization (Collaborative Filtering)

**Algorithm**: Alternating Least Squares (ALS) from `implicit` library

**Approach**:
- Decomposes user-item rating matrix into user and item latent factor matrices
- Learns representations that capture user preferences and item characteristics
- Uses implicit feedback signals from ratings

**Configuration** ([src/config.py](src/config.py)):
```python
MF_CONFIG = {
    "factors": 128,           # Latent dimension size
    "regularization": 1.0,    # L2 regularization strength
    "iterations": 100,        # Training iterations
    "alpha": 10,              # Confidence scaling for implicit feedback
    "use_gpu": False,         # GPU acceleration (optional)
}
```

**Training Process**:
1. Convert ratings to sparse CSR matrix (SciPy)
2. Map string IDs (user_id, parent_asin) to numeric indices
3. Train ALS model on sparse matrix
4. Save factor matrices, mappings, and sparse matrix

**Inference Capabilities**:
- **Item-to-Item**: Find similar products based on latent factors
- **User-to-Item**: Generate personalized recommendations for users
- **Filter Options**: Exclude already-rated items

**Performance**:
- **Training Time**: 10-15 minutes
- **Model Size**: ~200 MB
- **Item Similarity**: <50ms per request
- **User Recommendations**: <100ms per request
- **Memory**: 2GB during training, 500MB serving

**Files**:
- `models/als_model.npz` - User and item factor matrices
- `models/mf_mappings.pkl` - ID to index mappings
- `models/user_item_matrix.npz` - Sparse rating matrix

### Model 2: Similarity Search (Content-Based)

**Algorithm**: Sentence Transformers + FAISS

**Approach**:
- Encode product titles into dense 384-dimensional embeddings
- Build FAISS index for fast similarity search
- Use cosine similarity for semantic matching

**Configuration** ([src/config.py](src/config.py)):
```python
SS_MODEL_NAME = "all-MiniLM-L6-v2"  # Sentence transformer model
SS_BATCH_SIZE = 256                  # Batch encoding for efficiency
```

**Training Process**:
1. Load all product titles from database
2. Encode titles using pre-trained Sentence Transformer
3. Normalize embeddings for cosine similarity
4. Build FAISS IndexFlatIP (inner product index)
5. Save index, embeddings, and product metadata

**Inference Capabilities**:
- **Semantic Search**: Natural language queries → relevant products
- **Similarity Scoring**: Cosine similarity scores (0-1)
- **Top-K Retrieval**: Configurable result count (1-5000)

**Performance**:
- **Training Time**: 5-10 minutes
- **Model Size**: ~150 MB
- **Search Latency**: <30ms per query
- **Memory**: ~500 MB serving
- **Embedding Dimension**: 384

**Files**:
- `models/faiss_index.bin` - FAISS vector index
- `models/embeddings.npy` - Product embeddings (90K × 384)
- `models/product_ids.npy` - Product metadata (IDs, titles)

### Model Comparison

| Feature | Matrix Factorization | Similarity Search |
|---------|---------------------|-------------------|
| **Type** | Collaborative | Content-Based |
| **Input** | User-item ratings | Product titles |
| **Output** | Item/user recommendations | Semantic search results |
| **Personalization** | Yes (user-specific) | No (query-specific) |
| **Cold Start** | Poor (needs user history) | Good (works for new items) |
| **Latency** | 50-100ms | <30ms |
| **Use Case** | "Users also liked" | "Search products" |

---

## API Documentation

### Base URL

```
http://localhost:8000
```

### Core Prediction Endpoints

#### 1. System Status

```bash
GET /status

# Returns: Database stats, model readiness, embedding dimensions
```

**Response**:
```json
{
  "database": {
    "total_ratings": 2016674,
    "unique_users": 1692877,
    "unique_products": 90276,
    "average_rating": 4.23
  },
  "models": {
    "mf_ready": true,
    "ss_ready": true,
    "embedding_dim": 384
  }
}
```

#### 2. Train Models

```bash
POST /training
Content-Type: application/json

{
  "model_type": "mf",           # "mf" | "ss" | "both"
  "force_retrain": false        # true to force retrain
}

# Returns: Training status and duration
```

#### 3. Item-to-Item Recommendations (Collaborative)

```bash
POST /matrix_factor/similar
Content-Type: application/json

{
  "item_id": "B08C9LPCQV",
  "n": 5
}
```

**Response**:
```json
{
  "item_id": "B08C9LPCQV",
  "recommendations": [
    {
      "item_id": "B07XYZ1234",
      "score": 0.92,
      "title": "Portable Air Fryer",
      "image_url": "https://..."
    },
    ...
  ]
}
```

#### 4. User Recommendations (Collaborative)

```bash
POST /matrix_factor/user
Content-Type: application/json

{
  "user_id": "USER_12345",
  "n": 10,
  "filter_already_liked": true
}
```

**Response**:
```json
{
  "user_id": "USER_12345",
  "recommendations": [
    {
      "item_id": "B09ABC5678",
      "score": 0.87,
      "title": "Smart Thermostat",
      "price": "$89.99",
      "image_url": "https://..."
    },
    ...
  ]
}
```

#### 5. Semantic Search (Content-Based)

```bash
POST /search
Content-Type: application/json

{
  "query": "washing machine energy efficient",
  "top_k": 10
}
```

**Response**:
```json
{
  "query": "washing machine energy efficient",
  "results": [
    {
      "parent_asin": "B08DEF9012",
      "similarity": 0.89,
      "title": "Energy Star Washing Machine",
      "price": "$599.99",
      "average_rating": 4.6,
      "rating_number": 1523
    },
    ...
  ]
}
```

#### 6. Get All Products

```bash
GET /products/all

# Returns: Array of all products with full metadata
```

#### 7. Get Product Details

```bash
GET /products/{product_id}

# Example: GET /products/B08C9LPCQV
```

### Admin Data Management Endpoints

**Review Operations**:

```bash
# Insert reviews from JSON
POST /admin/reviews/insert
{"reviews": [{...}]}

# Bulk insert from CSV
POST /admin/reviews/insert-csv
{"csv_data": "user_id,parent_asin,rating,..."}

# Delete first N reviews
DELETE /admin/reviews/first/{n}

# Delete last N reviews
DELETE /admin/reviews/last/{n}

# Delete by specific IDs
DELETE /admin/reviews/by-ids
{"ids": [1, 2, 3]}

# Delete by user IDs (e.g., account closure)
DELETE /admin/reviews/by-users
{"user_ids": ["USER_1", "USER_2"]}
```

**Product Operations**:

```bash
# Insert products from JSON
POST /admin/products/insert
{"products": [{...}]}

# Bulk insert from CSV
POST /admin/products/insert-csv
{"csv_data": "parent_asin,title,price,..."}

# Delete first N products (CASCADE to reviews)
DELETE /admin/products/first/{n}

# Delete last N products (CASCADE to reviews)
DELETE /admin/products/last/{n}

# Delete by ASINs (CASCADE to reviews)
DELETE /admin/products/by-asins
{"asins": ["B08C9LPCQV", "B07XYZ1234"]}

# Get product statistics by category
GET /admin/products/stats
```

### Monitoring & Tracking Endpoints

```bash
# Track user events
POST /tracking/event
{
  "user_id": "USER_123",
  "event_type": "click_product",  # page_load | click_product | scroll_end | purchase | page_visit
  "key": "B08C9LPCQV",
  "key_type": "product",
  "list_of_recommendations": ["B08C9LPCQV", ...],
  "rating": 5,
  "timestamp": 1234567890
}

# Get all observations with quality metrics
GET /monitoring/observations

# Clear observation records
DELETE /monitoring/observations
```

**Event Types & Quality Signals**:
- `page_load` - Recommendations displayed
- `click_product` - User clicked recommendation (**GOOD** signal)
- `scroll_end` - User scrolled to end without clicking (**BAD** signal)
- `purchase` - User purchased product (**GOOD** signal)
- `page_visit` - User visited product page

See [ADMIN_API_DOCUMENTATION.md](ADMIN_API_DOCUMENTATION.md) for complete endpoint documentation.

---

## Frontend Interfaces

### 1. Login Page ([ui/login.html](ui/login.html))

**Purpose**: User/Admin authentication entry point

**Features**:
- "Login as User" - Generates unique UUID for session
- "Login as Admin" - Access to admin dashboard
- Session storage for user ID

**Access**: http://localhost:8000/ui/login.html

### 2. Landing Page ([ui/landing.html](ui/landing.html))

**Purpose**: Product catalog with search and browsing

**Features**:
- **Semantic Search**: Real-time product search with similarity scores
- **Pagination**: 100 products per page
- **Product Cards**: Images, titles, ratings, prices
- **User Tracking**:
  - Page load events
  - Click tracking on products
  - Scroll-to-end detection (bad signal if no clicks)
- **Session Persistence**: User ID from login

**Access**: http://localhost:8000/ui/landing.html

### 3. Product Detail Page ([ui/product.html](ui/product.html))

**Purpose**: Product details with recommendations

**Features**:
- **Product Information**: Title, image, rating, price, description, features
- **Collaborative Recommendations**: "Users also liked this" (ALS)
- **Content Recommendations**: "Similar products" (Semantic)
- **Purchase Flow**: Rating selector (1-5 stars) + purchase button
- **Tracking**:
  - Page visit events
  - Recommendation clicks
  - Scroll-to-end detection per section
  - Purchase events with ratings

**Access**: http://localhost:8000/ui/product.html?id=B08C9LPCQV

### 4. Admin Dashboard ([ui/admin.html](ui/admin.html))

**Purpose**: Model monitoring and data management

**Features**:

**Tab 1: Model Monitoring/Observability**
- **Statistics Cards**: Total observations, good/bad/pending counts, success rate
- **Observations Table**: User ID, key, type, event type, recommendations, quality, timestamp
- **Export Functionality**: Download good recommendations as CSV
- **Clear Data**: Reset observation records

**Tab 2: Data Management**
- **Insert Operations**:
  - Insert reviews from JSON or CSV
  - Insert products from JSON or CSV
- **Delete Operations**:
  - Delete first/last N reviews or products
  - Delete by specific IDs or ASINs
  - User account closure (delete all user reviews)
- **Analytics**: Product statistics by category

**Access**: http://localhost:8000/ui/admin.html

---

## Configuration

All configuration is centralized in [src/config.py](src/config.py).

### Directory Paths

```python
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
```

### Database

```python
DB_PATH = PROCESSED_DATA_DIR / "appliances_data.db"
```

### Data Files

```python
REVIEWS_JSONL = RAW_DATA_DIR / "Appliances.jsonl" / "Appliances.jsonl"
PRODUCTS_JSONL = RAW_DATA_DIR / "meta_Appliances.jsonl" / "meta_Appliances.jsonl"
```

### Model Paths

```python
# Matrix Factorization
MF_MODEL_PATH = MODELS_DIR / "als_model.npz"
MF_MAPPINGS_PATH = MODELS_DIR / "mf_mappings.pkl"
MF_SPARSE_MATRIX_PATH = MODELS_DIR / "user_item_matrix.npz"

# Similarity Search
SS_INDEX_PATH = MODELS_DIR / "faiss_index.bin"
SS_EMBEDDINGS_PATH = MODELS_DIR / "embeddings.npy"
SS_PRODUCT_IDS_PATH = MODELS_DIR / "product_ids.npy"
```

### Model Hyperparameters

```python
# Matrix Factorization Config
MF_CONFIG = {
    "factors": 128,           # Latent factors dimension
    "regularization": 1.0,    # L2 regularization
    "iterations": 100,        # Training iterations
    "alpha": 10,              # Confidence scaling
    "use_gpu": False,         # GPU support
}

# Similarity Search Config
SS_MODEL_NAME = "all-MiniLM-L6-v2"  # Sentence transformer
SS_BATCH_SIZE = 256                  # Encoding batch size
```

### API Server

```python
API_HOST = "0.0.0.0"
API_PORT = 8000
API_DEBUG = True
```

### Observations

```python
OBSERVATIONS_PATH = PROCESSED_DATA_DIR / "observations.json"
```

---

## Performance

### Data Loading Benchmarks

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Total Time** | 15-30 min | 2-5 min | **3-10x faster** |
| **Records/sec** | 1,800 | 13,506 | **7.5x faster** |
| **Reviews Loading** | 8-15 min | 1-2 min | **6-8x faster** |
| **Products Loading** | 7-15 min | 1-3 min | **5-7x faster** |

**Optimization Techniques**:
1. Parallel JSONL parsing with joblib
2. Single transaction for atomic operations
3. Deferred index creation (40-60% speedup)
4. SQLite performance pragmas
5. Batch insert optimization
6. Smart pre-validation

See [PERFORMANCE_OPTIMIZATIONS.md](PERFORMANCE_OPTIMIZATIONS.md) for technical details.

### Model Performance

| Model | Training Time | Model Size | Inference Latency | Memory (Serving) |
|-------|--------------|------------|-------------------|------------------|
| **Matrix Factorization** | 10-15 min | 200 MB | <50ms (item), <100ms (user) | 500 MB |
| **Similarity Search** | 5-10 min | 150 MB | <30ms | 500 MB |

### API Performance

- **Status Endpoint**: <10ms
- **Product Lookup**: <10ms (indexed)
- **Item Recommendations**: <50ms
- **User Recommendations**: <100ms
- **Semantic Search**: <30ms
- **Concurrent Requests**: Async-capable (FastAPI)

### Scalability

**Current Limits (SQLite)**:
- Database Size: ~500 MB (can handle 10M+ records with proper indexing)
- Concurrent Writes: Limited (single writer)
- Read Performance: Excellent (indexed queries <10ms)

**Production Scaling Path**:
1. **Database**: Migrate to PostgreSQL/MySQL for multi-writer support
2. **API**: Horizontal scaling with load balancer (stateless design)
3. **Caching**: Redis for model predictions and query results
4. **Containerization**: Docker + Kubernetes for orchestration
5. **Model Serving**: Separate training and serving infrastructure

---

## Documentation

This project includes comprehensive documentation:

| Document | Description |
|----------|-------------|
| [README.md](README.md) | **This file** - Complete project overview |
| [QUICKSTART.md](QUICKSTART.md) | 5-step setup guide for quick start |
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | Architecture overview and design decisions |
| [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md) | Database tables, constraints, indexes, queries |
| [ADMIN_API_DOCUMENTATION.md](ADMIN_API_DOCUMENTATION.md) | Admin endpoint reference with examples |
| [PERFORMANCE_OPTIMIZATIONS.md](PERFORMANCE_OPTIMIZATIONS.md) | Performance improvements and benchmarks |

**In-Code Documentation**:
- Docstrings in all modules and functions
- Inline comments for complex logic
- Type hints for clarity

**API Documentation**:
- **Swagger UI**: http://localhost:8000/docs (interactive)
- **ReDoc**: http://localhost:8000/redoc (reference)

---

## Development

### Development Workflow

```bash
# 1. Activate environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Make changes to code
# Edit files in src/

# 3. Test changes
# Run API server
python run_server.py

# Test endpoints
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "blender", "top_k": 5}'

# 4. Retrain models (if needed)
python train_models.py

# 5. Format code (optional)
black src/
flake8 src/
```

### Project Extension Ideas

**Machine Learning**:
- Hybrid models combining collaborative + content-based
- Deep learning (neural collaborative filtering)
- Contextual bandits for exploration/exploitation
- Time-aware recommendations
- Cold start strategies

**Data Pipeline**:
- Real-time data ingestion
- Feature store integration
- Data validation (Great Expectations)
- A/B testing framework

**Deployment**:
- Docker containerization
- Kubernetes orchestration
- CI/CD pipeline (GitHub Actions)
- Model registry (MLflow)
- Monitoring dashboard (Grafana)

**API**:
- Authentication & authorization (JWT)
- Rate limiting
- Caching layer (Redis)
- GraphQL support
- WebSocket for real-time updates

**Frontend**:
- React/Vue.js rewrite
- Mobile app (React Native)
- Advanced filtering and sorting
- Personalized homepage
- Shopping cart functionality

---

## Future Enhancements

### High Priority

1. **Unit Tests**: Add pytest test suite for critical functions
2. **Monitoring Dashboard**: Real-time metrics visualization (Grafana/Prometheus)
3. **Model Versioning**: MLflow integration for experiment tracking
4. **CI/CD Pipeline**: Automated testing and deployment
5. **Containerization**: Dockerfile and docker-compose for easy deployment

### Medium Priority

1. **Authentication**: JWT-based API authentication
2. **Rate Limiting**: Prevent API abuse
3. **Caching Layer**: Redis for frequently accessed predictions
4. **PostgreSQL Migration**: Production-grade database
5. **Model Explainability**: SHAP values for recommendation explanations

### Lower Priority

1. **Feature Store**: Centralized feature management (Feast)
2. **A/B Testing**: Framework for model comparison
3. **Advanced Analytics**: User segmentation, cohort analysis
4. **Recommendation Diversity**: Ensure diverse recommendations
5. **Multi-Model Ensemble**: Combine multiple recommendation strategies

---

## ML Lifecycle Coverage

This project demonstrates a complete ML lifecycle:

| Stage | Implementation | Status |
|-------|---------------|--------|
| **1. Data Acquisition** | JSONL files downloaded | ✅ |
| **2. Data Storage** | SQLite with optimized schema | ✅ |
| **3. Data Preprocessing** | ETL pipeline with validation | ✅ |
| **4. Feature Engineering** | Sparse matrices, embeddings | ✅ |
| **5. Model Training** | ALS + Sentence Transformers | ✅ |
| **6. Model Evaluation** | Statistics tracking | ✅ |
| **7. Model Persistence** | Save/load functionality | ✅ |
| **8. Model Serving** | REST API with FastAPI | ✅ |
| **9. Inference** | Real-time predictions | ✅ |
| **10. User Tracking** | Behavior monitoring | ✅ |
| **11. Data Lifecycle** | Insert/delete operations | ✅ |
| **12. Monitoring** | Observation quality metrics | ✅ |
| **13. Model Retraining** | On-demand via API | ✅ |
| **14. Testing** | Unit tests | ⏳ TODO |
| **15. Deployment** | Containerization | ⏳ TODO |

---

## Contributing

Contributions are welcome! Areas for improvement:

1. **Testing**: Write unit tests with pytest
2. **Documentation**: Expand docstrings and examples
3. **Performance**: Optimize model inference further
4. **Features**: Implement hybrid recommendation models
5. **DevOps**: Add Docker, CI/CD, monitoring

**Development Setup**:
```bash
# Fork and clone the repository
git clone <your-fork-url>
cd appliances/recommender_ml_system

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test
python run_server.py

# Commit and push
git add .
git commit -m "Add feature: description"
git push origin feature/your-feature-name

# Create pull request
```

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- **Amazon Reviews Dataset**: 2018 Amazon product reviews and metadata
- **Implicit Library**: Fast ALS implementation
- **Sentence Transformers**: Pre-trained embedding models
- **FAISS**: Efficient similarity search by Meta AI
- **FastAPI**: Modern Python web framework

---

## Contact & Support

For questions, issues, or suggestions:

- **GitHub Issues**: [Create an issue](https://github.com/your-repo/issues)
- **Documentation**: See docs/ folder for detailed guides
- **Email**: your-email@example.com

---

## Project Statistics

- **Lines of Code**: ~3,000+
- **Documentation**: 6 comprehensive guides
- **API Endpoints**: 20+ endpoints
- **Models**: 2 recommendation algorithms
- **Dataset Size**: 2M+ ratings, 90K+ products
- **Database**: 500 MB optimized SQLite
- **Performance**: <50ms inference latency

---

**Built with ❤️ for demonstrating end-to-end ML systems**

