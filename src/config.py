"""
Configuration settings for the recommender system.
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Database
DB_PATH = PROCESSED_DATA_DIR / "appliances_data.db"

# Data paths
REVIEWS_JSONL = RAW_DATA_DIR / "Appliances.jsonl" / "Appliances.jsonl"
PRODUCTS_JSONL = RAW_DATA_DIR / "meta_Appliances.jsonl" / "meta_Appliances.jsonl"

# Matrix Factorization Model Config
MF_MODEL_PATH = MODELS_DIR / "als_model.npz"
MF_MAPPINGS_PATH = MODELS_DIR / "mf_mappings.pkl"
MF_SPARSE_MATRIX_PATH = MODELS_DIR / "user_item_matrix.npz"

MF_CONFIG = {
    "factors": 128,
    "regularization": 1.0,
    "iterations": 100,
    "alpha": 10,
    "use_gpu": False,
}

# Similarity Search Config
SS_MODEL_NAME = "all-MiniLM-L6-v2"
SS_INDEX_PATH = MODELS_DIR / "faiss_index.bin"
SS_EMBEDDINGS_PATH = MODELS_DIR / "embeddings.npy"
SS_PRODUCT_IDS_PATH = MODELS_DIR / "product_ids.npy"
SS_BATCH_SIZE = 256

# Model Monitoring Config
OBSERVATIONS_PATH = PROCESSED_DATA_DIR / "observations.json"

# API Config
API_HOST = "0.0.0.0"
API_PORT = 8000
API_DEBUG = True
