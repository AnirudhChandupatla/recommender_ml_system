# Quick Start Guide

Get the recommender system up and running in 5 steps!

## Step 1: Setup Environment

```bash
# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Prepare Data

Copy your data files to the `data/raw/` directory:

```
data/raw/
├── Appliances.jsonl/
│   └── Appliances.jsonl
└── meta_Appliances.jsonl/
    └── meta_Appliances.jsonl
```

If your data is in the parent directory, you can create symbolic links:

```bash
# Windows (run as Administrator)
mklink /D data\raw\Appliances.jsonl ..\Appliances.jsonl
mklink /D data\raw\meta_Appliances.jsonl ..\meta_Appliances.jsonl

# Linux/Mac
ln -s ../Appliances.jsonl data/raw/Appliances.jsonl
ln -s ../meta_Appliances.jsonl data/raw/meta_Appliances.jsonl
```

## Step 3: Load Data into Database

```bash
python load_data.py
```

This creates a SQLite database with all reviews and product metadata. Takes ~2-5 minutes.

## Step 4: Train Models

```bash
python train_models.py
```

Choose option 3 to train both models:
- Matrix Factorization (MF): ~10-15 minutes
- Similarity Search (SS): ~5-10 minutes

## Step 5: Start API Server

```bash
python run_server.py
```

The server will start at `http://localhost:8000`

## Testing the API

### Option A: Interactive Docs
Open your browser and go to: `http://localhost:8000/docs`

This provides a Swagger UI where you can test all endpoints interactively.

### Option B: Command Line (curl)

```bash
# Check status
curl http://localhost:8000/status

# Get similar items
curl -X POST http://localhost:8000/matrix_factor/similar \
  -H "Content-Type: application/json" \
  -d '{"item_id": "B08C9LPCQV", "n": 5}'

# Search products
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "washing machine", "top_k": 5}'

# Get user recommendations
curl -X POST http://localhost:8000/matrix_factor/user \
  -H "Content-Type: application/json" \
  -d '{"user_id": "AFQLNQNQYFWQZPJQZS6V3NZU4QBQ", "n": 5}'
```

### Option C: Python

```python
import requests

# Search for products
response = requests.post(
    "http://localhost:8000/search",
    json={"query": "washing machine", "top_k": 5}
)
print(response.json())

# Get similar items
response = requests.post(
    "http://localhost:8000/matrix_factor/similar",
    json={"item_id": "B08C9LPCQV", "n": 5}
)
print(response.json())
```

## What's Next?

- Read the full [README.md](README.md) for detailed documentation
- Explore the code in `src/` directory
- Customize model parameters in `src/config.py`
- Build a frontend UI to consume these APIs
- Extend with monitoring, A/B testing, and automated retraining

## Common Issues

**Issue**: `ModuleNotFoundError: No module named 'src'`
**Solution**: Make sure you're running scripts from the project root directory.

**Issue**: `FileNotFoundError: [Errno 2] No such file or directory`
**Solution**: Check that data files are in `data/raw/` directory.

**Issue**: Database already exists
**Solution**: The script will ask if you want to overwrite. Say 'yes' to reload data.

**Issue**: Model takes too long to train
**Solution**: Reduce the dataset size in `load_data.py` by sampling fewer records.

## Performance Tips

For faster training on large datasets:

1. **Use GPU** (if available):
   - In `src/config.py`, set `"use_gpu": True` in `MF_CONFIG`
   - Install `faiss-gpu` instead of `faiss-cpu`

2. **Reduce iterations**:
   - In `src/config.py`, set `"iterations": 50` instead of 100

3. **Sample data**:
   - In `load_data.py`, add sampling logic to load fewer records

Happy recommending! 🚀
