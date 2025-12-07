"""
FastAPI application for the recommender system.
Provides endpoints for training and inference.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import tempfile
from pathlib import Path

from ..config import (
    DB_PATH, MF_CONFIG, MF_MODEL_PATH, MF_MAPPINGS_PATH, MF_SPARSE_MATRIX_PATH,
    SS_MODEL_NAME, SS_INDEX_PATH, SS_EMBEDDINGS_PATH, SS_PRODUCT_IDS_PATH, SS_BATCH_SIZE
)
from ..etl.database import RecommenderDB
from ..models.matrix_factorization import MatrixFactorizationModel
from ..models.similarity_search import SimilaritySearchModel


# Initialize FastAPI app
app = FastAPI(
    title="Recommender System API",
    description="API for product recommendations using Matrix Factorization and Similarity Search",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (UI directory)
# Get the path to the UI directory (two levels up from this file)
UI_DIR = Path(__file__).parent.parent.parent / "ui"
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")

# Global model instances
mf_model = None
ss_model = None
db = None


# Request/Response Models
class TrainingRequest(BaseModel):
    """Request model for training endpoint."""
    model_type: str = Field(..., description="Type of model to train: 'mf' or 'ss'")
    force_retrain: bool = Field(False, description="Force retraining even if model exists")


class TrainingResponse(BaseModel):
    """Response model for training endpoint."""
    status: str
    message: str
    model_type: str
    stats: Optional[Dict[str, Any]] = None


class SimilarItemsRequest(BaseModel):
    """Request model for similar items recommendation."""
    item_id: str = Field(..., description="Product ASIN")
    n: int = Field(10, ge=1, le=50, description="Number of recommendations")


class UserRecommendationRequest(BaseModel):
    """Request model for user-based recommendations."""
    user_id: str = Field(..., description="User identifier")
    n: int = Field(10, ge=1, le=50, description="Number of recommendations")
    filter_already_liked: bool = Field(True, description="Filter items user has already rated")


class SearchRequest(BaseModel):
    """Request model for similarity search."""
    query: str = Field(..., description="Search query")
    top_k: Optional[int] = Field(default=None, description="Number of results (1-5000). If not provided, defaults to 1000. Use GET /products/all for the full catalog.")


class RecommendationItem(BaseModel):
    """Single recommendation item."""
    item_id: str
    score: float
    title: Optional[str] = None
    image_url: Optional[str] = None


class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    recommendations: List[RecommendationItem]


class SearchResult(BaseModel):
    """Single search result."""
    product_id: str
    title: str
    score: float
    image_url: Optional[str] = None
    rating: Optional[float] = None
    price: Optional[float] = None
    description: Optional[List[str]] = None


class SearchResponse(BaseModel):
    """Response model for search."""
    results: List[SearchResult]


# ============================================================
# ADMIN API MODELS FOR DATA LIFECYCLE TESTING
# ============================================================

class InsertReviewsRequest(BaseModel):
    """Request model for inserting reviews."""
    reviews: List[Dict[str, Any]] = Field(..., description="List of review records")


class InsertProductsRequest(BaseModel):
    """Request model for inserting products."""
    products: List[Dict[str, Any]] = Field(..., description="List of product records")


class DeleteByIdsRequest(BaseModel):
    """Request model for deleting by IDs."""
    ids: List[int] = Field(..., description="List of review IDs to delete")


class DeleteByUserIdsRequest(BaseModel):
    """Request model for deleting reviews by user IDs."""
    user_ids: List[str] = Field(..., description="List of user IDs to delete reviews for")


class DeleteByAsinsRequest(BaseModel):
    """Request model for deleting products by ASINs."""
    asins: List[str] = Field(..., description="List of product ASINs to delete")


class AdminOperationResponse(BaseModel):
    """Response model for admin operations."""
    status: str
    message: str
    rows_affected: int
    db_stats: Optional[Dict[str, Any]] = None


# Helper Functions
def initialize_db():
    """Initialize database connection."""
    global db
    if db is None:
        db = RecommenderDB(str(DB_PATH))
    return db


def load_mf_model():
    """Load or initialize Matrix Factorization model."""
    global mf_model
    if mf_model is None:
        mf_model = MatrixFactorizationModel(MF_CONFIG)
        # Try to load existing model
        if MF_MODEL_PATH.exists() and MF_MAPPINGS_PATH.exists() and MF_SPARSE_MATRIX_PATH.exists():
            try:
                mf_model.load_model(MF_MODEL_PATH, MF_MAPPINGS_PATH, MF_SPARSE_MATRIX_PATH)
            except Exception as e:
                print(f"Could not load existing MF model: {e}")
    return mf_model


def load_ss_model():
    """Load or initialize Similarity Search model."""
    global ss_model
    if ss_model is None:
        ss_model = SimilaritySearchModel(model_name=SS_MODEL_NAME, batch_size=SS_BATCH_SIZE)
        # Try to load existing model
        if SS_INDEX_PATH.exists() and SS_EMBEDDINGS_PATH.exists() and SS_PRODUCT_IDS_PATH.exists():
            try:
                ss_model.load_model(SS_INDEX_PATH, SS_EMBEDDINGS_PATH, SS_PRODUCT_IDS_PATH)
            except Exception as e:
                print(f"Could not load existing SS model: {e}")
    return ss_model


def enrich_with_product_info(recommendations: List[Dict[str, Any]], db: RecommenderDB) -> List[RecommendationItem]:
    """
    Enrich recommendations with product information from database.

    Args:
        recommendations: List of recommendation dicts with item_id and score
        db: Database instance

    Returns:
        List of enriched RecommendationItem objects
    """
    if not recommendations:
        return []

    # Batch lookup all products in a single query for performance
    item_ids = [rec['item_id'] for rec in recommendations]
    products_map = db.get_products_by_asins(item_ids)

    enriched = []
    for rec in recommendations:
        item_id = rec['item_id']
        product = products_map.get(item_id)

        image_url = None
        title = None
        if product:
            title = product.get('title')
            images = product.get('images', [])
            if images and len(images) > 0:
                image_url = images[0].get('large') or images[0].get('thumb')

        enriched.append(RecommendationItem(
            item_id=item_id,
            score=rec['score'],
            title=title,
            image_url=image_url
        ))

    return enriched


def enrich_search_results(results: List[Dict[str, Any]], db: RecommenderDB) -> List[SearchResult]:
    """
    Enrich search results with product details.

    Args:
        results: List of search result dicts
        db: Database instance

    Returns:
        List of enriched SearchResult objects
    """
    if not results:
        return []

    # Batch lookup all products in a single query for performance
    product_ids = [result['product_id'] for result in results]
    products_map = db.get_products_by_asins(product_ids)

    enriched = []
    for result in results:
        product_id = result['product_id']
        product = products_map.get(product_id)

        image_url = None
        rating = None
        price = None
        description = None

        if product:
            images = product.get('images', [])
            if images and len(images) > 0:
                image_url = images[0].get('large') or images[0].get('thumb')

            rating = product.get('average_rating')
            price = product.get('price')
            description = product.get('description')

        enriched.append(SearchResult(
            product_id=product_id,
            title=result['title'],
            score=result['score'],
            image_url=image_url,
            rating=rating,
            price=price,
            description=description
        ))

    return enriched


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Recommender System API",
        "version": "1.0.0",
        "endpoints": {
            "training": "/training",
            "matrix_factor_similar": "/matrix_factor/similar",
            "matrix_factor_user": "/matrix_factor/user",
            "search": "/search",
            "products_all": "/products/all",
            "status": "/status",
            "ui": "/ui/login.html"
        }
    }


@app.get("/status")
async def get_status():
    """Get the status of all models."""
    db = initialize_db()
    db_stats = db.get_stats()

    mf_stats = {"status": "not_loaded"}
    if mf_model is not None:
        mf_stats = mf_model.get_stats()

    ss_stats = {"status": "not_loaded"}
    if ss_model is not None:
        ss_stats = ss_model.get_stats()

    return {
        "database": db_stats,
        "matrix_factorization": mf_stats,
        "similarity_search": ss_stats
    }


@app.post("/training", response_model=TrainingResponse)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Train a model (either Matrix Factorization or Similarity Search).

    Args:
        request: Training request with model_type ('mf' or 'ss')

    Returns:
        Training response with status and statistics
    """
    model_type = request.model_type.lower()

    if model_type not in ['mf', 'ss']:
        raise HTTPException(status_code=400, detail="model_type must be 'mf' or 'ss'")

    db = initialize_db()

    try:
        if model_type == 'mf':
            # Train Matrix Factorization model
            model = MatrixFactorizationModel(MF_CONFIG)

            # Check if model already exists
            if not request.force_retrain and MF_MODEL_PATH.exists():
                model.load_model(MF_MODEL_PATH, MF_MAPPINGS_PATH, MF_SPARSE_MATRIX_PATH)
                return TrainingResponse(
                    status="loaded",
                    message="Matrix Factorization model already exists. Use force_retrain=true to retrain.",
                    model_type="mf",
                    stats=model.get_stats()
                )

            # Get ratings from database
            ratings = db.get_all_ratings()
            if len(ratings) == 0:
                raise HTTPException(status_code=400, detail="No ratings found in database")

            # Train model
            model.train(ratings)
            model.save_model(MF_MODEL_PATH, MF_MAPPINGS_PATH, MF_SPARSE_MATRIX_PATH)

            # Update global model
            global mf_model
            mf_model = model

            return TrainingResponse(
                status="success",
                message="Matrix Factorization model trained successfully",
                model_type="mf",
                stats=model.get_stats()
            )

        else:  # model_type == 'ss'
            # Train Similarity Search model
            model = SimilaritySearchModel(model_name=SS_MODEL_NAME, batch_size=SS_BATCH_SIZE)

            # Check if model already exists
            if not request.force_retrain and SS_INDEX_PATH.exists():
                model.load_model(SS_INDEX_PATH, SS_EMBEDDINGS_PATH, SS_PRODUCT_IDS_PATH)
                return TrainingResponse(
                    status="loaded",
                    message="Similarity Search model already exists. Use force_retrain=true to retrain.",
                    model_type="ss",
                    stats=model.get_stats()
                )

            # Get products from database
            products = db.get_all_products()
            if len(products) == 0:
                raise HTTPException(status_code=400, detail="No products found in database")

            # Train model
            model.train(products)
            model.save_model(SS_INDEX_PATH, SS_EMBEDDINGS_PATH, SS_PRODUCT_IDS_PATH)

            # Update global model
            global ss_model
            ss_model = model

            return TrainingResponse(
                status="success",
                message="Similarity Search model trained successfully",
                model_type="ss",
                stats=model.get_stats()
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/matrix_factor/similar", response_model=RecommendationResponse)
async def get_similar_items(request: SimilarItemsRequest):
    """
    Get similar items using Matrix Factorization.

    Args:
        request: Request with item_id and number of recommendations

    Returns:
        List of similar items with scores
    """
    model = load_mf_model()
    db = initialize_db()

    if model.model is None:
        raise HTTPException(
            status_code=400,
            detail="Matrix Factorization model not trained. Please train the model first."
        )

    try:
        recommendations = model.get_similar_items(request.item_id, n=request.n)
        enriched = enrich_with_product_info(recommendations, db)

        return RecommendationResponse(recommendations=enriched)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


@app.post("/matrix_factor/user", response_model=RecommendationResponse)
async def get_user_recommendations(request: UserRecommendationRequest):
    """
    Get personalized recommendations for a user using Matrix Factorization.

    Args:
        request: Request with user_id and parameters

    Returns:
        List of recommended items with scores
    """
    model = load_mf_model()
    db = initialize_db()

    if model.model is None:
        raise HTTPException(
            status_code=400,
            detail="Matrix Factorization model not trained. Please train the model first."
        )

    try:
        recommendations = model.recommend_for_user(
            request.user_id,
            n=request.n,
            filter_already_liked=request.filter_already_liked
        )
        enriched = enrich_with_product_info(recommendations, db)

        return RecommendationResponse(recommendations=enriched)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


@app.get("/products/all", response_model=SearchResponse)
async def get_all_products():
    """
    Get all products from the catalog (optimized for landing page).
    Returns products without semantic search ranking for better performance.

    Returns:
        List of all products with their details
    """
    db = initialize_db()

    try:
        # Get all products directly from database (much faster than similarity search)
        products = db.get_all_products()

        # Convert to SearchResult format
        results = []
        for product in products:
            image_url = None
            images = product.get('images', [])
            if images and len(images) > 0:
                image_url = images[0].get('large') or images[0].get('thumb')

            results.append(SearchResult(
                product_id=product['parent_asin'],
                title=product['title'],
                score=0.0,  # No relevance score for all products
                image_url=image_url,
                rating=product.get('average_rating'),
                price=product.get('price'),
                description=product.get('description')
            ))

        return SearchResponse(results=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get products: {str(e)}")


@app.get("/products/{product_id}", response_model=SearchResult)
async def get_product_by_id(product_id: str):
    """
    Get a single product by ASIN (optimized for product detail page).

    Args:
        product_id: Product ASIN

    Returns:
        Product details
    """
    db = initialize_db()

    try:
        product = db.get_product_by_asin(product_id)

        if not product:
            raise HTTPException(status_code=404, detail=f"Product {product_id} not found")

        image_url = None
        images = product.get('images', [])
        if images and len(images) > 0:
            image_url = images[0].get('large') or images[0].get('thumb')

        return SearchResult(
            product_id=product['parent_asin'],
            title=product['title'],
            score=0.0,
            image_url=image_url,
            rating=product.get('average_rating'),
            price=product.get('price'),
            description=product.get('description')
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get product: {str(e)}")


@app.post("/search", response_model=SearchResponse)
async def search_products(request: SearchRequest):
    """
    Search for products using semantic similarity.

    Args:
        request: Request with search query and top_k

    Returns:
        List of search results with scores
    """
    model = load_ss_model()
    db = initialize_db()

    if model.index is None:
        raise HTTPException(
            status_code=400,
            detail="Similarity Search model not trained. Please train the model first."
        )

    try:
        # Log the request for debugging
        import traceback
        print(f"Search request - query: '{request.query}', top_k: {request.top_k}")

        # Validate and limit top_k for safety
        top_k = request.top_k
        if top_k is not None and (top_k < 1 or top_k > 5000):
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 5000")

        # If top_k is None, limit to 1000 for reasonable performance
        # For getting all products, use /products/all endpoint instead
        if top_k is None:
            top_k = 1000
            print(f"top_k is None, limiting to {top_k} results")

        results = model.search(request.query, top_k=top_k)
        print(f"Search returned {len(results)} results")

        enriched = enrich_search_results(results, db)
        print(f"Enrichment completed, returning {len(enriched)} results")

        return SearchResponse(results=enriched)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Search error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# ============================================================
# ADMIN ENDPOINTS FOR DATA LIFECYCLE TESTING
# ============================================================

@app.post("/admin/reviews/insert", response_model=AdminOperationResponse)
async def insert_reviews(request: InsertReviewsRequest):
    """
    Insert reviews from JSON data.

    **Use Case**: Simulate new users rating products.

    **Request Body**:
    ```json
    {
        "reviews": [
            {
                "user_id": "USER123",
                "parent_asin": "B08C9LPCQV",
                "rating": 5.0,
                "timestamp": 1609459200000,
                "verified_purchase": 1,
                "helpful_vote": 0
            }
        ]
    }
    ```

    **Returns**: Operation status and database statistics.
    """
    db = initialize_db()

    try:
        rows_affected = db.insert_ratings(request.reviews)
        db_stats = db.get_stats()

        return AdminOperationResponse(
            status="success",
            message=f"Successfully inserted {rows_affected} reviews",
            rows_affected=rows_affected,
            db_stats=db_stats
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to insert reviews: {str(e)}")


@app.post("/admin/reviews/insert-csv", response_model=AdminOperationResponse)
async def insert_reviews_from_csv(file: UploadFile = File(...)):
    """
    Insert reviews from CSV file upload.

    **Use Case**: Bulk insert future reviews from CSV.

    **CSV Format**:
    - Headers: user_id, parent_asin, rating, timestamp, verified_purchase, helpful_vote
    - Example: `USER123,B08C9LPCQV,5.0,1609459200000,1,0`

    **Returns**: Operation status and database statistics.
    """
    db = initialize_db()

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Load from CSV
        rows_affected = db.load_reviews_from_csv(tmp_path)

        # Clean up temp file
        os.unlink(tmp_path)

        db_stats = db.get_stats()

        return AdminOperationResponse(
            status="success",
            message=f"Successfully inserted {rows_affected} reviews from CSV",
            rows_affected=rows_affected,
            db_stats=db_stats
        )
    except Exception as e:
        if 'tmp_path' in locals():
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"Failed to insert reviews from CSV: {str(e)}")


@app.post("/admin/products/insert", response_model=AdminOperationResponse)
async def insert_products(request: InsertProductsRequest):
    """
    Insert products from JSON data.

    **Use Case**: Add new products to the catalog.

    **Request Body**:
    ```json
    {
        "products": [
            {
                "parent_asin": "B08NEWPROD",
                "title": "New Product",
                "main_category": "Appliances",
                "average_rating": 0.0,
                "rating_number": 0,
                "price": 99.99,
                "store": "BrandName",
                "features": ["Feature 1", "Feature 2"],
                "description": ["Description text"],
                "images": [{"large": "http://...", "thumb": "http://..."}],
                "categories": [["Appliances", "Kitchen"]],
                "details": {"Weight": "5 lbs"}
            }
        ]
    }
    ```

    **Returns**: Operation status and database statistics.
    """
    db = initialize_db()

    try:
        rows_affected = db.insert_products(request.products)
        db_stats = db.get_stats()

        return AdminOperationResponse(
            status="success",
            message=f"Successfully inserted {rows_affected} products",
            rows_affected=rows_affected,
            db_stats=db_stats
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to insert products: {str(e)}")


@app.post("/admin/products/insert-csv", response_model=AdminOperationResponse)
async def insert_products_from_csv(file: UploadFile = File(...)):
    """
    Insert products from CSV file upload.

    **Use Case**: Bulk insert future products from CSV.

    **CSV Format**:
    - Headers: parent_asin, title, main_category, average_rating, rating_number, price, store, features, description, images, categories, details
    - JSON fields (features, description, images, categories, details) should be JSON-encoded strings

    **Returns**: Operation status and database statistics.
    """
    db = initialize_db()

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Load from CSV
        rows_affected = db.load_products_from_csv(tmp_path)

        # Clean up temp file
        os.unlink(tmp_path)

        db_stats = db.get_stats()

        return AdminOperationResponse(
            status="success",
            message=f"Successfully inserted {rows_affected} products from CSV",
            rows_affected=rows_affected,
            db_stats=db_stats
        )
    except Exception as e:
        if 'tmp_path' in locals():
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"Failed to insert products from CSV: {str(e)}")


@app.delete("/admin/reviews/first/{n}", response_model=AdminOperationResponse)
async def delete_first_n_reviews(n: int):
    """
    Delete the first N reviews (ordered by ID).

    **Use Case**: Remove oldest reviews for testing.

    **Path Parameter**:
    - `n`: Number of reviews to delete

    **Example**: `DELETE /admin/reviews/first/100`

    **Returns**: Operation status and database statistics.
    """
    db = initialize_db()

    if n <= 0:
        raise HTTPException(status_code=400, detail="N must be positive")

    try:
        rows_affected = db.delete_reviews_first_n(n)
        db_stats = db.get_stats()

        return AdminOperationResponse(
            status="success",
            message=f"Successfully deleted first {rows_affected} reviews",
            rows_affected=rows_affected,
            db_stats=db_stats
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete reviews: {str(e)}")


@app.delete("/admin/reviews/last/{n}", response_model=AdminOperationResponse)
async def delete_last_n_reviews(n: int):
    """
    Delete the last N reviews (ordered by ID).

    **Use Case**: Remove most recent reviews for testing.

    **Path Parameter**:
    - `n`: Number of reviews to delete

    **Example**: `DELETE /admin/reviews/last/100`

    **Returns**: Operation status and database statistics.
    """
    db = initialize_db()

    if n <= 0:
        raise HTTPException(status_code=400, detail="N must be positive")

    try:
        rows_affected = db.delete_reviews_last_n(n)
        db_stats = db.get_stats()

        return AdminOperationResponse(
            status="success",
            message=f"Successfully deleted last {rows_affected} reviews",
            rows_affected=rows_affected,
            db_stats=db_stats
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete reviews: {str(e)}")


@app.delete("/admin/reviews/by-ids", response_model=AdminOperationResponse)
async def delete_reviews_by_ids(request: DeleteByIdsRequest):
    """
    Delete reviews by specific IDs.

    **Use Case**: Remove specific reviews by ID.

    **Request Body**:
    ```json
    {
        "ids": [1, 2, 3, 100, 200]
    }
    ```

    **Returns**: Operation status and database statistics.
    """
    db = initialize_db()

    try:
        rows_affected = db.delete_reviews_by_ids(request.ids)
        db_stats = db.get_stats()

        return AdminOperationResponse(
            status="success",
            message=f"Successfully deleted {rows_affected} reviews",
            rows_affected=rows_affected,
            db_stats=db_stats
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete reviews: {str(e)}")


@app.delete("/admin/reviews/by-users", response_model=AdminOperationResponse)
async def delete_reviews_by_user_ids(request: DeleteByUserIdsRequest):
    """
    Delete all reviews by specific user IDs.

    **Use Case**: Simulate user account closure - removes all reviews for specified users.

    **Request Body**:
    ```json
    {
        "user_ids": ["USER123", "USER456", "USER789"]
    }
    ```

    **Returns**: Operation status and database statistics.
    """
    db = initialize_db()

    try:
        rows_affected = db.delete_reviews_by_user_ids(request.user_ids)
        db_stats = db.get_stats()

        return AdminOperationResponse(
            status="success",
            message=f"Successfully deleted {rows_affected} reviews for {len(request.user_ids)} users",
            rows_affected=rows_affected,
            db_stats=db_stats
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete reviews: {str(e)}")


@app.delete("/admin/products/first/{n}", response_model=AdminOperationResponse)
async def delete_first_n_products(n: int):
    """
    Delete the first N products (ordered by ASIN).

    **Use Case**: Remove products for testing. CASCADE will also delete associated reviews.

    **Path Parameter**:
    - `n`: Number of products to delete

    **Example**: `DELETE /admin/products/first/10`

    **Returns**: Operation status and database statistics.
    """
    db = initialize_db()

    if n <= 0:
        raise HTTPException(status_code=400, detail="N must be positive")

    try:
        rows_affected = db.delete_products_first_n(n)
        db_stats = db.get_stats()

        return AdminOperationResponse(
            status="success",
            message=f"Successfully deleted first {rows_affected} products (CASCADE also deleted associated reviews)",
            rows_affected=rows_affected,
            db_stats=db_stats
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete products: {str(e)}")


@app.delete("/admin/products/last/{n}", response_model=AdminOperationResponse)
async def delete_last_n_products(n: int):
    """
    Delete the last N products (ordered by ASIN).

    **Use Case**: Remove products for testing. CASCADE will also delete associated reviews.

    **Path Parameter**:
    - `n`: Number of products to delete

    **Example**: `DELETE /admin/products/last/10`

    **Returns**: Operation status and database statistics.
    """
    db = initialize_db()

    if n <= 0:
        raise HTTPException(status_code=400, detail="N must be positive")

    try:
        rows_affected = db.delete_products_last_n(n)
        db_stats = db.get_stats()

        return AdminOperationResponse(
            status="success",
            message=f"Successfully deleted last {rows_affected} products (CASCADE also deleted associated reviews)",
            rows_affected=rows_affected,
            db_stats=db_stats
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete products: {str(e)}")


@app.delete("/admin/products/by-asins", response_model=AdminOperationResponse)
async def delete_products_by_asins(request: DeleteByAsinsRequest):
    """
    Delete products by specific ASINs.

    **Use Case**: Simulate product delisting. CASCADE will also delete associated reviews.

    **Request Body**:
    ```json
    {
        "asins": ["B08C9LPCQV", "B08ANOTHER", "B08PRODUCT"]
    }
    ```

    **Returns**: Operation status and database statistics.
    """
    db = initialize_db()

    try:
        rows_affected = db.delete_products_by_asins(request.asins)
        db_stats = db.get_stats()

        return AdminOperationResponse(
            status="success",
            message=f"Successfully deleted {rows_affected} products (CASCADE also deleted associated reviews)",
            rows_affected=rows_affected,
            db_stats=db_stats
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete products: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    from ..config import API_HOST, API_PORT, API_DEBUG

    uvicorn.run(app, host=API_HOST, port=API_PORT, reload=API_DEBUG)
