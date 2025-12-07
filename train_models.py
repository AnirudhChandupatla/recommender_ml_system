"""
Script to train both models locally (without API).
"""
from src.config import (
    DB_PATH, MF_CONFIG, MF_MODEL_PATH, MF_MAPPINGS_PATH, MF_SPARSE_MATRIX_PATH,
    SS_MODEL_NAME, SS_INDEX_PATH, SS_EMBEDDINGS_PATH, SS_PRODUCT_IDS_PATH, SS_BATCH_SIZE
)
from src.etl.database import RecommenderDB
from src.models.matrix_factorization import MatrixFactorizationModel
from src.models.similarity_search import SimilaritySearchModel


def train_matrix_factorization():
    """Train the Matrix Factorization model."""
    print("\n" + "=" * 60)
    print("Training Matrix Factorization Model")
    print("=" * 60)

    # Initialize database
    db = RecommenderDB(str(DB_PATH))

    # Get ratings
    print("\nLoading ratings from database...")
    ratings = db.get_all_ratings()
    print(f"Loaded {len(ratings)} ratings")

    # Train model
    model = MatrixFactorizationModel(MF_CONFIG)
    model.train(ratings)

    # Save model
    model.save_model(MF_MODEL_PATH, MF_MAPPINGS_PATH, MF_SPARSE_MATRIX_PATH)

    print("\nMatrix Factorization model training completed!")
    print(f"Model statistics: {model.get_stats()}")


def train_similarity_search():
    """Train the Similarity Search model."""
    print("\n" + "=" * 60)
    print("Training Similarity Search Model")
    print("=" * 60)

    # Initialize database
    db = RecommenderDB(str(DB_PATH))

    # Get products
    print("\nLoading products from database...")
    products = db.get_all_products()
    print(f"Loaded {len(products)} products")

    # Train model
    model = SimilaritySearchModel(model_name=SS_MODEL_NAME, batch_size=SS_BATCH_SIZE)
    model.train(products)

    # Save model
    model.save_model(SS_INDEX_PATH, SS_EMBEDDINGS_PATH, SS_PRODUCT_IDS_PATH)

    print("\nSimilarity Search model training completed!")
    print(f"Model statistics: {model.get_stats()}")


def main():
    print("=" * 60)
    print("Recommender System - Model Training Script")
    print("=" * 60)

    # Check if database exists
    if not DB_PATH.exists():
        print(f"\nERROR: Database not found at {DB_PATH}")
        print("Please run load_data.py first to create the database.")
        return

    print("\nWhich model would you like to train?")
    print("1. Matrix Factorization (MF)")
    print("2. Similarity Search (SS)")
    print("3. Both")

    choice = input("\nEnter choice (1/2/3): ").strip()

    try:
        if choice == '1':
            train_matrix_factorization()
        elif choice == '2':
            train_similarity_search()
        elif choice == '3':
            train_matrix_factorization()
            train_similarity_search()
        else:
            print("Invalid choice. Aborted.")
            return

        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
        print("\nYou can now start the API server with: python run_server.py")

    except Exception as e:
        print(f"\nERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
