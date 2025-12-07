"""
Example script demonstrating programmatic usage of the recommender system.
This shows how to use the models directly without the API.
"""
from src.config import (
    DB_PATH, MF_CONFIG, MF_MODEL_PATH, MF_MAPPINGS_PATH, MF_SPARSE_MATRIX_PATH,
    SS_MODEL_NAME, SS_INDEX_PATH, SS_EMBEDDINGS_PATH, SS_PRODUCT_IDS_PATH
)
from src.etl.database import RecommenderDB
from src.models.matrix_factorization import MatrixFactorizationModel
from src.models.similarity_search import SimilaritySearchModel


def example_matrix_factorization():
    """Example: Using Matrix Factorization for recommendations."""
    print("\n" + "=" * 60)
    print("Matrix Factorization Example")
    print("=" * 60)

    # Initialize and load model
    mf_model = MatrixFactorizationModel(MF_CONFIG)
    mf_model.load_model(MF_MODEL_PATH, MF_MAPPINGS_PATH, MF_SPARSE_MATRIX_PATH)

    print(f"Model loaded: {mf_model.get_stats()}")

    # Example 1: Get similar items
    print("\n--- Similar Items for B08C9LPCQV ---")
    try:
        similar_items = mf_model.get_similar_items("B08C9LPCQV", n=5)
        for item in similar_items:
            print(f"  {item['item_id']}: {item['score']:.4f}")
    except Exception as e:
        print(f"Error: {e}")

    # Example 2: User recommendations
    print("\n--- Recommendations for User AFQLNQNQYFWQZPJQZS6V3NZU4QBQ ---")
    try:
        user_recs = mf_model.recommend_for_user(
            "AFQLNQNQYFWQZPJQZS6V3NZU4QBQ",
            n=5,
            filter_already_liked=True
        )
        for item in user_recs:
            print(f"  {item['item_id']}: {item['score']:.4f}")
    except Exception as e:
        print(f"Error: {e}")


def example_similarity_search():
    """Example: Using Similarity Search."""
    print("\n" + "=" * 60)
    print("Similarity Search Example")
    print("=" * 60)

    # Initialize and load model
    ss_model = SimilaritySearchModel(model_name=SS_MODEL_NAME)
    ss_model.load_model(SS_INDEX_PATH, SS_EMBEDDINGS_PATH, SS_PRODUCT_IDS_PATH)

    print(f"Model loaded: {ss_model.get_stats()}")

    # Example searches
    queries = [
        "washing machine",
        "coffee maker",
        "refrigerator parts"
    ]

    for query in queries:
        print(f"\n--- Search: '{query}' ---")
        results = ss_model.search(query, top_k=3)
        for result in results:
            print(f"  {result['score']:.4f} - {result['title'][:60]}...")


def example_with_database():
    """Example: Combining models with database for rich results."""
    print("\n" + "=" * 60)
    print("Database Integration Example")
    print("=" * 60)

    # Initialize database
    db = RecommenderDB(str(DB_PATH))

    # Get database stats
    stats = db.get_stats()
    print(f"\nDatabase Stats:")
    print(f"  Total Ratings: {stats['n_ratings']:,}")
    print(f"  Unique Users: {stats['n_users']:,}")
    print(f"  Unique Products: {stats['n_products']:,}")
    print(f"  Average Rating: {stats['avg_rating']}")

    # Load matrix factorization model
    mf_model = MatrixFactorizationModel(MF_CONFIG)
    mf_model.load_model(MF_MODEL_PATH, MF_MAPPINGS_PATH, MF_SPARSE_MATRIX_PATH)

    # Get similar items and enrich with product info
    print("\n--- Similar Items with Product Details ---")
    item_id = "B08C9LPCQV"

    # Get product info
    product = db.get_product_by_asin(item_id)
    if product:
        print(f"\nQuery Product: {product['title']}")
        print(f"Rating: {product.get('average_rating', 'N/A')}")
        print(f"Store: {product.get('store', 'N/A')}")

    # Get similar items
    similar = mf_model.get_similar_items(item_id, n=5)

    print("\nSimilar Products:")
    for i, item in enumerate(similar, 1):
        prod = db.get_product_by_asin(item['item_id'])
        if prod:
            print(f"\n{i}. {prod['title'][:60]}...")
            print(f"   Score: {item['score']:.4f}")
            print(f"   Rating: {prod.get('average_rating', 'N/A')}")
            print(f"   Store: {prod.get('store', 'N/A')}")


def example_user_journey():
    """Example: Simulating a user journey."""
    print("\n" + "=" * 60)
    print("User Journey Simulation")
    print("=" * 60)

    db = RecommenderDB(str(DB_PATH))
    mf_model = MatrixFactorizationModel(MF_CONFIG)
    mf_model.load_model(MF_MODEL_PATH, MF_MAPPINGS_PATH, MF_SPARSE_MATRIX_PATH)
    ss_model = SimilaritySearchModel(model_name=SS_MODEL_NAME)
    ss_model.load_model(SS_INDEX_PATH, SS_EMBEDDINGS_PATH, SS_PRODUCT_IDS_PATH)

    print("\nScenario: User searches for 'dryer parts'")

    # Step 1: Search
    search_results = ss_model.search("dryer parts", top_k=3)
    print("\nSearch Results:")
    for i, result in enumerate(search_results, 1):
        print(f"{i}. {result['title'][:60]}... (Score: {result['score']:.4f})")

    # Step 2: User clicks on first result
    clicked_item = search_results[0]['product_id']
    print(f"\nUser clicks on: {clicked_item}")

    # Step 3: Show similar items (customers also bought)
    similar_items = mf_model.get_similar_items(clicked_item, n=3)
    print("\nCustomers who bought this also bought:")
    for i, item in enumerate(similar_items, 1):
        prod = db.get_product_by_asin(item['item_id'])
        if prod:
            print(f"{i}. {prod['title'][:60]}... (Score: {item['score']:.4f})")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Recommender System - Example Usage")
    print("=" * 60)

    try:
        # Check if models exist
        if not MF_MODEL_PATH.exists():
            print("\nERROR: Matrix Factorization model not found.")
            print("Please run: python train_models.py")
            return

        if not SS_INDEX_PATH.exists():
            print("\nERROR: Similarity Search model not found.")
            print("Please run: python train_models.py")
            return

        # Run examples
        example_matrix_factorization()
        example_similarity_search()
        example_with_database()
        example_user_journey()

        print("\n" + "=" * 60)
        print("Examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
