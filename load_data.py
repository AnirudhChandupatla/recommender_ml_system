"""
Script to load data from JSONL files into SQLite database.
"""
import argparse
from pathlib import Path
from src.etl.data_loader import populate_database
from src.config import REVIEWS_JSONL, PRODUCTS_JSONL, PROCESSED_DATA_DIR


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Load data from JSONL files into SQLite database"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to SQLite database file (default: data/processed/appliances_data.db)"
    )
    parser.add_argument(
        "--reviews",
        type=str,
        default=None,
        help=f"Path to reviews JSONL file (default: {REVIEWS_JSONL})"
    )
    parser.add_argument(
        "--products",
        type=str,
        default=None,
        help=f"Path to products JSONL file (default: {PRODUCTS_JSONL})"
    )
    parser.add_argument(
        "--split-future-data",
        action="store_true",
        help="Split data into current (DB) and future (testing) sets"
    )
    parser.add_argument(
        "--n-future-reviews",
        type=int,
        default=10000,
        help="Number of reviews to set aside for future testing (default: 10,000)"
    )
    parser.add_argument(
        "--n-future-products",
        type=int,
        default=1000,
        help="Number of products to set aside for future testing (default: 1,000)"
    )

    args = parser.parse_args()

    # Determine database path
    if args.db_path:
        db_path = Path(args.db_path)
    else:
        # Use default name "appliances_data.db" in processed directory
        db_path = PROCESSED_DATA_DIR / "appliances_data.db"

    # Determine data file paths
    reviews_path = Path(args.reviews) if args.reviews else REVIEWS_JSONL
    products_path = Path(args.products) if args.products else PRODUCTS_JSONL

    print("=" * 60)
    print("Recommender System - Data Loading Script")
    print("=" * 60)
    print()

    # Check if data files exist
    if not reviews_path.exists():
        print(f"ERROR: Reviews file not found at: {reviews_path}")
        print("Please place the Appliances.jsonl file in data/raw/Appliances.jsonl/")
        print("Or specify the path with --reviews option")
        return

    if not products_path.exists():
        print(f"ERROR: Products file not found at: {products_path}")
        print("Please place the meta_Appliances.jsonl file in data/raw/meta_Appliances.jsonl/")
        print("Or specify the path with --products option")
        return

    print(f"Reviews file: {reviews_path}")
    print(f"Products file: {products_path}")
    print(f"Database will be created at: {db_path}")

    # Configure future data paths
    future_reviews_csv = None
    future_products_csv = None

    if args.split_future_data:
        future_reviews_csv = PROCESSED_DATA_DIR / "future_reviews.csv"
        future_products_csv = PROCESSED_DATA_DIR / "future_products.csv"
        print(f"\n📦 Future Data Splitting ENABLED:")
        print(f"   Future reviews: {args.n_future_reviews:,} records → {future_reviews_csv}")
        print(f"   Future products: {args.n_future_products:,} records → {future_products_csv}")

    print()

    # Ask for confirmation if database already exists
    if db_path.exists():
        response = input("Database already exists. Overwrite? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Aborted.")
            return
        print()

    # Load data
    try:
        populate_database(
            reviews_path,
            products_path,
            db_path,
            split_future_data=args.split_future_data,
            future_reviews_csv=future_reviews_csv,
            future_products_csv=future_products_csv,
            n_future_reviews=args.n_future_reviews,
            n_future_products=args.n_future_products
        )
        print("\n" + "=" * 60)
        print("Data loading completed successfully!")
        print("=" * 60)
        print(f"\nDatabase location: {db_path}")

        if args.split_future_data:
            print(f"\n📦 Future Data Files Created:")
            print(f"   Reviews CSV: {future_reviews_csv}")
            print(f"   Products CSV: {future_products_csv}")
            print(f"\nThese files can be used with the /admin API endpoints for lifecycle testing.")
    except Exception as e:
        print(f"\nERROR: Data loading failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
