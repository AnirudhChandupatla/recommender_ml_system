"""
Data loading utilities to populate the database from JSONL files.
Optimized for maximum performance with parallel parsing and bulk inserts.
"""
import json
import sqlite3
import time
import multiprocessing
import random
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import timedelta
from collections import defaultdict
from joblib import Parallel, delayed

from .database import RecommenderDB


def parse_batch(lines: List[str]) -> List[Dict[str, Any]]:
    """
    Parse a batch of JSONL lines in parallel.

    Args:
        lines: List of JSONL lines to parse

    Returns:
        List of parsed JSON objects
    """
    batch = []
    for line in lines:
        try:
            # Fixed: Single JSON parse instead of triple parsing
            data = json.loads(line.strip())
            batch.append(data)
        except json.JSONDecodeError:
            continue
    return batch


def read_jsonl_parallel(file_path: Path, batch_size: int = 50000, n_jobs: int = -1) -> List[List[Dict[str, Any]]]:
    """
    Read JSONL file with parallel parsing for maximum speed.

    Args:
        file_path: Path to JSONL file
        batch_size: Number of records per batch for insertion
        n_jobs: Number of parallel jobs (-1 = all CPUs)

    Returns:
        List of batches, where each batch is a list of parsed dictionaries
    """
    print(f"\n📖 Reading {file_path.name}...")

    # Step 1: Read all lines (fast - just file I/O)
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)
    print(f"   Found {total_lines:,} lines")

    # Step 2: Split into chunks for parallel processing
    n_cores = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    chunk_size = max(1000, len(lines) // (n_cores * 4))  # Create 4x chunks for load balancing
    chunks = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]

    # Step 3: Parse in parallel
    print(f"   Parsing using {n_cores} cores...")
    parse_start = time.time()

    parsed_batches = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(parse_batch)(chunk) for chunk in chunks
    )

    parse_time = time.time() - parse_start
    print(f"   ✓ Parsing completed in {timedelta(seconds=int(parse_time))}")

    # Step 4: Flatten and re-batch to target batch_size
    all_data = [item for batch in parsed_batches for item in batch]
    batches = [all_data[i:i+batch_size] for i in range(0, len(all_data), batch_size)]

    print(f"   ✓ Created {len(batches)} batches of ~{batch_size:,} records each")

    return batches


def split_future_data_function(
    reviews: List[Dict],
    products: List[Dict],
    n_future_reviews: int = 10000,
    n_future_products: int = 1000
) -> Tuple[List, List, List, List]:
    """
    Split data into current (for DB) and future (for testing).

    Strategy:
    - Select products with LEAST reviews (easier to manage)
    - Ensure FK integrity: future reviews only reference valid products

    Args:
        reviews: List of all review records
        products: List of all product records
        n_future_reviews: Number of reviews to set aside (default: 10,000)
        n_future_products: Number of products to set aside (default: 1,000)

    Returns:
        Tuple of (db_reviews, db_products, future_reviews, future_products)
    """
    print(f"\n📦 Splitting future data...")
    print(f"   Target: {n_future_products:,} products, {n_future_reviews:,} reviews")

    # Step 1: Count reviews per product
    review_counts = defaultdict(int)
    for review in reviews:
        review_counts[review.get('parent_asin')] += 1

    # Step 2: Sort products by review count (ascending - least reviews first)
    products_sorted = sorted(
        products,
        key=lambda p: review_counts.get(p.get('parent_asin'), 0)
    )

    # Step 3: Split products - take bottom 1K (least reviews)
    future_products = products_sorted[:n_future_products]
    db_products = products_sorted[n_future_products:]

    future_product_asins = {p.get('parent_asin') for p in future_products}
    db_product_asins = {p.get('parent_asin') for p in db_products}

    print(f"   Selected {len(future_products):,} products with least reviews")

    # Step 4: Split reviews maintaining FK integrity
    future_reviews = []
    db_reviews = []
    unknown_asin_reviews = []

    for review in reviews:
        asin = review.get('parent_asin')

        # Reviews for future products go to future set (up to limit)
        if asin in future_product_asins:
            if len(future_reviews) < n_future_reviews:
                future_reviews.append(review)
            else:
                db_reviews.append(review)  # Overflow goes to DB

        # Reviews for DB products go to DB
        elif asin in db_product_asins:
            db_reviews.append(review)

        # Reviews for unknown products (shouldn't happen with valid data)
        else:
            unknown_asin_reviews.append(review)
            print(f"   WARNING: Review references unknown product {asin}")

    # Step 5: If we don't have enough future reviews, sample from db_reviews
    if len(future_reviews) < n_future_reviews:
        shortage = n_future_reviews - len(future_reviews)
        print(f"   Need {shortage:,} more reviews for future set")

        # Get reviews that reference db_products (maintain FK integrity)
        candidates = [r for r in db_reviews if r.get('parent_asin') in db_product_asins]

        if len(candidates) >= shortage:
            additional = random.sample(candidates, shortage)
            future_reviews.extend(additional)
            # Remove selected reviews from db_reviews
            additional_ids = {id(r) for r in additional}
            db_reviews = [r for r in db_reviews if id(r) not in additional_ids]
            print(f"   Added {len(additional):,} reviews from DB products to future set")
        else:
            print(f"   WARNING: Only {len(candidates):,} candidates available, need {shortage:,}")
            future_reviews.extend(candidates)
            db_reviews = [r for r in db_reviews if r not in candidates]

    print(f"\n✅ Data Split Summary:")
    print(f"   DB Products:      {len(db_products):,}")
    print(f"   DB Reviews:       {len(db_reviews):,}")
    print(f"   Future Products:  {len(future_products):,}")
    print(f"   Future Reviews:   {len(future_reviews):,}")

    # Verify FK integrity
    verify_fk_integrity(db_reviews, db_products, future_reviews, future_products)

    return db_reviews, db_products, future_reviews, future_products


def verify_fk_integrity(
    db_reviews: List[Dict],
    db_products: List[Dict],
    future_reviews: List[Dict],
    future_products: List[Dict]
):
    """
    Verify that all reviews reference valid products.

    FK Integrity Rules:
    - All DB reviews must reference DB products
    - Future reviews can reference either future products OR DB products

    Args:
        db_reviews: Reviews going to database
        db_products: Products going to database
        future_reviews: Reviews set aside for future
        future_products: Products set aside for future

    Raises:
        AssertionError: If FK integrity is violated
    """
    db_asins = {p.get('parent_asin') for p in db_products}
    future_asins = {p.get('parent_asin') for p in future_products}
    all_asins = db_asins | future_asins

    # Check DB reviews - must only reference DB products
    print(f"\n🔍 Verifying FK integrity...")
    db_violations = []
    for review in db_reviews:
        asin = review.get('parent_asin')
        if asin not in db_asins:
            db_violations.append(asin)

    if db_violations:
        print(f"   ❌ ERROR: {len(db_violations)} DB reviews reference non-DB products")
        print(f"   First few violations: {list(set(db_violations))[:5]}")
        raise AssertionError(f"DB review FK violation: {len(db_violations)} reviews reference non-DB products")

    # Check future reviews - can reference any valid product
    future_violations = []
    for review in future_reviews:
        asin = review.get('parent_asin')
        if asin not in all_asins:
            future_violations.append(asin)

    if future_violations:
        print(f"   ❌ ERROR: {len(future_violations)} future reviews reference non-existent products")
        print(f"   First few violations: {list(set(future_violations))[:5]}")
        raise AssertionError(f"Future review FK violation: {len(future_violations)} reviews reference non-existent products")

    # Summary
    future_refs_to_db = sum(1 for r in future_reviews if r.get('parent_asin') in db_asins)
    future_refs_to_future = sum(1 for r in future_reviews if r.get('parent_asin') in future_asins)

    print(f"   ✅ FK integrity verified!")
    print(f"   DB reviews → DB products: {len(db_reviews):,}")
    print(f"   Future reviews → DB products: {future_refs_to_db:,}")
    print(f"   Future reviews → Future products: {future_refs_to_future:,}")


def save_future_data_to_csv(
    reviews: List[Dict],
    products: List[Dict],
    reviews_path: Path,
    products_path: Path
):
    """
    Save future data to CSV files.

    JSON fields (features, description, etc.) are serialized to JSON strings.

    Args:
        reviews: List of review records
        products: List of product records
        reviews_path: Path to save reviews CSV
        products_path: Path to save products CSV
    """
    print(f"\n💾 Saving future data to CSV...")

    # Save products
    if products:
        product_fields = [
            'parent_asin', 'title', 'main_category', 'average_rating',
            'rating_number', 'price', 'store', 'features', 'description',
            'images', 'categories', 'details'
        ]

        with open(products_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=product_fields)
            writer.writeheader()

            for product in products:
                # Convert lists/dicts to JSON strings
                row = {}
                for field in product_fields:
                    value = product.get(field)
                    if isinstance(value, (list, dict)):
                        row[field] = json.dumps(value)
                    else:
                        row[field] = value
                writer.writerow(row)

        print(f"   ✅ Saved {len(products):,} products to {products_path.name}")

    # Save reviews
    if reviews:
        review_fields = [
            'user_id', 'parent_asin', 'rating', 'timestamp',
            'verified_purchase', 'helpful_vote', 'text', 'title', 'images', 'asin'
        ]

        with open(reviews_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=review_fields)
            writer.writeheader()
            writer.writerows(reviews)

        print(f"   ✅ Saved {len(reviews):,} reviews to {reviews_path.name}")


def load_reviews_to_db_fast(reviews_path: Path, conn: sqlite3.Connection, batch_size: int = 50000) -> int:
    """
    Load reviews with parallel parsing and detailed progress logging.

    Args:
        reviews_path: Path to reviews JSONL file
        conn: SQLite connection (single transaction)
        batch_size: Number of records per batch

    Returns:
        Total number of records loaded
    """
    print("\n" + "="*70)
    print("📥 LOADING REVIEWS")
    print("="*70)

    # Parse in parallel
    batches = read_jsonl_parallel(reviews_path, batch_size)

    # Insert batches sequentially (SQLite limitation)
    print(f"\n💾 Inserting {sum(len(b) for b in batches):,} reviews into database...")

    cursor = conn.cursor()
    total_loaded = 0
    insert_start = time.time()

    for batch_num, batch_data in enumerate(batches, 1):
        # Transform to rating records
        rating_records = []
        for review in batch_data:
            rating_records.append({
                'user_id': review.get('user_id'),
                'parent_asin': review.get('parent_asin'),
                'rating': float(review.get('rating', 0)),
                'timestamp': review.get('timestamp'),
                'verified_purchase': 1 if review.get('verified_purchase') else 0,
                'helpful_vote': review.get('helpful_vote', 0)
            })

        # Insert batch
        cursor.executemany("""
            INSERT OR REPLACE INTO user_item_rating
            (user_id, parent_asin, rating, timestamp, verified_purchase, helpful_vote)
            VALUES (:user_id, :parent_asin, :rating, :timestamp, :verified_purchase, :helpful_vote)
        """, rating_records)

        total_loaded += len(rating_records)

        # Progress logging every batch
        elapsed = time.time() - insert_start
        throughput = total_loaded / elapsed if elapsed > 0 else 0
        print(f"   Batch {batch_num:3d}/{len(batches)}: "
              f"{len(rating_records):6,} records | "
              f"Total: {total_loaded:10,} | "
              f"{total_loaded/1e6:5.2f}M | "
              f"{throughput:8,.0f} rec/sec")

    insert_time = time.time() - insert_start

    print(f"\n✅ Loaded {total_loaded:,} ratings")
    print(f"   Insert time: {timedelta(seconds=int(insert_time))}")
    print(f"   Throughput: {total_loaded / insert_time:,.0f} records/sec")

    return total_loaded


def load_products_to_db_fast(products_path: Path, conn: sqlite3.Connection, batch_size: int = 50000) -> int:
    """
    Load products with parallel parsing and detailed progress logging.

    Args:
        products_path: Path to products JSONL file
        conn: SQLite connection (single transaction)
        batch_size: Number of records per batch

    Returns:
        Total number of records loaded
    """
    print("\n" + "="*70)
    print("📥 LOADING PRODUCTS")
    print("="*70)

    # Parse in parallel
    batches = read_jsonl_parallel(products_path, batch_size)

    # Insert batches sequentially
    print(f"\n💾 Inserting {sum(len(b) for b in batches):,} products into database...")

    cursor = conn.cursor()
    total_loaded = 0
    insert_start = time.time()

    for batch_num, batch_data in enumerate(batches, 1):
        # Transform and serialize JSON fields inline
        product_records = []
        for product in batch_data:
            product_records.append({
                'parent_asin': product.get('parent_asin'),
                'title': product.get('title'),
                'main_category': product.get('main_category'),
                'average_rating': product.get('average_rating'),
                'rating_number': product.get('rating_number'),
                'price': product.get('price'),
                'store': product.get('store'),
                'features': json.dumps(product.get('features', [])),
                'description': json.dumps(product.get('description', [])),
                'images': json.dumps(product.get('images', [])),
                'categories': json.dumps(product.get('categories', [])),
                'details': json.dumps(product.get('details', {}))
            })

        # Insert batch
        cursor.executemany("""
            INSERT OR REPLACE INTO product_catalog
            (parent_asin, title, main_category, average_rating, rating_number,
             price, store, features, description, images, categories, details)
            VALUES (:parent_asin, :title, :main_category, :average_rating, :rating_number,
                    :price, :store, :features, :description, :images, :categories, :details)
        """, product_records)

        total_loaded += len(product_records)

        # Progress logging
        elapsed = time.time() - insert_start
        throughput = total_loaded / elapsed if elapsed > 0 else 0
        print(f"   Batch {batch_num:3d}/{len(batches)}: "
              f"{len(product_records):6,} records | "
              f"Total: {total_loaded:10,} | "
              f"{throughput:8,.0f} rec/sec")

    insert_time = time.time() - insert_start

    print(f"\n✅ Loaded {total_loaded:,} products")
    print(f"   Insert time: {timedelta(seconds=int(insert_time))}")
    print(f"   Throughput: {total_loaded / insert_time:,.0f} records/sec")

    return total_loaded


def populate_database(
    reviews_path: Path,
    products_path: Path,
    db_path: Path,
    split_future_data: bool = False,
    future_reviews_csv: Path = None,
    future_products_csv: Path = None,
    n_future_reviews: int = 10000,
    n_future_products: int = 1000
):
    """
    Populate database with optimized single-transaction approach.

    Performance optimizations:
    - Single transaction for all inserts (no intermediate commits)
    - Deferred index creation (indexes created AFTER data loading)
    - SQLite pragmas for bulk insert mode
    - Parallel JSONL parsing
    - Detailed progress logging

    Args:
        reviews_path: Path to reviews JSONL
        products_path: Path to products JSONL
        db_path: Path to SQLite database
        split_future_data: If True, set aside data for future testing (default: False)
        future_reviews_csv: Path to save future reviews CSV (required if split_future_data=True)
        future_products_csv: Path to save future products CSV (required if split_future_data=True)
        n_future_reviews: Number of reviews to set aside (default: 10,000)
        n_future_products: Number of products to set aside (default: 1,000)
    """
    print("\n" + "="*70)
    print("🚀 OPTIMIZED DATA LOADING")
    print("="*70)
    print(f"   Database: {db_path}")
    print(f"   CPU Cores: {multiprocessing.cpu_count()}")
    if split_future_data:
        print(f"   Mode: Split future data ({n_future_products:,} products, {n_future_reviews:,} reviews)")
    else:
        print(f"   Mode: Load all data")
    print("="*70)

    overall_start = time.time()

    # Initialize database (creates tables)
    db = RecommenderDB(str(db_path))

    # Get raw connection for single transaction
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Enable foreign key constraints
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        # ============================================================
        # PHASE 1: OPTIMIZATION SETUP
        # ============================================================
        print("\n⚙️  Optimizing database for bulk insert...")

        # Enable fast insert mode
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA journal_mode = MEMORY")
        conn.execute("PRAGMA cache_size = 100000")
        conn.execute("PRAGMA temp_store = MEMORY")

        print("   ✓ SQLite pragmas configured")
        print("   ✓ Foreign key constraints enabled")

        # Drop indexes (will recreate after data loading)
        print("   ⚙️  Dropping indexes...")
        db.drop_indexes()
        print("   ✓ Indexes dropped")

        # ============================================================
        # PHASE 2: DATA PARSING
        # ============================================================
        print("\n📖 Parsing data files...")

        # Parse all products
        products_batches = read_jsonl_parallel(products_path, batch_size=50000)
        all_products = [item for batch in products_batches for item in batch]
        print(f"   ✓ Loaded {len(all_products):,} products into memory")

        # Parse all reviews
        reviews_batches = read_jsonl_parallel(reviews_path, batch_size=50000)
        all_reviews = [item for batch in reviews_batches for item in batch]
        print(f"   ✓ Loaded {len(all_reviews):,} reviews into memory")

        # ============================================================
        # PHASE 3: DATA SPLITTING (OPTIONAL)
        # ============================================================
        if split_future_data:
            if not future_reviews_csv or not future_products_csv:
                raise ValueError("future_reviews_csv and future_products_csv must be provided when split_future_data=True")

            db_reviews, db_products, future_reviews, future_products = split_future_data_function(
                all_reviews,
                all_products,
                n_future_reviews=n_future_reviews,
                n_future_products=n_future_products
            )

            # Save future data to CSV
            save_future_data_to_csv(
                future_reviews,
                future_products,
                future_reviews_csv,
                future_products_csv
            )
        else:
            # Use all data
            db_reviews = all_reviews
            db_products = all_products

        # ============================================================
        # PHASE 4: DATABASE LOADING (SINGLE TRANSACTION)
        # ============================================================
        print("\n🔄 Beginning single transaction...")
        conn.execute("BEGIN TRANSACTION")

        # IMPORTANT: Load products FIRST (they are referenced by FK)
        print("\n" + "="*70)
        print("📥 LOADING PRODUCTS TO DATABASE")
        print("="*70)
        products_start = time.time()

        # Insert products in batches
        cursor = conn.cursor()
        total_products = 0
        product_batch_size = 50000
        product_batches = [db_products[i:i+product_batch_size] for i in range(0, len(db_products), product_batch_size)]

        for batch_num, batch_data in enumerate(product_batches, 1):
            # Transform and serialize JSON fields inline
            product_records = []
            for product in batch_data:
                product_records.append({
                    'parent_asin': product.get('parent_asin'),
                    'title': product.get('title'),
                    'main_category': product.get('main_category'),
                    'average_rating': product.get('average_rating'),
                    'rating_number': product.get('rating_number'),
                    'price': product.get('price'),
                    'store': product.get('store'),
                    'features': json.dumps(product.get('features', [])),
                    'description': json.dumps(product.get('description', [])),
                    'images': json.dumps(product.get('images', [])),
                    'categories': json.dumps(product.get('categories', [])),
                    'details': json.dumps(product.get('details', {}))
                })

            # Insert batch
            cursor.executemany("""
                INSERT OR REPLACE INTO product_catalog
                (parent_asin, title, main_category, average_rating, rating_number,
                 price, store, features, description, images, categories, details)
                VALUES (:parent_asin, :title, :main_category, :average_rating, :rating_number,
                        :price, :store, :features, :description, :images, :categories, :details)
            """, product_records)

            total_products += len(product_records)
            elapsed = time.time() - products_start
            throughput = total_products / elapsed if elapsed > 0 else 0
            print(f"   Batch {batch_num:3d}/{len(product_batches)}: "
                  f"{len(product_records):6,} records | "
                  f"Total: {total_products:10,} | "
                  f"{throughput:8,.0f} rec/sec")

        n_products = total_products
        products_time = time.time() - products_start
        print(f"\n✅ Loaded {n_products:,} products")
        print(f"   Insert time: {timedelta(seconds=int(products_time))}")

        # Then load reviews (they reference products via FK)
        print("\n" + "="*70)
        print("📥 LOADING REVIEWS TO DATABASE")
        print("="*70)
        reviews_start = time.time()

        # Insert reviews in batches
        total_reviews = 0
        review_batch_size = 50000
        review_batches = [db_reviews[i:i+review_batch_size] for i in range(0, len(db_reviews), review_batch_size)]

        for batch_num, batch_data in enumerate(review_batches, 1):
            # Transform to rating records
            rating_records = []
            for review in batch_data:
                rating_records.append({
                    'user_id': review.get('user_id'),
                    'parent_asin': review.get('parent_asin'),
                    'rating': float(review.get('rating', 0)),
                    'timestamp': review.get('timestamp'),
                    'verified_purchase': 1 if review.get('verified_purchase') else 0,
                    'helpful_vote': review.get('helpful_vote', 0)
                })

            # Insert batch
            cursor.executemany("""
                INSERT OR REPLACE INTO user_item_rating
                (user_id, parent_asin, rating, timestamp, verified_purchase, helpful_vote)
                VALUES (:user_id, :parent_asin, :rating, :timestamp, :verified_purchase, :helpful_vote)
            """, rating_records)

            total_reviews += len(rating_records)
            elapsed = time.time() - reviews_start
            throughput = total_reviews / elapsed if elapsed > 0 else 0
            print(f"   Batch {batch_num:3d}/{len(review_batches)}: "
                  f"{len(rating_records):6,} records | "
                  f"Total: {total_reviews:10,} | "
                  f"{total_reviews/1e6:5.2f}M | "
                  f"{throughput:8,.0f} rec/sec")

        n_ratings = total_reviews
        reviews_time = time.time() - reviews_start
        print(f"\n✅ Loaded {n_ratings:,} ratings")
        print(f"   Insert time: {timedelta(seconds=int(reviews_time))}")

        # ============================================================
        # PHASE 5: COMMIT TRANSACTION
        # ============================================================
        print("\n" + "="*70)
        print("💾 Committing transaction...")
        commit_start = time.time()
        conn.commit()
        commit_time = time.time() - commit_start
        print(f"   ✓ Transaction committed in {timedelta(seconds=int(commit_time))}")

        # ============================================================
        # PHASE 6: CREATE INDEXES
        # ============================================================
        print("\n🔨 Creating indexes...")
        index_start = time.time()
        db.create_indexes()
        index_time = time.time() - index_start
        print(f"   ✓ Indexes created in {timedelta(seconds=int(index_time))}")

        # ============================================================
        # PHASE 7: RESTORE SAFE MODE & OPTIMIZE
        # ============================================================
        print("\n🔧 Restoring safe mode...")
        conn.execute("PRAGMA synchronous = FULL")
        conn.execute("PRAGMA journal_mode = DELETE")
        print("   ✓ Safe mode restored")

        print("\n🔧 Optimizing database...")
        optimize_start = time.time()
        conn.execute("VACUUM")
        conn.execute("ANALYZE")
        optimize_time = time.time() - optimize_start
        print(f"   ✓ Database optimized in {timedelta(seconds=int(optimize_time))}")

        # ============================================================
        # FINAL STATISTICS
        # ============================================================
        overall_time = time.time() - overall_start

        print("\n" + "="*70)
        print("📊 PERFORMANCE SUMMARY")
        print("="*70)

        print(f"\n⏱️  Products:")
        print(f"   Total: {n_products:,}")
        print(f"   Time: {timedelta(seconds=int(products_time))}")
        print(f"   Throughput: {n_products / products_time:,.0f} records/sec")

        print(f"\n⏱️  Reviews:")
        print(f"   Total: {n_ratings:,}")
        print(f"   Time: {timedelta(seconds=int(reviews_time))}")
        print(f"   Throughput: {n_ratings / reviews_time:,.0f} records/sec")

        print(f"\n⏱️  Commit: {timedelta(seconds=int(commit_time))}")
        print(f"⏱️  Indexing: {timedelta(seconds=int(index_time))}")
        print(f"⏱️  Optimization: {timedelta(seconds=int(optimize_time))}")

        total_records = n_ratings + n_products
        print(f"\n✅ TOTAL TIME: {timedelta(seconds=int(overall_time))}")
        print(f"   Total Records: {total_records:,}")
        print(f"   Overall Throughput: {total_records / overall_time:,.0f} records/sec")

        # Database statistics
        stats = db.get_stats()
        print("\n" + "="*70)
        print("📈 DATABASE STATISTICS")
        print("="*70)
        print(f"   Total Ratings: {stats['n_ratings']:,}")
        print(f"   Unique Users: {stats['n_users']:,}")
        print(f"   Unique Products: {stats['n_products']:,}")
        print(f"   Average Rating: {stats['avg_rating']}")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("   Rolling back transaction...")
        conn.rollback()
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    from ..config import REVIEWS_JSONL, PRODUCTS_JSONL, DB_PATH

    populate_database(REVIEWS_JSONL, PRODUCTS_JSONL, DB_PATH)
