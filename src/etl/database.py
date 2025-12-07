"""
Database schema and operations for the recommender system.
"""
import sqlite3
from typing import List, Tuple, Optional, Dict, Any
from contextlib import contextmanager
import json


class RecommenderDB:
    """Handles database operations for the recommender system."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._create_tables()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        # Enable foreign key constraints (disabled by default in SQLite)
        conn.execute("PRAGMA foreign_keys = ON")

        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _create_tables(self):
        """Create necessary database tables with foreign key constraints."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Enable foreign key constraints (disabled by default in SQLite)
            cursor.execute("PRAGMA foreign_keys = ON")

            # Product catalog table (must be created FIRST - referenced by FK)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS product_catalog (
                    parent_asin TEXT PRIMARY KEY,
                    title TEXT,
                    main_category TEXT,
                    average_rating REAL,
                    rating_number INTEGER,
                    price REAL,
                    store TEXT,
                    features TEXT,
                    description TEXT,
                    images TEXT,
                    categories TEXT,
                    details TEXT
                )
            """)

            # User-Item-Rating table (with FK constraint)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_item_rating (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    parent_asin TEXT NOT NULL,
                    rating REAL NOT NULL,
                    timestamp INTEGER,
                    verified_purchase INTEGER DEFAULT 0,
                    helpful_vote INTEGER DEFAULT 0,
                    UNIQUE(user_id, parent_asin),
                    FOREIGN KEY (parent_asin) REFERENCES product_catalog(parent_asin)
                        ON DELETE CASCADE
                        ON UPDATE CASCADE
                )
            """)

            # Note: Indexes are created separately via create_indexes()
            # for better bulk insert performance

    def insert_ratings(self, ratings: List[Dict[str, Any]]) -> int:
        """
        Insert user-item ratings in batch.

        Args:
            ratings: List of rating dictionaries

        Returns:
            Number of rows inserted
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.executemany("""
                INSERT OR REPLACE INTO user_item_rating
                (user_id, parent_asin, rating, timestamp, verified_purchase, helpful_vote)
                VALUES (:user_id, :parent_asin, :rating, :timestamp, :verified_purchase, :helpful_vote)
            """, ratings)

            return cursor.rowcount

    def insert_products(self, products: List[Dict[str, Any]]) -> int:
        """
        Insert products in batch.

        Args:
            products: List of product dictionaries

        Returns:
            Number of rows inserted
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Convert list/dict fields to JSON strings
            processed_products = []
            for product in products:
                processed = product.copy()
                for field in ['features', 'description', 'images', 'categories', 'details']:
                    if field in processed and processed[field] is not None:
                        if isinstance(processed[field], (list, dict)):
                            processed[field] = json.dumps(processed[field])
                processed_products.append(processed)

            cursor.executemany("""
                INSERT OR REPLACE INTO product_catalog
                (parent_asin, title, main_category, average_rating, rating_number,
                 price, store, features, description, images, categories, details)
                VALUES (:parent_asin, :title, :main_category, :average_rating, :rating_number,
                        :price, :store, :features, :description, :images, :categories, :details)
            """, processed_products)

            return cursor.rowcount

    def get_all_ratings(self) -> List[Tuple[str, str, float]]:
        """
        Get all ratings as (user_id, parent_asin, rating) tuples.

        Returns:
            List of rating tuples
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT user_id, parent_asin, rating
                FROM user_item_rating
                ORDER BY user_id, parent_asin
            """)
            return cursor.fetchall()

    def get_all_products(self) -> List[Dict[str, Any]]:
        """
        Get all products with their metadata.

        Returns:
            List of product dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT parent_asin, title, main_category, average_rating,
                       rating_number, price, store, features, description,
                       images, categories, details
                FROM product_catalog
            """)

            rows = cursor.fetchall()
            products = []
            for row in rows:
                product = dict(row)
                # Parse JSON fields back to Python objects
                for field in ['features', 'description', 'images', 'categories', 'details']:
                    if product[field]:
                        try:
                            product[field] = json.loads(product[field])
                        except json.JSONDecodeError:
                            pass
                products.append(product)

            return products

    def get_product_by_asin(self, parent_asin: str) -> Optional[Dict[str, Any]]:
        """
        Get a single product by ASIN.

        Args:
            parent_asin: Product ASIN

        Returns:
            Product dictionary or None
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT parent_asin, title, main_category, average_rating,
                       rating_number, price, store, features, description,
                       images, categories, details
                FROM product_catalog
                WHERE parent_asin = ?
            """, (parent_asin,))

            row = cursor.fetchone()
            if row:
                product = dict(row)
                # Parse JSON fields
                for field in ['features', 'description', 'images', 'categories', 'details']:
                    if product[field]:
                        try:
                            product[field] = json.loads(product[field])
                        except json.JSONDecodeError:
                            pass
                return product
            return None

    def get_products_by_asins(self, asins: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get multiple products by ASINs in a single query (batch lookup for performance).
        Handles large lists by chunking to avoid SQLite's variable limit (~999).

        Args:
            asins: List of product ASINs

        Returns:
            Dictionary mapping ASIN to product dictionary
        """
        if not asins:
            return {}

        products = {}
        # SQLite has a limit of 999 variables, so chunk the requests
        chunk_size = 500  # Use 500 to be safe

        for i in range(0, len(asins), chunk_size):
            chunk = asins[i:i + chunk_size]

            with self.get_connection() as conn:
                cursor = conn.cursor()
                placeholders = ','.join('?' * len(chunk))
                cursor.execute(f"""
                    SELECT parent_asin, title, main_category, average_rating,
                           rating_number, price, store, features, description,
                           images, categories, details
                    FROM product_catalog
                    WHERE parent_asin IN ({placeholders})
                """, chunk)

                rows = cursor.fetchall()
                for row in rows:
                    product = dict(row)
                    # Parse JSON fields
                    for field in ['features', 'description', 'images', 'categories', 'details']:
                        if product[field]:
                            try:
                                product[field] = json.loads(product[field])
                            except json.JSONDecodeError:
                                pass
                    products[product['parent_asin']] = product

        return products

    def get_user_ratings(self, user_id: str) -> List[Tuple[str, float]]:
        """
        Get all ratings for a specific user.

        Args:
            user_id: User identifier

        Returns:
            List of (parent_asin, rating) tuples
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT parent_asin, rating
                FROM user_item_rating
                WHERE user_id = ?
                ORDER BY rating DESC
            """, (user_id,))
            return cursor.fetchall()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with statistics
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM user_item_rating")
            n_ratings = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT user_id) FROM user_item_rating")
            n_users = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM product_catalog")
            n_products = cursor.fetchone()[0]

            cursor.execute("SELECT AVG(rating) FROM user_item_rating")
            avg_rating = cursor.fetchone()[0]

            return {
                "n_ratings": n_ratings,
                "n_users": n_users,
                "n_products": n_products,
                "avg_rating": round(avg_rating, 2) if avg_rating else 0
            }

    def drop_indexes(self):
        """
        Drop indexes before bulk insert for better performance.
        Call this BEFORE loading large amounts of data.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DROP INDEX IF EXISTS idx_user_rating")
            cursor.execute("DROP INDEX IF EXISTS idx_item_rating")
            cursor.execute("DROP INDEX IF EXISTS idx_rating_value")

    def create_indexes(self):
        """
        Create indexes after bulk insert.
        Call this AFTER loading data to optimize query performance.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_rating
                ON user_item_rating(user_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_item_rating
                ON user_item_rating(parent_asin)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_rating_value
                ON user_item_rating(rating)
            """)

    # ============================================================
    # DELETION METHODS FOR LIFECYCLE TESTING
    # ============================================================

    def delete_reviews_by_ids(self, review_ids: List[int]) -> int:
        """
        Delete reviews by their IDs.

        Args:
            review_ids: List of review IDs to delete

        Returns:
            Number of rows deleted
        """
        if not review_ids:
            return 0

        with self.get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ','.join('?' * len(review_ids))
            cursor.execute(f"""
                DELETE FROM user_item_rating
                WHERE id IN ({placeholders})
            """, review_ids)
            return cursor.rowcount

    def delete_reviews_by_user_ids(self, user_ids: List[str]) -> int:
        """
        Delete all reviews by specific user IDs (simulates user account closure).

        Args:
            user_ids: List of user IDs to delete reviews for

        Returns:
            Number of rows deleted
        """
        if not user_ids:
            return 0

        with self.get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ','.join('?' * len(user_ids))
            cursor.execute(f"""
                DELETE FROM user_item_rating
                WHERE user_id IN ({placeholders})
            """, user_ids)
            return cursor.rowcount

    def delete_reviews_first_n(self, n: int) -> int:
        """
        Delete the first N reviews (ordered by ID).

        Args:
            n: Number of reviews to delete

        Returns:
            Number of rows deleted
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM user_item_rating
                WHERE id IN (
                    SELECT id FROM user_item_rating
                    ORDER BY id ASC
                    LIMIT ?
                )
            """, (n,))
            return cursor.rowcount

    def delete_reviews_last_n(self, n: int) -> int:
        """
        Delete the last N reviews (ordered by ID).

        Args:
            n: Number of reviews to delete

        Returns:
            Number of rows deleted
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM user_item_rating
                WHERE id IN (
                    SELECT id FROM user_item_rating
                    ORDER BY id DESC
                    LIMIT ?
                )
            """, (n,))
            return cursor.rowcount

    def delete_products_by_asins(self, asins: List[str]) -> int:
        """
        Delete products by their ASINs.
        CASCADE will also delete all associated reviews.

        Args:
            asins: List of product ASINs to delete

        Returns:
            Number of rows deleted
        """
        if not asins:
            return 0

        with self.get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ','.join('?' * len(asins))
            cursor.execute(f"""
                DELETE FROM product_catalog
                WHERE parent_asin IN ({placeholders})
            """, asins)
            return cursor.rowcount

    def delete_products_first_n(self, n: int) -> int:
        """
        Delete the first N products (ordered by parent_asin).
        CASCADE will also delete all associated reviews.

        Args:
            n: Number of products to delete

        Returns:
            Number of rows deleted
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM product_catalog
                WHERE parent_asin IN (
                    SELECT parent_asin FROM product_catalog
                    ORDER BY parent_asin ASC
                    LIMIT ?
                )
            """, (n,))
            return cursor.rowcount

    def delete_products_last_n(self, n: int) -> int:
        """
        Delete the last N products (ordered by parent_asin).
        CASCADE will also delete all associated reviews.

        Args:
            n: Number of products to delete

        Returns:
            Number of rows deleted
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM product_catalog
                WHERE parent_asin IN (
                    SELECT parent_asin FROM product_catalog
                    ORDER BY parent_asin DESC
                    LIMIT ?
                )
            """, (n,))
            return cursor.rowcount

    # ============================================================
    # CSV/FILE LOADING METHODS FOR FUTURE DATA
    # ============================================================

    def load_reviews_from_csv(self, csv_path: str) -> int:
        """
        Load reviews from CSV file.

        Args:
            csv_path: Path to CSV file with reviews

        Returns:
            Number of rows inserted
        """
        import csv

        reviews = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                reviews.append({
                    'user_id': row['user_id'],
                    'parent_asin': row['parent_asin'],
                    'rating': float(row['rating']),
                    'timestamp': int(row['timestamp']) if row['timestamp'] else None,
                    'verified_purchase': int(row['verified_purchase']) if row.get('verified_purchase') else 0,
                    'helpful_vote': int(row['helpful_vote']) if row.get('helpful_vote') else 0
                })

        return self.insert_ratings(reviews)

    def load_products_from_csv(self, csv_path: str) -> int:
        """
        Load products from CSV file.

        Args:
            csv_path: Path to CSV file with products

        Returns:
            Number of rows inserted
        """
        import csv

        products = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # JSON fields are already serialized in CSV
                products.append({
                    'parent_asin': row['parent_asin'],
                    'title': row.get('title'),
                    'main_category': row.get('main_category'),
                    'average_rating': float(row['average_rating']) if row.get('average_rating') else None,
                    'rating_number': int(row['rating_number']) if row.get('rating_number') else None,
                    'price': float(row['price']) if row.get('price') else None,
                    'store': row.get('store'),
                    'features': row.get('features'),  # Already JSON string
                    'description': row.get('description'),  # Already JSON string
                    'images': row.get('images'),  # Already JSON string
                    'categories': row.get('categories'),  # Already JSON string
                    'details': row.get('details')  # Already JSON string
                })

        # Don't re-serialize JSON fields since they're already strings in CSV
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT OR REPLACE INTO product_catalog
                (parent_asin, title, main_category, average_rating, rating_number,
                 price, store, features, description, images, categories, details)
                VALUES (:parent_asin, :title, :main_category, :average_rating, :rating_number,
                        :price, :store, :features, :description, :images, :categories, :details)
            """, products)
            return cursor.rowcount
