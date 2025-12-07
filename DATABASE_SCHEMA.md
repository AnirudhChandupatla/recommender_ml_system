# Database Schema Documentation

## Overview

The recommender system uses SQLite database with proper foreign key constraints to ensure referential integrity.

## Tables

### 1. `product_catalog` (Parent Table)

**Purpose**: Stores product metadata

**Schema**:
```sql
CREATE TABLE product_catalog (
    parent_asin TEXT PRIMARY KEY,      -- Product identifier (PK)
    title TEXT,                        -- Product title
    main_category TEXT,                -- Main product category
    average_rating REAL,               -- Average rating from reviews
    rating_number INTEGER,             -- Number of ratings
    price REAL,                        -- Product price
    store TEXT,                        -- Store/brand name
    features TEXT,                     -- JSON: Product features
    description TEXT,                  -- JSON: Product description
    images TEXT,                       -- JSON: Product images
    categories TEXT,                   -- JSON: Category hierarchy
    details TEXT                       -- JSON: Additional details
)
```

**JSON Fields**: `features`, `description`, `images`, `categories`, `details` are stored as JSON strings.

---

### 2. `user_item_rating` (Child Table)

**Purpose**: Stores user ratings and reviews

**Schema**:
```sql
CREATE TABLE user_item_rating (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,             -- User identifier
    parent_asin TEXT NOT NULL,         -- Product ASIN (FK)
    rating REAL NOT NULL,              -- Rating value (1-5)
    timestamp INTEGER,                 -- Unix timestamp
    verified_purchase INTEGER DEFAULT 0,  -- Boolean flag
    helpful_vote INTEGER DEFAULT 0,    -- Helpfulness votes
    UNIQUE(user_id, parent_asin),      -- One rating per user per product
    FOREIGN KEY (parent_asin)
        REFERENCES product_catalog(parent_asin)
        ON DELETE CASCADE              -- Delete ratings if product deleted
        ON UPDATE CASCADE              -- Update ratings if product ASIN changes
)
```

---

## Foreign Key Constraints

### Relationship
- **`user_item_rating.parent_asin`** → **`product_catalog.parent_asin`**
- Type: Many-to-One (many ratings for one product)
- Enforcement: `ON DELETE CASCADE`, `ON UPDATE CASCADE`

### What This Ensures

1. **Referential Integrity**: Every rating must reference a valid product
2. **No Orphan Ratings**: Cannot insert a rating for a non-existent product
3. **Cascading Deletes**: If a product is deleted, all its ratings are automatically deleted
4. **Cascading Updates**: If a product's ASIN is updated, all related ratings update automatically

### SQLite Foreign Key Support

**Important**: Foreign keys are **disabled by default** in SQLite. Our implementation enables them:

```python
# In database.py - enabled for all connections
conn.execute("PRAGMA foreign_keys = ON")
```

---

## Indexes

### Performance Indexes (Created AFTER Bulk Insert)

```sql
-- Index on user_id for fast user lookup
CREATE INDEX idx_user_rating ON user_item_rating(user_id);

-- Index on parent_asin for fast product lookup
CREATE INDEX idx_item_rating ON user_item_rating(parent_asin);

-- Index on rating value for filtering by rating
CREATE INDEX idx_rating_value ON user_item_rating(rating);
```

**Why Deferred?**: Indexes are dropped before bulk insert and recreated after for 40-60% performance improvement.

---

## Data Loading Order

**CRITICAL**: Due to foreign key constraints, data must be loaded in this order:

1. **Products FIRST** (parent table - referenced by FK)
2. **Reviews SECOND** (child table - contains FK)

```python
# Correct order in populate_database()
n_products = load_products_to_db_fast(products_path, conn)  # 1. Products
n_ratings = load_reviews_to_db_fast(reviews_path, conn)     # 2. Reviews
```

If loaded in reverse order, SQLite will reject review inserts due to FK violations.

---

## Querying Examples

### Get Product with All Ratings
```sql
SELECT p.*, r.rating, r.user_id, r.timestamp
FROM product_catalog p
LEFT JOIN user_item_rating r ON p.parent_asin = r.parent_asin
WHERE p.parent_asin = 'B08C9LPCQV';
```

### Get User's Rated Products
```sql
SELECT p.title, p.store, r.rating, r.timestamp
FROM user_item_rating r
JOIN product_catalog p ON r.parent_asin = p.parent_asin
WHERE r.user_id = 'AFQLNQNQYFWQZPJQZS6V3NZU4QBQ'
ORDER BY r.timestamp DESC;
```

### Find Products with No Ratings
```sql
SELECT p.*
FROM product_catalog p
LEFT JOIN user_item_rating r ON p.parent_asin = r.parent_asin
WHERE r.id IS NULL;
```

### Count Ratings Per Product
```sql
SELECT p.parent_asin, p.title, COUNT(r.id) as rating_count
FROM product_catalog p
LEFT JOIN user_item_rating r ON p.parent_asin = r.parent_asin
GROUP BY p.parent_asin
ORDER BY rating_count DESC
LIMIT 10;
```

---

## Constraints Validation

### Check FK Constraint is Enabled
```python
import sqlite3
conn = sqlite3.connect('data/processed/appliances_data.db')
result = conn.execute("PRAGMA foreign_keys").fetchone()
print(f"Foreign keys enabled: {result[0] == 1}")  # Should be True
```

### Check FK Violations
```sql
PRAGMA foreign_key_check(user_item_rating);
```

Returns empty if no violations, otherwise shows violating rows.

---

## Migration from Old Schema

If you have an existing database without FK constraints:

### Option 1: Recreate Database
```bash
# Delete old database
rm data/processed/appliances_data.db

# Reload with new schema
python load_data.py
```

### Option 2: Manual Migration
```python
import sqlite3

# Backup old data
old_conn = sqlite3.connect('old_db.db')
ratings = old_conn.execute("SELECT * FROM user_item_rating").fetchall()
products = old_conn.execute("SELECT * FROM product_catalog").fetchall()

# Create new database with FK constraints
new_conn = sqlite3.connect('new_db.db')
new_conn.execute("PRAGMA foreign_keys = ON")

# Create tables (with FK)
from src.etl.database import RecommenderDB
db = RecommenderDB('new_db.db')

# Insert products FIRST
# Insert ratings SECOND
# ...
```

---

## Performance Impact

### Foreign Key Checking Overhead

- **Insert**: ~2-5% slower (FK validation)
- **Update**: ~2-5% slower (FK validation)
- **Delete**: Depends on cascade (can be slower)
- **Query**: No impact (FK not checked on SELECT)

**Overall**: The integrity benefits outweigh the minor performance cost.

### Optimization During Bulk Load

During bulk insert, FK constraints are checked but don't significantly impact performance because:
1. We use single transaction (validation batched)
2. Products are loaded before reviews (no violations)
3. `INSERT OR REPLACE` handles duplicates

---

## Best Practices

### 1. Always Enable Foreign Keys
```python
conn.execute("PRAGMA foreign_keys = ON")
```

### 2. Load Parent Tables First
```python
# Correct order
load_products()  # Parent
load_reviews()   # Child
```

### 3. Handle FK Violations Gracefully
```python
try:
    cursor.execute("INSERT INTO user_item_rating (...)")
except sqlite3.IntegrityError as e:
    if "FOREIGN KEY constraint failed" in str(e):
        print(f"Product {parent_asin} does not exist")
```

### 4. Use Transactions for Consistency
```python
conn.execute("BEGIN TRANSACTION")
# Insert products
# Insert reviews
conn.commit()  # All or nothing
```

### 5. Verify FK After Loading
```python
violations = conn.execute("PRAGMA foreign_key_check").fetchall()
assert len(violations) == 0, f"FK violations: {violations}"
```

---

## Schema Diagram

```
┌─────────────────────────────┐
│    product_catalog          │
│  (Parent/Referenced Table)  │
├─────────────────────────────┤
│ parent_asin (PK) ◄──────────┼───┐
│ title                       │   │
│ main_category               │   │
│ average_rating              │   │
│ ...                         │   │
└─────────────────────────────┘   │
                                  │
                                  │ Foreign Key
                                  │ ON DELETE CASCADE
                                  │ ON UPDATE CASCADE
                                  │
┌─────────────────────────────┐   │
│    user_item_rating         │   │
│   (Child/Referencing Table) │   │
├─────────────────────────────┤   │
│ id (PK)                     │   │
│ user_id                     │   │
│ parent_asin (FK) ───────────┼───┘
│ rating                      │
│ timestamp                   │
│ ...                         │
└─────────────────────────────┘
```

---

## Future Enhancements

Potential schema improvements:

1. **Users Table**: Create separate `users` table with `user_id` as PK
2. **Categories Table**: Normalize categories into separate table
3. **Stores Table**: Normalize stores/brands
4. **Review Text**: Add review title/text fields
5. **Composite Indexes**: Add multi-column indexes for common queries

---

## References

- [SQLite Foreign Key Support](https://www.sqlite.org/foreignkeys.html)
- [SQLite PRAGMA foreign_keys](https://www.sqlite.org/pragma.html#pragma_foreign_keys)
- [Database Normalization](https://en.wikipedia.org/wiki/Database_normalization)
