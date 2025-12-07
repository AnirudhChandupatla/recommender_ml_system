# Admin API Documentation for ML Lifecycle Testing

## Overview

This document describes the Admin API endpoints designed for testing the end-to-end ML project lifecycle. These endpoints allow you to simulate real-world scenarios such as:

1. **New users rating products** → Insert new reviews
2. **Users closing accounts** → Delete user reviews
3. **New products being introduced** → Insert new products
4. **Products being delisted** → Delete products

All endpoints maintain **foreign key integrity** - when products are deleted, associated reviews are automatically deleted via CASCADE.

---

## Base URL

```
http://localhost:8000
```

---

## Table of Contents

1. [Authentication](#authentication)
2. [Review Management](#review-management)
   - [Insert Reviews (JSON)](#insert-reviews-json)
   - [Insert Reviews (CSV Upload)](#insert-reviews-csv-upload)
   - [Delete First N Reviews](#delete-first-n-reviews)
   - [Delete Last N Reviews](#delete-last-n-reviews)
   - [Delete Reviews by IDs](#delete-reviews-by-ids)
   - [Delete Reviews by User IDs](#delete-reviews-by-user-ids)
3. [Product Management](#product-management)
   - [Insert Products (JSON)](#insert-products-json)
   - [Insert Products (CSV Upload)](#insert-products-csv-upload)
   - [Delete First N Products](#delete-first-n-products)
   - [Delete Last N Products](#delete-last-n-products)
   - [Delete Products by ASINs](#delete-products-by-asins)
4. [Usage Examples](#usage-examples)
5. [Testing Workflow](#testing-workflow)

---

## Authentication

Currently, no authentication is required for these endpoints. In production, you should add authentication middleware.

---

## Review Management

### Insert Reviews (JSON)

**Endpoint:** `POST /admin/reviews/insert`

**Use Case:** Simulate new users rating products

**Request Body:**
```json
{
  "reviews": [
    {
      "user_id": "NEW_USER_123",
      "parent_asin": "B08C9LPCQV",
      "rating": 5.0,
      "timestamp": 1609459200000,
      "verified_purchase": 1,
      "helpful_vote": 0
    },
    {
      "user_id": "NEW_USER_456",
      "parent_asin": "B08C9LPCQV",
      "rating": 4.0,
      "timestamp": 1609459300000,
      "verified_purchase": 1,
      "helpful_vote": 2
    }
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Successfully inserted 2 reviews",
  "rows_affected": 2,
  "db_stats": {
    "n_ratings": 2000002,
    "n_users": 1700001,
    "n_products": 94000,
    "avg_rating": 4.2
  }
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/admin/reviews/insert" \
  -H "Content-Type: application/json" \
  -d '{
    "reviews": [
      {
        "user_id": "TEST_USER_1",
        "parent_asin": "B08C9LPCQV",
        "rating": 5.0,
        "timestamp": 1609459200000,
        "verified_purchase": 1,
        "helpful_vote": 0
      }
    ]
  }'
```

---

### Insert Reviews (CSV Upload)

**Endpoint:** `POST /admin/reviews/insert-csv`

**Use Case:** Bulk insert future reviews from CSV file

**CSV Format:**
```csv
user_id,parent_asin,rating,timestamp,verified_purchase,helpful_vote
USER123,B08C9LPCQV,5.0,1609459200000,1,0
USER456,B08ANOTHER,4.0,1609459300000,1,2
USER789,B08PRODUCT,3.5,1609459400000,0,0
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/admin/reviews/insert-csv" \
  -F "file=@future_reviews.csv"
```

**Response:**
```json
{
  "status": "success",
  "message": "Successfully inserted 10000 reviews from CSV",
  "rows_affected": 10000,
  "db_stats": {
    "n_ratings": 2010000,
    "n_users": 1701000,
    "n_products": 94000,
    "avg_rating": 4.2
  }
}
```

---

### Delete First N Reviews

**Endpoint:** `DELETE /admin/reviews/first/{n}`

**Use Case:** Remove oldest reviews for testing

**Path Parameter:**
- `n` - Number of reviews to delete

**Example:** `DELETE /admin/reviews/first/100`

**cURL Example:**
```bash
curl -X DELETE "http://localhost:8000/admin/reviews/first/100"
```

**Response:**
```json
{
  "status": "success",
  "message": "Successfully deleted first 100 reviews",
  "rows_affected": 100,
  "db_stats": {
    "n_ratings": 1999900,
    "n_users": 1700000,
    "n_products": 94000,
    "avg_rating": 4.2
  }
}
```

---

### Delete Last N Reviews

**Endpoint:** `DELETE /admin/reviews/last/{n}`

**Use Case:** Remove most recent reviews for testing

**Path Parameter:**
- `n` - Number of reviews to delete

**Example:** `DELETE /admin/reviews/last/50`

**cURL Example:**
```bash
curl -X DELETE "http://localhost:8000/admin/reviews/last/50"
```

---

### Delete Reviews by IDs

**Endpoint:** `DELETE /admin/reviews/by-ids`

**Use Case:** Remove specific reviews by their database IDs

**Request Body:**
```json
{
  "ids": [1, 2, 3, 100, 200]
}
```

**cURL Example:**
```bash
curl -X DELETE "http://localhost:8000/admin/reviews/by-ids" \
  -H "Content-Type: application/json" \
  -d '{"ids": [1, 2, 3, 100, 200]}'
```

---

### Delete Reviews by User IDs

**Endpoint:** `DELETE /admin/reviews/by-users`

**Use Case:** Simulate user account closure - removes all reviews for specified users

**Request Body:**
```json
{
  "user_ids": ["USER123", "USER456", "USER789"]
}
```

**cURL Example:**
```bash
curl -X DELETE "http://localhost:8000/admin/reviews/by-users" \
  -H "Content-Type: application/json" \
  -d '{"user_ids": ["USER123", "USER456"]}'
```

**Response:**
```json
{
  "status": "success",
  "message": "Successfully deleted 25 reviews for 2 users",
  "rows_affected": 25,
  "db_stats": {
    "n_ratings": 1999975,
    "n_users": 1699998,
    "n_products": 94000,
    "avg_rating": 4.2
  }
}
```

---

## Product Management

### Insert Products (JSON)

**Endpoint:** `POST /admin/products/insert`

**Use Case:** Add new products to the catalog

**Request Body:**
```json
{
  "products": [
    {
      "parent_asin": "B08NEWPROD",
      "title": "New Smart Refrigerator",
      "main_category": "Appliances",
      "average_rating": 0.0,
      "rating_number": 0,
      "price": 1299.99,
      "store": "SmartHome",
      "features": ["Smart connectivity", "Energy efficient", "Large capacity"],
      "description": ["A revolutionary smart refrigerator"],
      "images": [
        {
          "large": "https://example.com/image_large.jpg",
          "thumb": "https://example.com/image_thumb.jpg"
        }
      ],
      "categories": [["Appliances", "Kitchen", "Refrigerators"]],
      "details": {
        "Weight": "200 lbs",
        "Dimensions": "36 x 70 x 30 inches"
      }
    }
  ]
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/admin/products/insert" \
  -H "Content-Type: application/json" \
  -d '{
    "products": [
      {
        "parent_asin": "B08NEWPROD",
        "title": "New Product",
        "main_category": "Appliances",
        "price": 99.99,
        "store": "TestStore"
      }
    ]
  }'
```

**Response:**
```json
{
  "status": "success",
  "message": "Successfully inserted 1 products",
  "rows_affected": 1,
  "db_stats": {
    "n_ratings": 2000000,
    "n_users": 1700000,
    "n_products": 94001,
    "avg_rating": 4.2
  }
}
```

---

### Insert Products (CSV Upload)

**Endpoint:** `POST /admin/products/insert-csv`

**Use Case:** Bulk insert future products from CSV file

**CSV Format:**
```csv
parent_asin,title,main_category,average_rating,rating_number,price,store,features,description,images,categories,details
B08PROD1,"Product 1",Appliances,0.0,0,99.99,Store1,"[""Feature 1""]","[""Description""]","[{""large"":""url""}]","[[""Appliances""]]","{}"
B08PROD2,"Product 2",Appliances,0.0,0,149.99,Store2,"[""Feature 2""]","[""Description""]","[{""large"":""url""}]","[[""Appliances""]]","{}"
```

**Note:** JSON fields (features, description, images, categories, details) must be JSON-encoded strings in the CSV.

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/admin/products/insert-csv" \
  -F "file=@future_products.csv"
```

---

### Delete First N Products

**Endpoint:** `DELETE /admin/products/first/{n}`

**Use Case:** Remove products for testing. **CASCADE also deletes associated reviews.**

**Path Parameter:**
- `n` - Number of products to delete

**Example:** `DELETE /admin/products/first/10`

**cURL Example:**
```bash
curl -X DELETE "http://localhost:8000/admin/products/first/10"
```

**Response:**
```json
{
  "status": "success",
  "message": "Successfully deleted first 10 products (CASCADE also deleted associated reviews)",
  "rows_affected": 10,
  "db_stats": {
    "n_ratings": 1999800,
    "n_users": 1700000,
    "n_products": 93990,
    "avg_rating": 4.2
  }
}
```

---

### Delete Last N Products

**Endpoint:** `DELETE /admin/products/last/{n}`

**Use Case:** Remove products for testing. **CASCADE also deletes associated reviews.**

**Path Parameter:**
- `n` - Number of products to delete

**Example:** `DELETE /admin/products/last/5`

**cURL Example:**
```bash
curl -X DELETE "http://localhost:8000/admin/products/last/5"
```

---

### Delete Products by ASINs

**Endpoint:** `DELETE /admin/products/by-asins`

**Use Case:** Simulate product delisting. **CASCADE also deletes associated reviews.**

**Request Body:**
```json
{
  "asins": ["B08C9LPCQV", "B08ANOTHER", "B08PRODUCT"]
}
```

**cURL Example:**
```bash
curl -X DELETE "http://localhost:8000/admin/products/by-asins" \
  -H "Content-Type: application/json" \
  -d '{"asins": ["B08C9LPCQV", "B08ANOTHER"]}'
```

**Response:**
```json
{
  "status": "success",
  "message": "Successfully deleted 2 products (CASCADE also deleted associated reviews)",
  "rows_affected": 2,
  "db_stats": {
    "n_ratings": 1998500,
    "n_users": 1700000,
    "n_products": 93998,
    "avg_rating": 4.2
  }
}
```

---

## Usage Examples

### Example 1: Testing New User Ratings

**Scenario:** Simulate 100 new users rating various products

```python
import requests
import random

# Generate 100 new reviews
new_reviews = []
user_ids = [f"NEW_USER_{i}" for i in range(1, 101)]
product_asins = ["B08C9LPCQV", "B08ANOTHER", "B08PRODUCT"]

for user_id in user_ids:
    new_reviews.append({
        "user_id": user_id,
        "parent_asin": random.choice(product_asins),
        "rating": random.choice([1.0, 2.0, 3.0, 4.0, 5.0]),
        "timestamp": int(time.time() * 1000),
        "verified_purchase": 1,
        "helpful_vote": 0
    })

# Insert via API
response = requests.post(
    "http://localhost:8000/admin/reviews/insert",
    json={"reviews": new_reviews}
)

print(response.json())
```

---

### Example 2: Simulating User Account Closure

**Scenario:** User "USER123" closes their account

```python
import requests

# Delete all reviews by USER123
response = requests.delete(
    "http://localhost:8000/admin/reviews/by-users",
    json={"user_ids": ["USER123"]}
)

print(f"Deleted {response.json()['rows_affected']} reviews")
```

---

### Example 3: Product Delisting

**Scenario:** Product "B08C9LPCQV" is being discontinued

```python
import requests

# Delete product and all associated reviews
response = requests.delete(
    "http://localhost:8000/admin/products/by-asins",
    json={"asins": ["B08C9LPCQV"]}
)

print(f"Deleted product. Also deleted {response.json()['rows_affected']} associated reviews via CASCADE")
```

---

### Example 4: Bulk Insert Future Data from CSV

**Scenario:** Load the 10K future reviews from CSV file

```bash
# Upload future_reviews.csv
curl -X POST "http://localhost:8000/admin/reviews/insert-csv" \
  -F "file=@data/processed/future_reviews.csv"

# Upload future_products.csv
curl -X POST "http://localhost:8000/admin/products/insert-csv" \
  -F "file=@data/processed/future_products.csv"
```

---

## Testing Workflow

### Initial Setup

1. **Load data with splitting enabled:**
   ```bash
   python load_data.py --split-future-data --n-future-reviews 10000 --n-future-products 1000
   ```

   This creates:
   - Database with main data
   - `future_reviews.csv` (10,000 reviews)
   - `future_products.csv` (1,000 products)

2. **Start the API server:**
   ```bash
   python -m src.api.app
   # Or
   uvicorn src.api.app:app --reload
   ```

3. **Train initial models:**
   ```bash
   curl -X POST "http://localhost:8000/training" \
     -H "Content-Type: application/json" \
     -d '{"model_type": "mf", "force_retrain": false}'

   curl -X POST "http://localhost:8000/training" \
     -H "Content-Type: application/json" \
     -d '{"model_type": "ss", "force_retrain": false}'
   ```

---

### Lifecycle Testing Scenarios

#### Scenario 1: New User Onboarding
```bash
# 1. Insert new reviews
curl -X POST "http://localhost:8000/admin/reviews/insert-csv" \
  -F "file=@future_reviews.csv"

# 2. Retrain models with new data
curl -X POST "http://localhost:8000/training" \
  -d '{"model_type": "mf", "force_retrain": true}'

# 3. Get recommendations for new users
curl -X POST "http://localhost:8000/matrix_factor/user" \
  -d '{"user_id": "NEW_USER_1", "n": 10}'
```

#### Scenario 2: Product Catalog Update
```bash
# 1. Insert new products
curl -X POST "http://localhost:8000/admin/products/insert-csv" \
  -F "file=@future_products.csv"

# 2. Retrain similarity search model
curl -X POST "http://localhost:8000/training" \
  -d '{"model_type": "ss", "force_retrain": true}'

# 3. Search for new products
curl -X POST "http://localhost:8000/search" \
  -d '{"query": "refrigerator", "top_k": 10}'
```

#### Scenario 3: User Churn (Account Closure)
```bash
# Delete reviews from churned users
curl -X DELETE "http://localhost:8000/admin/reviews/by-users" \
  -H "Content-Type: application/json" \
  -d '{"user_ids": ["USER_TO_DELETE_1", "USER_TO_DELETE_2"]}'

# Retrain with updated data
curl -X POST "http://localhost:8000/training" \
  -d '{"model_type": "mf", "force_retrain": true}'
```

#### Scenario 4: Product Discontinuation
```bash
# Delete discontinued products (CASCADE deletes reviews)
curl -X DELETE "http://localhost:8000/admin/products/by-asins" \
  -H "Content-Type: application/json" \
  -d '{"asins": ["B08DISCONTINUED1", "B08DISCONTINUED2"]}'

# Retrain both models
curl -X POST "http://localhost:8000/training" \
  -d '{"model_type": "mf", "force_retrain": true}'

curl -X POST "http://localhost:8000/training" \
  -d '{"model_type": "ss", "force_retrain": true}'
```

---

## Interactive API Documentation

Once the server is running, visit:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

These provide interactive documentation where you can test endpoints directly from the browser.

---

## Error Handling

All endpoints return appropriate HTTP status codes:

- `200 OK` - Successful operation
- `400 Bad Request` - Invalid input (e.g., negative N, missing required fields)
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error (with detailed message)

**Error Response Example:**
```json
{
  "detail": "Failed to insert reviews: FOREIGN KEY constraint failed"
}
```

---

## Foreign Key Constraints

**Important:** The database enforces foreign key constraints:

1. **Reviews must reference valid products:**
   - You cannot insert a review with `parent_asin` that doesn't exist in `product_catalog`
   - Insert products first, then reviews

2. **CASCADE deletion:**
   - When a product is deleted, all its reviews are automatically deleted
   - This simulates real-world product delisting

3. **Orphan prevention:**
   - The system prevents orphan reviews (reviews without products)
   - Always ensure products exist before inserting reviews

---

## Database Statistics

All mutation endpoints return updated database statistics in the response:

```json
{
  "db_stats": {
    "n_ratings": 2000000,     // Total reviews
    "n_users": 1700000,       // Unique users
    "n_products": 94000,      // Total products
    "avg_rating": 4.2         // Average rating
  }
}
```

Use these to monitor the state of your database during testing.

---

## Notes for Admin Panel Development

When building the admin panel UI, consider:

1. **File Upload Components:**
   - Support CSV file upload for bulk operations
   - Validate CSV format before upload
   - Show preview of data to be inserted

2. **Confirmation Dialogs:**
   - Always confirm destructive operations (deletions)
   - Show impact (e.g., "This will delete 100 reviews")

3. **Progress Indicators:**
   - Bulk operations may take time
   - Show loading states during API calls

4. **Database Stats Display:**
   - Show real-time database statistics after each operation
   - Visualize data distribution (users, products, ratings)

5. **Audit Log:**
   - Track all admin operations
   - Store who did what and when

6. **Validation:**
   - Validate ASINs exist before deletion
   - Prevent deletion if would break FK constraints
   - Check CSV format matches expected schema

---

## Future Enhancements

Potential improvements for production:

1. **Authentication & Authorization:**
   - Add JWT-based authentication
   - Role-based access control (admin vs. user)

2. **Batch Operations:**
   - Add endpoints for batch operations with progress tracking
   - Support for background jobs with status polling

3. **Dry-Run Mode:**
   - Add `?dry_run=true` parameter to preview changes
   - Return impact analysis without executing

4. **Rollback Support:**
   - Create snapshots before operations
   - Allow rollback to previous state

5. **Rate Limiting:**
   - Prevent abuse of admin endpoints
   - Implement request throttling

---

## Support

For issues or questions:
- Check the interactive docs at `/docs`
- Review the database schema in `DATABASE_SCHEMA.md`
- Examine the source code in `src/api/app.py` and `src/etl/database.py`
