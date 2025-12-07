"""
Helper utilities for the recommender system.
"""
from typing import Dict, Any, List
import json


def format_product_display(product: Dict[str, Any]) -> str:
    """
    Format product information for display.

    Args:
        product: Product dictionary

    Returns:
        Formatted string
    """
    title = product.get('title', 'N/A')
    asin = product.get('parent_asin', 'N/A')
    rating = product.get('average_rating', 'N/A')
    price = product.get('price', 'N/A')

    return f"{title}\n  ASIN: {asin}, Rating: {rating}, Price: ${price}"


def get_image_url(product: Dict[str, Any]) -> str:
    """
    Extract the primary image URL from product.

    Args:
        product: Product dictionary

    Returns:
        Image URL or empty string
    """
    images = product.get('images', [])
    if images and len(images) > 0:
        return images[0].get('large') or images[0].get('thumb', '')
    return ''


def calculate_metrics(predictions: List[float], actuals: List[float]) -> Dict[str, float]:
    """
    Calculate evaluation metrics.

    Args:
        predictions: Predicted values
        actuals: Actual values

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import numpy as np

    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae
    }


def log_event(event_type: str, message: str, data: Dict[str, Any] = None):
    """
    Log an event (can be extended to write to file or logging service).

    Args:
        event_type: Type of event
        message: Event message
        data: Additional event data
    """
    import datetime

    log_entry = {
        'timestamp': datetime.datetime.now().isoformat(),
        'event_type': event_type,
        'message': message,
        'data': data or {}
    }

    print(json.dumps(log_entry, indent=2))
