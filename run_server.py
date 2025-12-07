"""
Script to run the API server.
"""
import uvicorn
from src.config import API_HOST, API_PORT, API_DEBUG


if __name__ == "__main__":
    print(f"Starting Recommender System API server...")
    print(f"Server will be available at: http://{API_HOST}:{API_PORT}")
    print(f"API Documentation: http://{API_HOST}:{API_PORT}/docs")
    print(f"Interactive API: http://{API_HOST}:{API_PORT}/redoc")
    print("\nPress CTRL+C to stop the server\n")

    uvicorn.run(
        "src.api.app:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_DEBUG
    )
