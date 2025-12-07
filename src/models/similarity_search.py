"""
Similarity search using Sentence Transformers and FAISS.
Enables semantic search over product titles.
"""
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from pathlib import Path


class SimilaritySearchModel:
    """Semantic similarity search for product titles."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 256):
        """
        Initialize the similarity search model.

        Args:
            model_name: Name of the sentence transformer model
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.encoder = None
        self.index = None
        self.product_ids = None
        self.product_titles = None
        self.embeddings = None
        self.dimension = None

    def _load_encoder(self):
        """Load the sentence transformer model."""
        if self.encoder is None:
            print(f"Loading sentence transformer model: {self.model_name}")
            self.encoder = SentenceTransformer(self.model_name)
            self.dimension = self.encoder.get_sentence_embedding_dimension()
            print(f"Model loaded. Embedding dimension: {self.dimension}")

    def train(self, products: List[Dict[str, Any]]):
        """
        Build the FAISS index from product titles.

        Args:
            products: List of product dictionaries with 'parent_asin' and 'title'
        """
        self._load_encoder()

        # Extract product IDs and titles
        df = pd.DataFrame(products)
        self.product_ids = df['parent_asin'].to_numpy()
        self.product_titles = df['title'].tolist()

        print(f"Encoding {len(self.product_titles)} product titles...")

        # Encode titles
        self.embeddings = self.encoder.encode(
            self.product_titles,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # Normalize for cosine similarity
        ).astype('float32')

        print(f"Embeddings shape: {self.embeddings.shape}")

        # Build FAISS index
        print("Building FAISS index...")
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine if normalized)
        self.index.add(self.embeddings)

        print(f"Index built with {self.index.ntotal} vectors")

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar product titles.

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            List of search results with product_id, title, and score
        """
        if self.encoder is None or self.index is None:
            raise ValueError("Model not trained. Call train() first or load a trained model.")

        # Encode query
        query_embedding = self.encoder.encode(
            [query],
            normalize_embeddings=True
        ).astype('float32')

        # Search in FAISS index
        scores, indices = self.index.search(query_embedding, top_k)

        # Format results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            results.append({
                'product_id': self.product_ids[idx],
                'title': self.product_titles[idx],
                'score': float(score)
            })

        return results

    def save_model(self, index_path: Path, embeddings_path: Path, product_ids_path: Path):
        """
        Save the FAISS index and metadata.

        Args:
            index_path: Path to save FAISS index
            embeddings_path: Path to save embeddings
            product_ids_path: Path to save product IDs
        """
        if self.index is None:
            raise ValueError("No index to save. Train the model first.")

        # Save FAISS index
        faiss.write_index(self.index, str(index_path))

        # Save embeddings
        np.save(embeddings_path, self.embeddings)

        # Save product metadata
        metadata = {
            'product_ids': self.product_ids,
            'product_titles': self.product_titles,
            'model_name': self.model_name,
            'dimension': self.dimension
        }
        np.save(product_ids_path, metadata)

        print(f"FAISS index saved to {index_path}")
        print(f"Embeddings saved to {embeddings_path}")
        print(f"Product metadata saved to {product_ids_path}")

    def load_model(self, index_path: Path, embeddings_path: Path, product_ids_path: Path):
        """
        Load a trained FAISS index and metadata.

        Args:
            index_path: Path to FAISS index
            embeddings_path: Path to embeddings
            product_ids_path: Path to product IDs
        """
        # Load encoder
        self._load_encoder()

        # Load FAISS index
        self.index = faiss.read_index(str(index_path))

        # Load embeddings
        self.embeddings = np.load(embeddings_path)

        # Load product metadata
        metadata = np.load(product_ids_path, allow_pickle=True).item()
        self.product_ids = metadata['product_ids']
        self.product_titles = metadata['product_titles']
        self.dimension = metadata['dimension']

        print(f"FAISS index loaded from {index_path}")
        print(f"Index contains {self.index.ntotal} vectors")
        print(f"Embedding dimension: {self.dimension}")

    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        if self.index is None:
            return {"status": "not_trained"}

        return {
            "status": "trained",
            "n_products": self.index.ntotal if self.index else 0,
            "model_name": self.model_name,
            "embedding_dimension": self.dimension
        }
