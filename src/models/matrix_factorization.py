"""
Matrix Factorization model for collaborative filtering recommendations.
Uses Implicit library's ALS (Alternating Least Squares).
"""
import os
import pickle
import numpy as np
import pandas as pd
from scipy import sparse
from pandas.api.types import CategoricalDtype
from implicit.als import AlternatingLeastSquares
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path


class MatrixFactorizationModel:
    """Matrix Factorization recommender using ALS."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model with configuration.

        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model = None
        self.user_item_matrix = None
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}

    def prepare_sparse_matrix(self, ratings: List[Tuple[str, str, float]]) -> sparse.csr_matrix:
        """
        Convert ratings to sparse user-item matrix.

        Args:
            ratings: List of (user_id, item_id, rating) tuples

        Returns:
            CSR sparse matrix of shape (n_users, n_items)
        """
        print("Preparing sparse matrix from ratings...")

        df = pd.DataFrame(ratings, columns=['user_id', 'parent_asin', 'rating'])

        # Create categorical types
        users = df["user_id"].unique()
        items = df["parent_asin"].unique()

        user_type = CategoricalDtype(categories=users, ordered=False)
        item_type = CategoricalDtype(categories=items, ordered=False)

        # Convert to categorical codes
        row_idx = df["user_id"].astype(user_type).cat.codes
        col_idx = df["parent_asin"].astype(item_type).cat.codes

        # Create mappings
        self.user_to_idx = {user_id: int(idx) for user_id, idx in zip(df["user_id"].values, row_idx.values)}
        self.idx_to_user = {int(idx): user_id for idx, user_id in zip(row_idx.values, df["user_id"].values)}
        self.item_to_idx = {item_id: int(idx) for item_id, idx in zip(df["parent_asin"].values, col_idx.values)}
        self.idx_to_item = {int(idx): item_id for idx, item_id in zip(col_idx.values, df["parent_asin"].values)}

        # Create sparse matrix
        n_users = len(users)
        n_items = len(items)
        data = df["rating"].values

        coo = sparse.coo_matrix((data, (row_idx, col_idx)), shape=(n_users, n_items))
        csr_matrix = coo.tocsr()

        print(f"Created sparse matrix: {csr_matrix.shape} ({n_users} users, {n_items} items)")
        return csr_matrix

    def train(self, ratings: List[Tuple[str, str, float]]):
        """
        Train the ALS model.

        Args:
            ratings: List of (user_id, item_id, rating) tuples
        """
        # Set environment variable for OpenBLAS
        os.environ['OPENBLAS_NUM_THREADS'] = '1'

        # Prepare data
        self.user_item_matrix = self.prepare_sparse_matrix(ratings)

        # Initialize model
        self.model = AlternatingLeastSquares(
            factors=self.config.get('factors', 128),
            regularization=self.config.get('regularization', 1.0),
            iterations=self.config.get('iterations', 100),
            alpha=self.config.get('alpha', 10),
            use_gpu=self.config.get('use_gpu', False),
            calculate_training_loss=True
        )

        # Train model
        print("Training ALS model...")
        self.model.fit(self.user_item_matrix)
        print("Training completed!")

    def get_similar_items(self, item_id: str, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get similar items for a given item.

        Args:
            item_id: Product ASIN
            n: Number of recommendations

        Returns:
            List of similar items with scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if item_id not in self.item_to_idx:
            raise ValueError(f"Item {item_id} not found in training data.")

        item_idx = self.item_to_idx[item_id]
        indices, scores = self.model.similar_items(item_idx, N=n)

        results = []
        for idx, score in zip(indices, scores):
            results.append({
                'item_id': self.idx_to_item[int(idx)],
                'score': float(score)
            })

        return results

    def recommend_for_user(self, user_id: str, n: int = 10,
                           filter_already_liked: bool = True) -> List[Dict[str, Any]]:
        """
        Recommend items for a user.

        Args:
            user_id: User identifier
            n: Number of recommendations
            filter_already_liked: Whether to filter items the user has already rated

        Returns:
            List of recommended items with scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if user_id not in self.user_to_idx:
            raise ValueError(f"User {user_id} not found in training data.")

        user_idx = self.user_to_idx[user_id]
        user_items = self.user_item_matrix[user_idx]

        indices, scores = self.model.recommend(
            user_idx,
            user_items,
            N=n,
            filter_already_liked_items=filter_already_liked
        )

        results = []
        for idx, score in zip(indices, scores):
            results.append({
                'item_id': self.idx_to_item[int(idx)],
                'score': float(score)
            })

        return results

    def save_model(self, model_path: Path, mappings_path: Path, matrix_path: Path):
        """
        Save the trained model and mappings.

        Args:
            model_path: Path to save model parameters
            mappings_path: Path to save ID mappings
            matrix_path: Path to save sparse matrix
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        # Save model parameters
        model_data = {
            'user_factors': self.model.user_factors,
            'item_factors': self.model.item_factors
        }
        np.savez(model_path, **model_data)

        # Save mappings
        mappings = {
            'user_to_idx': self.user_to_idx,
            'idx_to_user': self.idx_to_user,
            'item_to_idx': self.item_to_idx,
            'idx_to_item': self.idx_to_item,
        }
        with open(mappings_path, 'wb') as f:
            pickle.dump(mappings, f)

        # Save sparse matrix
        sparse.save_npz(matrix_path, self.user_item_matrix)

        print(f"Model saved to {model_path}")
        print(f"Mappings saved to {mappings_path}")
        print(f"Matrix saved to {matrix_path}")

    def load_model(self, model_path: Path, mappings_path: Path, matrix_path: Path):
        """
        Load a trained model and mappings.

        Args:
            model_path: Path to model parameters
            mappings_path: Path to ID mappings
            matrix_path: Path to sparse matrix
        """
        # Load model parameters
        model_data = np.load(model_path)

        # Recreate model
        self.model = AlternatingLeastSquares(
            factors=self.config.get('factors', 128),
            regularization=self.config.get('regularization', 1.0),
            iterations=self.config.get('iterations', 100),
            alpha=self.config.get('alpha', 10),
            use_gpu=self.config.get('use_gpu', False),
        )

        # Set the factors directly
        self.model.user_factors = model_data['user_factors']
        self.model.item_factors = model_data['item_factors']

        # Load mappings
        with open(mappings_path, 'rb') as f:
            mappings = pickle.load(f)

        self.user_to_idx = mappings['user_to_idx']
        self.idx_to_user = mappings['idx_to_user']
        self.item_to_idx = mappings['item_to_idx']
        self.idx_to_item = mappings['idx_to_item']

        # Load sparse matrix
        self.user_item_matrix = sparse.load_npz(matrix_path)

        print(f"Model loaded from {model_path}")
        print(f"Matrix shape: {self.user_item_matrix.shape}")

    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        if self.model is None:
            return {"status": "not_trained"}

        return {
            "status": "trained",
            "n_users": len(self.user_to_idx),
            "n_items": len(self.item_to_idx),
            "n_factors": self.config.get('factors', 128),
            "matrix_shape": self.user_item_matrix.shape if self.user_item_matrix is not None else None
        }
