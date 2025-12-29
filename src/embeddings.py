"""Embedding module: Generate vectors using sentence-transformers"""
from sentence_transformers import SentenceTransformer
from typing import List
import os
import numpy as np


class EmbeddingModel:
    """Embedding model wrapper"""

    def __init__(self):
        """
        Initialize embedding model

        Args:
            model_name: Model name, defaults to multilingual model
        """
        model_name = os.getenv('EMBEDDING_MODEL', 'paraphrase-multilingual-MiniLM-L12-v2')
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"✓ Model loaded, vector dimension: {self.dimension}")

    def encode(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Convert texts to vectors

        Args:
            texts: List of texts
            show_progress: Whether to show progress bar

        Returns:
            Array of vectors
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """
        Convert single text to vector

        Args:
            text: Single text

        Returns:
            Vector
        """
        return self.model.encode(text, convert_to_numpy=True)

    def get_dimension(self) -> int:
        """Get vector dimension"""
        return self.dimension


if __name__ == '__main__':
    # Test embedding model
    model = EmbeddingModel()

    test_texts = [
        "The boy who cried wolf learned a valuable lesson about honesty.",
        "Slow and steady wins the race."
    ]

    embeddings = model.encode(test_texts, show_progress=False)
    print(f"\nTest results:")
    print(f"  Input texts: {len(test_texts)}")
    print(f"  Output vector shape: {embeddings.shape}")
    print(f"  Vector dimension: {model.get_dimension()}")
