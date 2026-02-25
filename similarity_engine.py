# similarity_engine.py

import numpy as np
from typing import Dict, List, Tuple


class SimilarityEngine:
    """
    Responsible ONLY for computing similarity
    between query embedding and stored case embeddings.
    """

    def __init__(self, case_embeddings: Dict[str, np.ndarray]):
        """
        Parameters:
        -----------
        case_embeddings : dict
            Dictionary of {case_id: embedding_vector}
        """
        self.case_embeddings = case_embeddings

    # ---------------------------------------------------
    # Public Method
    # ---------------------------------------------------

    def retrieve_top_k(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Compute similarity between query embedding and
        all stored case embeddings.

        Returns:
        --------
        List of (case_id, similarity_score) sorted descending.
        """

        similarities = []

        for case_id, embedding in self.case_embeddings.items():
            score = self._cosine_similarity(query_embedding, embedding)
            similarities.append((case_id, float(score)))

        # Sort in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    # ---------------------------------------------------
    # Private Methods
    # ---------------------------------------------------

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        """

        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0

        return np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2)
        )