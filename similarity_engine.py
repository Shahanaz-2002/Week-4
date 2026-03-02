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
        Compute cosine similarity between query embedding
        and all stored case embeddings.

        Returns:
        --------
        List of (case_id, similarity_score) sorted descending.
        """

        if not self.case_embeddings:
            return []

        similarities = []

        # Precompute query norm once
        query_norm = np.linalg.norm(query_embedding)

        if query_norm == 0:
            return []

        for case_id, embedding in self.case_embeddings.items():

            emb_norm = np.linalg.norm(embedding)

            if emb_norm == 0:
                score = 0.0
            else:
                score = float(
                    np.dot(query_embedding, embedding)
                    / (query_norm * emb_norm)
                )

            similarities.append((case_id, score))

        # Sort descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Ensure top_k does not exceed available cases
        top_k = min(top_k, len(similarities))

        return similarities[:top_k]