"""
Movie vectorization utilities for TMDB data using pandas and scikit-learn
(Enhanced Version - Genre + Popularity + Language Features)
"""
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd

# ============================================================================
# GENRE CONFIGURATION
# ============================================================================
# This list must be consistent with the one in main.py
# TMDB Genre IDs as of late 2025
GENRE_LIST = [
    {"id": 28, "name": "Action"},
    {"id": 12, "name": "Adventure"},
    {"id": 16, "name": "Animation"},
    {"id": 35, "name": "Comedy"},
    {"id": 80, "name": "Crime"},
    {"id": 99, "name": "Documentary"},
    {"id": 18, "name": "Drama"},
    {"id": 10751, "name": "Family"},
    {"id": 14, "name": "Fantasy"},
    {"id": 36, "name": "History"},
    {"id": 27, "name": "Horror"},
    {"id": 10402, "name": "Music"},
    {"id": 9648, "name": "Mystery"},
    {"id": 10749, "name": "Romance"},
    {"id": 878, "name": "Science Fiction"},
    {"id": 10770, "name": "TV Movie"},
    {"id": 53, "name": "Thriller"},
    {"id": 10752, "name": "War"},
    {"id": 37, "name": "Western"}
]

# Create a mapping from genre ID -> vector index
GENRE_ID_TO_INDEX: Dict[int, int] = {genre['id']: i for i, genre in enumerate(GENRE_LIST)}
GENRE_DIMENSION: int = len(GENRE_LIST)  # Should be 19

# Additional feature dimensions
# is_popular (1 dimension), is_tamil, is_malayalam, is_hindi, is_english (4 dimensions)
ADDITIONAL_DIMENSIONS: int = 5

# Total dimensions = 19 genres + 1 popularity + 4 languages = 24
TOTAL_DIMENSION: int = GENRE_DIMENSION + ADDITIONAL_DIMENSIONS

# ============================================================================

def movies_to_matrix(items: Iterable[Dict]) -> Tuple[np.ndarray, List[Dict]]:
    """
    Convert TMDB movie documents to a multi-hot encoded feature matrix
    for use with Hamming distance.
    
    Feature vector structure:
    - First 19 dimensions: Genre (multi-hot encoding)
    - Dimension 20: is_popular (1 if vote_average >= 8.0, else 0)
    - Dimension 21: is_tamil (1 if original_language == 'ta', else 0)
    - Dimension 22: is_malayalam (1 if original_language == 'ml', else 0)
    - Dimension 23: is_hindi (1 if original_language == 'hi', else 0)
    - Dimension 24: is_english (1 if original_language == 'en', else 0)
    """
    all_items = list(items)
    if not all_items:
        raise RuntimeError("No TMDB items to process.")

    # Use a list to build vectors first
    vectors = []
    retained_items = []

    for movie in all_items:
        # Create a zero-vector for this movie
        # We use np.int32 because jakube/annoy Hamming distance works on integers
        vector = np.zeros(TOTAL_DIMENSION, dtype=np.int32)

        # ========================================
        # STEP 1: Process genre information
        # ========================================
        genre_ids = movie.get("genre_ids")
        
        # Only process movies that have genre information
        if not genre_ids:
            continue

        found_genres = 0
        for gid in genre_ids:
            # Find the index for this genre ID
            idx = GENRE_ID_TO_INDEX.get(gid)
            if idx is not None:
                # Set the '1' in our multi-hot vector
                vector[idx] = 1
                found_genres += 1

        # Only include movies that have at least one known genre
        if found_genres == 0:
            continue

        # ========================================
        # STEP 2: Process popularity feature
        # ========================================
        # is_popular: 1 if vote_average >= 8.0, else 0
        vote_average = movie.get("vote_average", 0)
        if vote_average >= 8.0:
            vector[GENRE_DIMENSION] = 1  # Index 19
        else:
            vector[GENRE_DIMENSION] = 0

        # ========================================
        # STEP 3: Process language features
        # ========================================
        original_language = movie.get("original_language", "")
        
        # is_tamil (index 20)
        if original_language == "ta":
            vector[GENRE_DIMENSION + 1] = 1
        
        # is_malayalam (index 21)
        if original_language == "ml":
            vector[GENRE_DIMENSION + 2] = 1
        
        # is_hindi (index 22)
        if original_language == "hi":
            vector[GENRE_DIMENSION + 3] = 1
        
        # is_english (index 23)
        if original_language == "en":
            vector[GENRE_DIMENSION + 4] = 1

        # ========================================
        # Add to results
        # ========================================
        vectors.append(vector)
        retained_items.append(movie)

    if not vectors:
        raise RuntimeError("No TMDB items contained usable genre features.")

    # Create the final feature matrix
    matrix = np.array(vectors, dtype=np.int32)
    
    return matrix, retained_items


def build_index(matrix: np.ndarray, metric: str = 'hamming', n_trees: int = 20, n_jobs: int = 4) -> 'JakubeIndexType':
    """Build a Jakube index from a feature matrix."""
    from jakube import (
        AngularIndex,
        DotProductIndex,
        EuclideanIndex,
        ManhattanIndex,
        HammingIndex
    )

    METRIC_MAP = {
        'angular': AngularIndex,
        'euclidean': EuclideanIndex,
        'manhattan': ManhattanIndex,
        'dot': DotProductIndex,
        'dotproduct': DotProductIndex,
        'hamming': HammingIndex  # Hamming is what we want
    }

    # Default to HammingIndex if metric is unrecognized
    IndexClass = METRIC_MAP.get(metric.lower(), HammingIndex)
    dims = matrix.shape[1]

    if IndexClass != HammingIndex and metric.lower() == 'hamming':
        print(f"Warning: Hamming metric specified but HammingIndex class not found. Falling back.")
    elif IndexClass == HammingIndex:
        print(f"Using HammingIndex for {dims} dimensions (19 genres + 1 popularity + 4 languages).")

    index = IndexClass(dims)

    # Simple loop to add items
    # The expensive part is index.build(), which is already multi-threaded
    for idx, row in enumerate(matrix):
        # Add item requires a list of integers for Hamming
        index.add_item(idx, row.tolist())

    # Build the trees using multiple threads (n_jobs)
    index.build(q=n_trees, n_threads=n_jobs)

    return index
