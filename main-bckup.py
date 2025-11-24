from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Union
import httpx
import random
import json
import os
from pathlib import Path

# Frontend directory path
FRONTEND_DIR = Path(__file__).parent

from jakube import (
    AngularIndex,
    DotProductIndex,
    EuclideanIndex,
    ManhattanIndex,
    HammingIndex
)

# ============================================================================
# CONFIGURATION
# ============================================================================
PORT = 8000
TMDB_API_KEY = "efa4bba6280252ded1c68c4884f56085"
if not TMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY environment variable must be set")

TMDB_BASE = "https://api.themoviedb.org/3"
BASE_DIR = Path(__file__).resolve().parent

# ============================================================================
# GENRE CONFIGURATION
# ============================================================================
# This list must be consistent with the one in movie_vectorizer.py
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

GENRE_DIMENSION: int = len(GENRE_LIST)  # Should be 19

# Additional feature dimensions
# is_popular (1 dimension), is_tamil, is_malayalam, is_hindi, is_english (4 dimensions)
ADDITIONAL_DIMENSIONS: int = 5

# Total dimensions = 19 genres + 1 popularity + 4 languages = 24
EMBEDDING_DIMENSION = GENRE_DIMENSION + ADDITIONAL_DIMENSIONS  # Now 24

# ============================================================================
INDEX_FILE = BASE_DIR / "movie_index.jakube"
MAP_FILE = BASE_DIR / "movie_id_map.json"
METRIC = 'hamming'

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JakubeIndexType = Union[AngularIndex, DotProductIndex, EuclideanIndex, ManhattanIndex, HammingIndex]

jakube_index: JakubeIndexType
tmdb_to_index: Dict[str, int] = {}
index_to_tmdb: Dict[str, int] = {}
index_loaded: bool = False

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/api/search")
async def search_movies(query: str = Query(..., min_length=1)):
    """Search for movies by title"""
    async with httpx.AsyncClient() as client:
        url = f"{TMDB_BASE}/search/movie"
        params = {
            "api_key": TMDB_API_KEY,
            "language": "en-US",
            "query": query,
            "page": 1
        }
        
        response = await client.get(url, params=params)
        data = response.json()
        results = data.get("results", [])[:10]
        
        return {"results": results}


@app.get("/api/movie/{movie_id}")
async def get_movie(movie_id: int):
    """Get movie details"""
    async with httpx.AsyncClient() as client:
        url = f"{TMDB_BASE}/movie/{movie_id}"
        params = {
            "api_key": TMDB_API_KEY,
            "language": "en-US"
        }
        
        response = await client.get(url, params=params)
        return response.json()


@app.get("/api/recommend/{movie_id}")
async def get_recommendations(movie_id: int, limit: int = Query(12, ge=1, le=20)):
    global jakube_index, tmdb_to_index, index_to_tmdb, index_loaded
    
    if not index_loaded:
        raise HTTPException(
            status_code=503,
            detail="Jakube recommender index not loaded"
        )
    
    movie_id_str = str(movie_id)
    if movie_id_str not in tmdb_to_index:
        raise HTTPException(
            status_code=404,
            detail="Movie not found in Jakube index"
        )
    
    try:
        internal_index = tmdb_to_index[movie_id_str]
        print(f"Internal index for movie {movie_id}: {internal_index}")
        
        # STEP 1: Get the source movie's details
        async with httpx.AsyncClient() as client:
            url = f"{TMDB_BASE}/movie/{movie_id}"
            params = {"api_key": TMDB_API_KEY, "language": "en-US"}
            response = await client.get(url, params=params)
            source_movie = response.json()
            source_rating = source_movie.get("vote_average", 0)
        
        # STEP 2: Determine if we need rating filtering
        filter_by_rating = source_rating >= 6.0
        
        # STEP 3: Request MORE neighbors to account for filtering
        # Start with 5x to have a better chance of finding enough high-rated movies
        if filter_by_rating:
            search_limit = limit * 5
            print(f"⚡ Rating filter active (source: {source_rating:.1f}): requesting {search_limit} neighbors")
        else:
            search_limit = limit + 1
            print(f"ℹ️  No rating filter: requesting {search_limit} neighbors")
        
        try:
            # Find items with the smallest Hamming distance
            neighbor_indices, distances = jakube_index.get_nns_by_item(
                int(internal_index),
                search_limit
            )
            print(f"Found {len(neighbor_indices)} neighbors")
        except Exception as e:
            print(f"Error querying Jakube: {str(e)}")
            raise
        
        # STEP 4: Convert internal indices to TMDB IDs
        similar_ids = []
        for idx in neighbor_indices:
            if idx == int(internal_index):
                continue  # Skip the movie itself
            
            tmdb_id = index_to_tmdb.get(str(idx))
            if tmdb_id:
                similar_ids.append(int(tmdb_id))
        
        print(f"Similar IDs before filtering: {len(similar_ids)} movies")
        
        # STEP 5: Fetch ALL movie details first
        all_candidates = []
        async with httpx.AsyncClient() as client:
            for mid in similar_ids:
                try:
                    url = f"{TMDB_BASE}/movie/{mid}"
                    params = {"api_key": TMDB_API_KEY, "language": "en-US"}
                    response = await client.get(url, params=params)
                    
                    if response.status_code == 200:
                        movie_data = response.json()
                        all_candidates.append(movie_data)
                            
                except Exception as e:
                    print(f"Error fetching movie {mid}: {str(e)}")
                    continue
        
        print(f"Fetched {len(all_candidates)} candidate movies")
        
        # STEP 6: Apply intelligent filtering with fallback logic
        recommendations = []
        
        if filter_by_rating:
            # First, try to get movies with rating >= 6.0
            high_rated = [m for m in all_candidates if m.get("vote_average", 0) >= 6.0]
            print(f"Found {len(high_rated)} movies with rating >= 6.0")
            
            if len(high_rated) >= limit:
                # We have enough high-rated movies
                recommendations = high_rated[:limit]
                print(f"✅ Using {len(recommendations)} high-rated movies (rating >= 6.0)")
                
            else:
                # Not enough high-rated movies, so we'll use fallback strategy
                print(f"⚠️  Only {len(high_rated)} high-rated movies found (need {limit})")
                print(f"📊 Applying fallback: sorting remaining by popularity")
                
                # Take all high-rated movies first
                recommendations = high_rated.copy()
                
                # Get remaining movies (rating < 6.0) and sort by popularity
                remaining = [m for m in all_candidates if m.get("vote_average", 0) < 6.0]
                
                # Sort by vote_average first, then by popularity as tiebreaker
                remaining_sorted = sorted(
                    remaining,
                    key=lambda m: (m.get("vote_average", 0), m.get("popularity", 0)),
                    reverse=True
                )
                
                # Fill up to limit with the best remaining movies
                needed = limit - len(recommendations)
                recommendations.extend(remaining_sorted[:needed])
                
                print(f"✅ Final mix: {len(high_rated)} high-rated + {len(recommendations) - len(high_rated)} best remaining")
                
                # Log the rating breakdown
                for i, movie in enumerate(recommendations):
                    rating = movie.get("vote_average", 0)
                    popularity = movie.get("popularity", 0)
                    title = movie.get("title", "Unknown")
                    if rating >= 6.0:
                        print(f"  {i+1}. ✓ {title} (⭐{rating:.1f}, 🔥{popularity:.0f})")
                    else:
                        print(f"  {i+1}. ↓ {title} (⭐{rating:.1f}, 🔥{popularity:.0f})")
        else:
            # No rating filter, just return top matches
            recommendations = all_candidates[:limit]
            print(f"✅ Returning {len(recommendations)} recommendations (no filter)")
        
        return {
            "results": recommendations,
            "algorithm": "jakube_enhanced_with_smart_fallback",
            "source_movie_id": movie_id,
            "source_movie_rating": source_rating,
            "rating_filter_applied": filter_by_rating,
            "high_rated_count": len([m for m in recommendations if m.get("vote_average", 0) >= 6.0]),
            "total_returned": len(recommendations)
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting recommendations: {str(e)}"
        )



@app.get("/")
async def serve_homepage():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/style.css")
async def serve_css():
    return FileResponse(FRONTEND_DIR / "style.css")


@app.get("/app.js")
async def serve_js():
    return FileResponse(FRONTEND_DIR / "app.js")


@app.on_event("startup")
async def startup_event():
    global jakube_index, tmdb_to_index, index_to_tmdb, index_loaded
    
    print("\nInitializing Enhanced Jakube Movie Recommender (Genre + Popularity + Language)...")
    
    # Initialize the correct index type
    METRIC_MAP = {
        'angular': AngularIndex,
        'euclidean': EuclideanIndex,
        'manhattan': ManhattanIndex,
        'dot': DotProductIndex,
        'dotproduct': DotProductIndex,
        'hamming': HammingIndex
    }
    
    IndexClass = METRIC_MAP.get(METRIC.lower(), HammingIndex)
    jakube_index = IndexClass(EMBEDDING_DIMENSION)  # Uses 24 dimensions
    
    if not INDEX_FILE.exists() or not MAP_FILE.exists():
        print(f"Error: Required files not found:")
        print(f"  - Index file: {INDEX_FILE} ({'exists' if INDEX_FILE.exists() else 'missing'})")
        print(f"  - Map file: {MAP_FILE} ({'exists' if MAP_FILE.exists() else 'missing'})")
        print("\nPlease run enhanced_index_builder.py first:")
        print("  python enhanced_index_builder.py --metric hamming --trees 20")
        return
    
    try:
        print(f"\nLoading Jakube index from {INDEX_FILE}...")
        jakube_index.load(str(INDEX_FILE))
        
        n_items = jakube_index.n_items()
        n_trees = jakube_index.n_trees()
        print(f"✓ Index loaded: {n_items} items across {n_trees} trees")
        
        print(f"\nLoading movie ID mappings from {MAP_FILE}...")
        with open(MAP_FILE, 'r') as f:
            maps = json.load(f)
            tmdb_to_index = maps['tmdb_to_index']
            index_to_tmdb = maps['index_to_tmdb']
        
        print(f"✓ Mappings loaded: {len(tmdb_to_index)} movies indexed")
        
        index_loaded = True
        
        print("\nEnhanced Jakube Recommender loaded successfully! 🎬")
        print(f"\n{'='*70}")
        print(f"🎬 Movie Recommendation System (Genre + Popularity + Language)")
        print(f"{'='*70}")
        print(f"🌐 Server: http://localhost:{PORT}")
        print(f"📊 API Docs: http://localhost:{PORT}/docs")
        print(f"📈 Index Status: {f'LOADED ({n_items} movies)' if index_loaded else 'NOT LOADED'}")
        print(f"📐 Metric: {METRIC.capitalize()} ({EMBEDDING_DIMENSION} dims)")
        print(f"   - 19 Genre dimensions")
        print(f"   - 1 Popularity dimension (rating >= 6.0)")
        print(f"   - 4 Language dimensions (Tamil, Malayalam, Hindi, English)")
        print(f"{'='*70}\n")
    
    except Exception as e:
        print(f"Error loading recommender: {str(e)}")
        index_loaded = False


@app.get("/health")
async def health():
    global index_loaded
    return {
        "status": "healthy",
        "algorithm": "jakube_enhanced_hamming_genre_popularity_language",
        "index_loaded": index_loaded
    }


# ============================================================================
# RUN
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
