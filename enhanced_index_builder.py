"""
Enhanced Movie Index Builder - Builds a comprehensive TMDB movie database
(Enhanced Version with Popularity and Language Features)
"""
import argparse
import asyncio
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set
import logging

import httpx
import numpy as np
from tqdm.asyncio import tqdm
from aiolimiter import AsyncLimiter

from movie_vectorizer import movies_to_matrix, build_index

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('index_builder.log')
    ]
)

logger = logging.getLogger(__name__)

TMDB_API_ROOT = "https://api.themoviedb.org/3"
BATCH_SIZE = 10  # Number of concurrent requests
RATE_LIMIT = 40  # Requests per 10 seconds
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5  # seconds

class TMDBCrawler:
    """Manages crawling the TMDB API with rate limiting and retries"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.rate_limiter = AsyncLimiter(RATE_LIMIT, 10.0)
        self.client: Optional[httpx.AsyncClient] = None
        self.movie_cache: Dict[int, Dict] = {}
        self.cache_file = Path("movie_cache.json")
        self._load_cache()

    def _load_cache(self):
        """Load previously cached movies"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                    self.movie_cache = {int(k): v for k, v in cache.items()}
                logger.info(f"Loaded {len(self.movie_cache)} movies from cache")
            except Exception as e:
                logger.error(f"Error loading cache: {e}")

    def _save_cache(self):
        """Save cached movies to disk"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.movie_cache, f)
            logger.info(f"Saved {len(self.movie_cache)} movies to cache")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    async def __aenter__(self):
        transport = httpx.AsyncHTTPTransport(retries=RETRY_ATTEMPTS)
        self.client = httpx.AsyncClient(timeout=30.0, transport=transport)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
        self._save_cache()

    async def fetch_page(
        self,
        endpoint: str,
        page: int,
        extra_params: Optional[Dict] = None
    ) -> Dict:
        """Fetch a single page with retries and rate limiting"""
        if not self.client:
            raise RuntimeError("Client not initialized")

        params = {
            "api_key": self.api_key,
            "language": "en-US",
            "page": page
        }
        if extra_params:
            params.update(extra_params)

        url = f"{TMDB_API_ROOT}/{endpoint}"
        
        async with self.rate_limiter:
            try:
                response = await self.client.get(url, params=params)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error fetching {e.request.url}: {e}")
                raise
            except httpx.RequestError as e:
                logger.error(f"Request error fetching {e.request.url}: {e}")
                raise

    async def fetch_year(
        self,
        year: int,
        max_pages: int = 500,
        progress_bar: Optional[tqdm] = None
    ) -> List[Dict]:
        """Fetch all movies for a specific year"""
        if not self.client:
            raise RuntimeError("Client not initialized")

        params = {
            "sort_by": "popularity.desc",
            "include_adult": False,
            "include_video": False,
            "primary_release_year": year,
        }

        dedup: OrderedDict[int, Dict] = OrderedDict()
        total_pages = None

        for batch_start in range(1, max_pages + 1, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, max_pages + 1)
            tasks = []

            for page in range(batch_start, batch_end):
                task = self.fetch_page("discover/movie", page, params)
                tasks.append(task)

            try:
                results = await asyncio.gather(*tasks)
                
                for result in results:
                    if total_pages is None:
                        total_pages = min(result.get("total_pages", max_pages), max_pages)
                    
                    for movie in result.get("results", []):
                        movie_id = movie.get("id")
                        if movie_id:
                            # Store movie with genre_ids, vote_average, and original_language
                            if movie_id not in self.movie_cache:
                                self.movie_cache[movie_id] = movie
                            dedup[movie_id] = movie

                if progress_bar:
                    progress_bar.update(len(tasks))

                if total_pages and batch_start >= total_pages:
                    break

            except Exception as e:
                logger.error(f"Error fetching batch {batch_start}-{batch_end}: {e}")
                continue

        return list(dedup.values())


async def build_comprehensive_index(
    api_key: str,
    start_year: int = 2005,
    end_year: int = 2025,
    max_pages_per_year: int = 500,
    n_trees: int = 20,
    metric: str = "hamming",
    n_jobs: int = 4,
    output_dir: str = "."
) -> None:
    """Build a comprehensive movie index from TMDB data"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_movies = []

    async with TMDBCrawler(api_key) as crawler:
        total_pages = (end_year - start_year + 1) * max_pages_per_year
        
        with tqdm(total=total_pages, desc="Fetching movies") as pbar:
            for year in range(start_year, end_year + 1):
                logger.info(f"\nProcessing year {year}...")
                
                try:
                    year_movies = await crawler.fetch_year(
                        year,
                        max_pages=max_pages_per_year,
                        progress_bar=pbar
                    )
                    total_movies.extend(year_movies)
                    logger.info(f"Found {len(year_movies)} movies from {year}")
                
                except Exception as e:
                    logger.error(f"Error processing year {year}: {e}")

    # Convert to feature matrix
    logger.info(f"\nConverting {len(total_movies)} movies to feature vectors (genres + popularity + languages)...")
    matrix, retained = movies_to_matrix(total_movies)

    # Build index
    logger.info(f"Building {metric} index with {n_trees} trees using {n_jobs} parallel jobs...")
    index = build_index(matrix, metric=metric, n_trees=n_trees, n_jobs=n_jobs)

    # Save everything
    index_path = output_dir / "movie_index.jakube"
    map_path = output_dir / "movie_id_map.json"
    metadata_path = output_dir / "movie_metadata.json"

    logger.info(f"Saving index to {index_path}...")
    index.save(str(index_path))

    logger.info(f"Saving ID mappings to {map_path}...")
    tmdb_to_index = {str(movie["id"]): idx for idx, movie in enumerate(retained)}
    index_to_tmdb = {str(idx): movie["id"] for idx, movie in enumerate(retained)}

    with open(map_path, "w") as f:
        json.dump({
            "tmdb_to_index": tmdb_to_index,
            "index_to_tmdb": index_to_tmdb,
        }, f, indent=2)

    logger.info(f"Saving movie metadata to {metadata_path}...")
    with open(metadata_path, "w") as f:
        json.dump(retained, f, indent=2)

    logger.info(f"\nSuccess! Built index with {index.n_items()} movies")
    logger.info(f"Files created:")
    logger.info(f"  • {index_path}")
    logger.info(f"  • {map_path}")
    logger.info(f"  • {metadata_path}")


async def main() -> None:
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Build a comprehensive Jakube movie recommendation index from TMDB"
    )
    
    parser.add_argument(
        "--start-year",
        type=int,
        default=2005,
        help="Start year for movie collection (default: 2005)",
    )
    
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="End year for movie collection (default: 2025)",
    )
    
    parser.add_argument(
        "--max-pages",
        type=int,
        default=500,
        help="Max pages per year (default: 500)",
    )
    
    parser.add_argument(
        "--trees",
        type=int,
        default=20,
        help="Trees in the Jakube forest (default: 20)",
    )
    
    parser.add_argument(
        "--metric",
        type=str,
        default="hamming",
        choices=["angular", "euclidean", "manhattan", "dot", "hamming"],
        help="Distance metric (default: hamming)",
    )
    
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Parallel jobs for index building (default: 4)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory (default: current)",
    )

    args = parser.parse_args()

    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        logger.error("Error: TMDB_API_KEY environment variable not set")
        sys.exit(1)

    try:
        await build_comprehensive_index(
            api_key=api_key,
            start_year=args.start_year,
            end_year=args.end_year,
            max_pages_per_year=args.max_pages,
            n_trees=args.trees,
            metric=args.metric,
            n_jobs=args.jobs,
            output_dir=args.output_dir
        )
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
