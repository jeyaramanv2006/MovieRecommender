// Movie Recommendation System - Frontend JavaScript
// Simple & Clean Implementation

const API_BASE = 'http://localhost:8000/api';
const TMDB_IMAGE_BASE = 'https://image.tmdb.org/t/p/w500';

// State
let selectedMovieId = null;

// DOM Elements
const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const searchResults = document.getElementById('searchResults');
const movieSection = document.getElementById('movieSection');
const selectedMovie = document.getElementById('selectedMovie');
const getRecommendationsBtn = document.getElementById('getRecommendationsBtn');
const recommendationsSection = document.getElementById('recommendationsSection');
const recommendations = document.getElementById('recommendations');
const algorithmUsed = document.getElementById('algorithmUsed');
const loadingOverlay = document.getElementById('loadingOverlay');

// Event Listeners
searchBtn.addEventListener('click', searchMovies);
searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') searchMovies();
});
getRecommendationsBtn.addEventListener('click', getRecommendations);

// Search Movies
async function searchMovies() {
    const query = searchInput.value.trim();
    if (!query) return;
    
    showLoading();
    try {
        const response = await fetch(`${API_BASE}/search?query=${encodeURIComponent(query)}`);
        const data = await response.json();
        
        displaySearchResults(data.results);
    } catch (error) {
        console.error('Search error:', error);
        alert('Failed to search movies. Please try again.');
    } finally {
        hideLoading();
    }
}

// Display Search Results
function displaySearchResults(results) {
    if (results.length === 0) {
        searchResults.innerHTML = '<p style="text-align: center; color: #888;">No movies found. Try a different search.</p>';
        searchResults.classList.remove('hidden');
        return;
    }
    
    searchResults.innerHTML = results.map(movie => `
        <div class="search-result-item" onclick="selectMovie(${movie.id})">
            ${movie.poster_path 
                ? `<img src="${TMDB_IMAGE_BASE}${movie.poster_path}" alt="${movie.title}" class="result-poster">`
                : `<div class="result-poster placeholder">🎬</div>`
            }
            <div class="result-info">
                <div class="result-title">${movie.title}</div>
                <div class="result-year">${movie.release_date ? movie.release_date.split('-')[0] : 'N/A'}</div>
            </div>
        </div>
    `).join('');
    
    searchResults.classList.remove('hidden');
}

// Select Movie
async function selectMovie(movieId) {
    selectedMovieId = movieId;
    
    showLoading();
    try {
        const response = await fetch(`${API_BASE}/movie/${movieId}`);
        const movie = await response.json();
        
        displaySelectedMovie(movie);
        
        // Hide recommendations section when new movie is selected
        recommendationsSection.classList.add('hidden');
        
        // Scroll to selected movie
        movieSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
    } catch (error) {
        console.error('Movie fetch error:', error);
        alert('Failed to load movie details. Please try again.');
    } finally {
        hideLoading();
    }
}

// Display Selected Movie
function displaySelectedMovie(movie) {
    const runtime = movie.runtime ? `${movie.runtime} min` : 'N/A';
    const genres = movie.genres?.map(g => g.name).join(', ') || 'N/A';
    
    selectedMovie.innerHTML = `
        ${movie.poster_path 
            ? `<img src="${TMDB_IMAGE_BASE}${movie.poster_path}" alt="${movie.title}" class="movie-poster">`
            : `<div class="movie-poster placeholder">🎬</div>`
        }
        <div class="movie-info">
            <h3 class="movie-title">${movie.title}</h3>
            <div class="movie-meta">
                <span class="meta-item">
                    <span>📅</span>
                    ${movie.release_date ? movie.release_date.split('-')[0] : 'N/A'}
                </span>
                <span class="meta-item">
                    <span>⭐</span>
                    ${movie.vote_average ? movie.vote_average.toFixed(1) : 'N/A'}/10
                </span>
                <span class="meta-item">
                    <span>⏱️</span>
                    ${runtime}
                </span>
            </div>
            <div class="movie-meta">
                <span class="meta-item">
                    <span>🎭</span>
                    ${genres}
                </span>
            </div>
            <p class="movie-overview">${movie.overview || 'No overview available.'}</p>
        </div>
    `;
    
    movieSection.classList.remove('hidden');
    getRecommendationsBtn.classList.remove('hidden');
}

// Get Recommendations
async function getRecommendations() {
    if (!selectedMovieId) return;
    
    showLoading();
    try {
        const response = await fetch(`${API_BASE}/recommend/${selectedMovieId}?limit=12`);
        const data = await response.json();
        
        displayRecommendations(data.results, data.algorithm);
        
        // Scroll to recommendations
        recommendationsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (error) {
        console.error('Recommendations error:', error);
        alert('Failed to get recommendations. Please try again.');
    } finally {
        hideLoading();
    }
}

// Display Recommendations
function displayRecommendations(movies, algorithm) {
    if (movies.length === 0) {
        recommendations.innerHTML = '<p style="text-align: center; color: #888;">No recommendations found.</p>';
        recommendationsSection.classList.remove('hidden');
        return;
    }
    
    // Update algorithm badge
    const algorithmText = algorithm === 'Jakube' 
        ? 'Jakube Algorithm (ANN Search)' 
        : 'TMDB Fallback (Collaborative Filtering)';
    algorithmUsed.textContent = algorithmText;
    
    recommendations.innerHTML = movies.map(movie => `
        <div class="rec-card" onclick="selectMovie(${movie.id})">
            ${movie.poster_path 
                ? `<img src="${TMDB_IMAGE_BASE}${movie.poster_path}" alt="${movie.title}" class="rec-poster">`
                : `<div class="rec-poster placeholder">🎬</div>`
            }
            <div class="rec-info">
                <div class="rec-title">${movie.title}</div>
                <div class="rec-details">
                    <span class="rec-year">${movie.release_date ? movie.release_date.split('-')[0] : 'N/A'}</span>
                    <span class="rec-rating">
                        ⭐ ${movie.vote_average ? movie.vote_average.toFixed(1) : 'N/A'}
                    </span>
                </div>
            </div>
        </div>
    `).join('');
    
    recommendationsSection.classList.remove('hidden');
}

// Loading Overlay
function showLoading() {
    loadingOverlay.classList.remove('hidden');
}

function hideLoading() {
    loadingOverlay.classList.add('hidden');
}

// Initial Message
console.log(`
🎬 Movie Recommendation System
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Algorithm: Jakube (Approximate Nearest Neighbor)
DSA Concepts:
  • Random Projection Trees
  • Binary Space Partitioning  
  • Priority Queue (k-NN)
  • High-dimensional Vector Search
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
`);
