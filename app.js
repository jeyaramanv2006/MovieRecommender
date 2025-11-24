// Movie Recommendation System - Enhanced Frontend
const API_BASE = 'http://localhost:8000/api';
const TMDB_IMAGE_BASE = 'https://image.tmdb.org/t/p/w500';

// State
let selectedMovieId = null;

// DOM Elements
const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const searchSection = document.getElementById('searchSection');
const searchResults = document.getElementById('searchResults');
const movieSection = document.getElementById('movieSection');
const selectedMovie = document.getElementById('selectedMovie');
const getRecommendationsBtn = document.getElementById('getRecommendationsBtn');
const recommendationsSection = document.getElementById('recommendationsSection');
const recommendations = document.getElementById('recommendations');
const algorithmUsed = document.getElementById('algorithmUsed');
const loadingOverlay = document.getElementById('loadingOverlay');
const newSearchBtn = document.getElementById('newSearchBtn');

// Event Listeners
searchBtn.addEventListener('click', searchMovies);
searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') searchMovies();
});
getRecommendationsBtn.addEventListener('click', getRecommendations);
newSearchBtn.addEventListener('click', resetSearch);

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
        showToast('Failed to search movies. Please try again.', 'error');
    } finally {
        hideLoading();
    }
}

// Display Search Results
function displaySearchResults(results) {
    searchResults.innerHTML = '';
    
    if (results.length === 0) {
        searchResults.innerHTML = `
            <div class="glass-card" style="padding: 2rem; text-align: center;">
                <p style="color: var(--text-muted); font-size: 1.125rem;">
                    No movies found. Try a different search.
                </p>
            </div>
        `;
        searchResults.classList.remove('hidden');
        return;
    }
    
    const resultsHTML = `
        <div class="glass-card">
            <h2 class="section-title">Search Results</h2>
            ${results.map(movie => `
                <div class="search-result-item" onclick="selectMovie(${movie.id})">
                    ${movie.poster_path 
                        ? `<img src="${TMDB_IMAGE_BASE}${movie.poster_path}" alt="${movie.title}" class="result-poster">`
                        : `<div class="result-poster placeholder">🎬</div>`
                    }
                    <div class="result-info">
                        <div class="result-title">${movie.title}</div>
                        <div class="result-year">${movie.release_date ? movie.release_date.split('-')[0] : 'N/A'}</div>
                        ${movie.vote_average ? `<div class="result-rating">⭐ ${movie.vote_average.toFixed(1)}</div>` : ''}
                    </div>
                </div>
            `).join('')}
        </div>
    `;
    
    searchResults.innerHTML = resultsHTML;
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
        
        // Hide other sections
        searchSection.classList.add('hidden');
        recommendationsSection.classList.add('hidden');
        
        // Show movie section
        movieSection.classList.remove('hidden');
        movieSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
    } catch (error) {
        console.error('Movie fetch error:', error);
        showToast('Failed to load movie details. Please try again.', 'error');
    } finally {
        hideLoading();
    }
}

// Display Selected Movie
function displaySelectedMovie(movie) {
    const year = movie.release_date ? movie.release_date.split('-')[0] : 'N/A';
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
                <span class="meta-item">📅 ${year}</span>
                ${movie.vote_average ? `<span class="meta-item">⭐ ${movie.vote_average.toFixed(1)}/10</span>` : ''}
                <span class="meta-item">⏱️ ${runtime}</span>
            </div>
            <div class="movie-meta">
                <span class="meta-item">🎭 ${genres}</span>
            </div>
            <p class="movie-overview">${movie.overview || 'No overview available.'}</p>
        </div>
    `;
    
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
        
        // Hide movie section and show recommendations
        movieSection.classList.add('hidden');
        recommendationsSection.classList.remove('hidden');
        recommendationsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (error) {
        console.error('Recommendations error:', error);
        showToast('Failed to get recommendations. Please try again.', 'error');
    } finally {
        hideLoading();
    }
}

// Display Recommendations
function displayRecommendations(movies, algorithm) {
    if (!movies || movies.length === 0) {
        recommendations.innerHTML = `
            <div class="glass-card" style="padding: 2rem; text-align: center; grid-column: 1/-1;">
                <p style="color: var(--text-muted); font-size: 1.125rem;">
                    No recommendations found.
                </p>
            </div>
        `;
        return;
    }
    
    // Update algorithm badge
    algorithmUsed.textContent = algorithm || 'Jakube Algorithm';
    
    recommendations.innerHTML = movies.map(movie => `
        <div class="rec-card">
            <div class="glass-card glass-card-hover" onclick="selectMovie(${movie.id})">
                ${movie.poster_path 
                    ? `<img src="${TMDB_IMAGE_BASE}${movie.poster_path}" alt="${movie.title}" class="rec-poster">`
                    : `<div class="rec-poster placeholder">🎬</div>`
                }
                <div class="rec-info">
                    <div class="rec-title">${movie.title}</div>
                    <div class="rec-details">
                        <span class="rec-year">${movie.release_date ? movie.release_date.split('-')[0] : 'N/A'}</span>
                        ${movie.vote_average ? `<span class="rec-rating">⭐ ${movie.vote_average.toFixed(1)}</span>` : ''}
                    </div>
                </div>
            </div>
        </div>
    `).join('');
}

// Reset Search
function resetSearch() {
    selectedMovieId = null;
    searchInput.value = '';
    searchSection.classList.remove('hidden');
    movieSection.classList.add('hidden');
    recommendationsSection.classList.add('hidden');
    searchResults.classList.add('hidden');
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Loading Overlay
function showLoading() {
    loadingOverlay.classList.remove('hidden');
}

function hideLoading() {
    loadingOverlay.classList.add('hidden');
}

// Toast Notification
function showToast(message, type = 'info') {
    alert(message); // Simple fallback, can be enhanced with custom toast
}

// Initial Console Message
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
