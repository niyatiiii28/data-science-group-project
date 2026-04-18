/**
 * Search page JavaScript — handles search, rendering, and debounce.
 */

// Debounce helper
function debounce(fn, delay) {
    let timer;
    return function (...args) {
        clearTimeout(timer);
        timer = setTimeout(() => fn.apply(this, args), delay);
    };
}

// Enter key triggers search
document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('searchInput');
    if (input) {
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') performSearch();
        });
    }
});

// Fill input with sample query
function tryQuery(query) {
    const input = document.getElementById('searchInput');
    if (input) {
        input.value = query;
        performSearch();
    }
}

// Main search function
async function performSearch() {
    const query = document.getElementById('searchInput').value.trim();
    if (!query) return;

    const modelSelect = document.getElementById('modelSelect');
    const topkSelect = document.getElementById('topkSelect');
    const modelId = modelSelect ? modelSelect.value : '';
    const topK = topkSelect ? parseInt(topkSelect.value) : 5;

    // Show loading
    show('loading');
    hide('results');
    hide('searchMeta');
    hide('noResults');

    try {
        let url = '/api/search';
        let body = { query, top_k: topK };

        if (modelId === 'ensemble') {
            url = '/api/search/ensemble';
        } else if (modelId) {
            url = `/api/search/${modelId}`;
        }

        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        const data = await response.json();

        hide('loading');

        if (data.error) {
            showError(data.error);
            return;
        }

        if (!data.results || data.results.length === 0) {
            show('noResults');
            return;
        }

        renderMeta(data);
        renderResults(data.results);

    } catch (err) {
        hide('loading');
        showError('Search failed: ' + err.message);
    }
}

function renderMeta(data) {
    const meta = document.getElementById('searchMeta');
    if (!meta) return;
    meta.innerHTML = `
        <strong>${data.total_results}</strong> results for "<em>${data.query}</em>"
        &mdash; Model: <strong>${data.model_id}</strong>
        &mdash; Time: <strong>${data.query_time_ms}ms</strong>
    `;
    show('searchMeta');
}

function renderResults(results) {
    const container = document.getElementById('results');
    if (!container) return;

    container.innerHTML = results.map(r => {
        const confScore = (r.confidence_score * 100).toFixed(1);
        let confClass = 'confidence-low';
        if (r.confidence_score >= 0.7) confClass = 'confidence-high';
        else if (r.confidence_score >= 0.4) confClass = 'confidence-mid';

        return `
        <div class="result-card">
            <div class="result-header">
                <div class="result-rank">${r.rank}</div>
                <div class="result-title">${escapeHtml(r.product_name)}</div>
                <span class="confidence-badge ${confClass}">${confScore}%</span>
            </div>
            <div class="result-meta">
                <span>📁 ${escapeHtml(r.department)} › ${escapeHtml(r.category)}</span>
                <span>🏷️ ${escapeHtml(r.brand)}</span>
                <span>💰 ₹${Number(r.price).toLocaleString()}</span>
                <span>⭐ ${r.rating}/5</span>
            </div>
            ${r.description ? `<div class="result-description">${escapeHtml(r.description)}</div>` : ''}
        </div>`;
    }).join('');

    show('results');
}

function showError(message) {
    const container = document.getElementById('results');
    if (container) {
        container.innerHTML = `<div class="no-results"><p style="color:#ef4444">${escapeHtml(message)}</p></div>`;
        show('results');
    }
}

// Utility
function show(id) { document.getElementById(id)?.classList.remove('hidden'); }
function hide(id) { document.getElementById(id)?.classList.add('hidden'); }
function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
