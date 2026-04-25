// Mobile nav toggle
document.addEventListener('DOMContentLoaded', function() {
  var toggle = document.getElementById('navToggle');
  var links = document.getElementById('navLinks');
  if (toggle && links) {
    toggle.addEventListener('click', function() {
      links.classList.toggle('open');
    });
  }

  // Dark mode toggle
  var darkBtn = document.getElementById('darkToggle');
  var saved = localStorage.getItem('theme');
  if (saved === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
    if (darkBtn) darkBtn.textContent = '☀️';
  }

  if (darkBtn) {
    darkBtn.addEventListener('click', function() {
      var current = document.documentElement.getAttribute('data-theme');
      if (current === 'dark') {
        document.documentElement.removeAttribute('data-theme');
        darkBtn.textContent = '🌙';
        localStorage.setItem('theme', 'light');
      } else {
        document.documentElement.setAttribute('data-theme', 'dark');
        darkBtn.textContent = '☀️';
        localStorage.setItem('theme', 'dark');
      }
    });
  }
});

/* ===== Filter & Pagination System ===== */
var currentSearchMode = 'title';
var activeTags = [];
var currentFilterContext = null;

function setSearchMode(mode, btn) {
  currentSearchMode = mode;
  // Update active state on buttons in same container
  var container = btn.parentElement;
  container.querySelectorAll('.search-mode-btn').forEach(function(b) {
    b.classList.remove('active');
  });
  btn.classList.add('active');
  // Re-run filter
  if (currentFilterContext) applyFilters();
}

function toggleTag(btn) {
  var tag = btn.getAttribute('data-tag');
  btn.classList.toggle('active');
  var idx = activeTags.indexOf(tag);
  if (idx > -1) {
    activeTags.splice(idx, 1);
  } else {
    activeTags.push(tag);
  }
  if (currentFilterContext) applyFilters();
}

function clearFilters() {
  activeTags = [];
  document.querySelectorAll('.tag-filter-btn').forEach(function(btn) {
    btn.classList.remove('active');
  });
  var ctx = currentFilterContext;
  if (ctx) {
    document.getElementById(ctx.searchInputId).value = '';
    applyFilters();
  }
}

function applyFilters() {
  var ctx = currentFilterContext;
  if (!ctx) return;

  var searchVal = document.getElementById(ctx.searchInputId).value.trim().toLowerCase();
  var cards = ctx.allCards;
  var visibleCards = [];

  cards.forEach(function(card) {
    var title = (card.getAttribute('data-title') || '').toLowerCase();
    var tags = (card.getAttribute('data-tags') || '').split(',').map(function(t) { return t.trim(); });
    var show = true;

    // Search filter
    if (searchVal) {
      if (currentSearchMode === 'title') {
        if (title.indexOf(searchVal) === -1) show = false;
      } else {
        // tag search
        var tagMatch = tags.some(function(t) {
          return t.toLowerCase().indexOf(searchVal) > -1;
        });
        if (!tagMatch) show = false;
      }
    }

    // Tag filter
    if (show && activeTags.length > 0) {
      var hasTag = activeTags.some(function(activeTag) {
        return tags.indexOf(activeTag) > -1;
      });
      if (!hasTag) show = false;
    }

    if (show) {
      visibleCards.push(card);
    }
  });

  ctx.filteredCards = visibleCards;
  ctx.currentPage = 1;
  renderPage();
}

function renderPage() {
  var ctx = currentFilterContext;
  if (!ctx) return;

  var cards = ctx.filteredCards;
  var perPage = ctx.perPage;
  var totalPages = Math.ceil(cards.length / perPage) || 1;
  var page = ctx.currentPage;

  // Hide all
  ctx.allCards.forEach(function(c) { c.classList.add('hidden'); });

  // Show current page
  var start = (page - 1) * perPage;
  var end = start + perPage;
  for (var i = start; i < end && i < cards.length; i++) {
    cards[i].classList.remove('hidden');
  }

  // No results
  var noResults = document.getElementById(ctx.noResultsId);
  if (cards.length === 0) {
    noResults.style.display = 'block';
  } else {
    noResults.style.display = 'none';
  }

  // Render pagination
  renderPagination(totalPages, page);
}

function renderPagination(totalPages, currentPage) {
  var ctx = currentFilterContext;
  var paginationEl = document.getElementById(ctx.paginationId);
  paginationEl.innerHTML = '';

  if (totalPages <= 1) return;

  // Previous button
  var prevBtn = document.createElement('button');
  prevBtn.className = 'pagination-btn pagination-arrow' + (currentPage === 1 ? ' disabled' : '');
  prevBtn.textContent = 'הקודם →';
  prevBtn.onclick = function() { if (currentPage > 1) goToPage(currentPage - 1); };
  paginationEl.appendChild(prevBtn);

  // Page numbers
  for (var i = 1; i <= totalPages; i++) {
    var btn = document.createElement('button');
    btn.className = 'pagination-btn' + (i === currentPage ? ' active' : '');
    btn.textContent = i;
    (function(pageNum) {
      btn.onclick = function() { goToPage(pageNum); };
    })(i);
    paginationEl.appendChild(btn);
  }

  // Next button
  var nextBtn = document.createElement('button');
  nextBtn.className = 'pagination-btn pagination-arrow' + (currentPage === totalPages ? ' disabled' : '');
  nextBtn.textContent = '← הבא';
  nextBtn.onclick = function() { if (currentPage < totalPages) goToPage(currentPage + 1); };
  paginationEl.appendChild(nextBtn);
}

function goToPage(page) {
  currentFilterContext.currentPage = page;
  renderPage();
  // Scroll to top of grid
  var grid = document.getElementById(currentFilterContext.gridId);
  if (grid) grid.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function initFilterSystem(gridId, searchInputId, paginationId, noResultsId, perPage) {
  var grid = document.getElementById(gridId);
  if (!grid) return;

  var allCards = Array.from(grid.querySelectorAll('.filterable-card'));

  currentFilterContext = {
    gridId: gridId,
    searchInputId: searchInputId,
    paginationId: paginationId,
    noResultsId: noResultsId,
    perPage: perPage || 10,
    allCards: allCards,
    filteredCards: allCards.slice(),
    currentPage: 1
  };

  // Bind search input
  var searchInput = document.getElementById(searchInputId);
  if (searchInput) {
    searchInput.addEventListener('input', function() {
      applyFilters();
    });
  }

  // Initial render
  renderPage();
}
