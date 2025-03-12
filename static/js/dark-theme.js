// Dark theme toggle functionality
document.addEventListener('DOMContentLoaded', function() {
    // Get theme toggle button element
    const themeToggle = document.getElementById('theme-toggle');

    // Check for saved theme preference or get OS preference
    const savedTheme = localStorage.getItem('theme');
    const osPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

    // Apply appropriate theme
    if (savedTheme === 'dark' || (!savedTheme && osPrefersDark)) {
        enableDarkTheme();
    } else {
        enableLightTheme();
    }

    // Theme toggle click handler
    themeToggle.addEventListener('click', function() {
        if (document.body.classList.contains('dark-theme')) {
            enableLightTheme();
        } else {
            enableDarkTheme();
        }
    });

    // Functions to enable dark/light themes
    function enableDarkTheme() {
        document.body.classList.remove('light-theme');
        document.body.classList.add('dark-theme');
        localStorage.setItem('theme', 'dark');

        // Update chart colors if they exist
        updateChartsTheme('dark');
    }

    function enableLightTheme() {
        document.body.classList.remove('dark-theme');
        document.body.classList.add('light-theme');
        localStorage.setItem('theme', 'light');

        // Update chart colors if they exist
        updateChartsTheme('light');
    }

    // Update chart colors based on theme
    function updateChartsTheme(theme) {
        // Check if Chart.js is loaded and charts exist
        if (typeof Chart !== 'undefined') {
            const textColor = theme === 'dark' ? '#e0e0e0' : '#666666';
            const gridColor = theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';

            // Update Chart.js defaults
            Chart.defaults.color = textColor;
            Chart.defaults.scale.grid.color = gridColor;

            // Get all charts on the page and update them
            const charts = Object.values(Chart.instances || {});
            charts.forEach(chart => {
                // Update text colors
                if (chart.options.scales && chart.options.scales.x) {
                    chart.options.scales.x.ticks.color = textColor;
                    chart.options.scales.x.title.color = textColor;
                }
                if (chart.options.scales && chart.options.scales.y) {
                    chart.options.scales.y.ticks.color = textColor;
                    chart.options.scales.y.title.color = textColor;
                }
                if (chart.options.scales && chart.options.scales.r) {
                    chart.options.scales.r.pointLabels.color = textColor;
                    chart.options.scales.r.angleLines.color = gridColor;
                }

                // Update grid colors
                if (chart.options.scales && chart.options.scales.x) {
                    chart.options.scales.x.grid.color = gridColor;
                }
                if (chart.options.scales && chart.options.scales.y) {
                    chart.options.scales.y.grid.color = gridColor;
                }
                if (chart.options.scales && chart.options.scales.r) {
                    chart.options.scales.r.grid.color = gridColor;
                }

                // Update the chart
                chart.update();
            });
        }
    }

    // Listen for OS theme changes
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
        // Only automatically switch if user hasn't manually set a preference
        if (!localStorage.getItem('theme')) {
            if (e.matches) {
                enableDarkTheme();
            } else {
                enableLightTheme();
            }
        }
    });
});

// Window-level function to fetch state, made global for timeline animations
window.fetchState = async function() {
    if (typeof identityChart === 'undefined' || typeof timelineChart === 'undefined') {
        return; // Charts not initialized yet
    }

    try {
        const response = await fetch('/state');
        if (!response.ok) {
            throw new Error('Failed to fetch state');
        }

        const data = await response.json();

        // Update all visualizations
        if (data.identity) {
            updateIdentity(data.identity);
        }

        if (data.time_perception) {
            updateTimePerception(data.time_perception);
        }

        if (data.futures) {
            updateFutures(data.futures);
        }

        if (data.preferred_futures) {
            updatePreferredFutures(data.preferred_futures);
        }

        if (data.future_history && data.future_history.length > 0) {
            // Only update if we have new data
            if (!window.lastHistoryTime ||
                window.lastHistoryTime !== data.future_history[data.future_history.length - 1].time) {
                updateTimeline(data.future_history);
                window.lastHistoryTime = data.future_history[data.future_history.length - 1].time;
            }
        }

    } catch (error) {
        console.error('Error updating state:', error);
    }
};

// Functions that need to be global for updateFutureColors
window.updateFutureColors = function() {
    // Update future item colors based on theme
    const isDark = document.body.classList.contains('dark-theme');

    // Preferred futures
    const preferredFutures = document.querySelectorAll('.preferred-future-item');
    preferredFutures.forEach(item => {
        if (isDark) {
            item.style.backgroundColor = '#1e2a1e';
            item.style.borderLeftColor = '#1d7334';
        } else {
            item.style.backgroundColor = '#f1f9f1';
            item.style.borderLeftColor = '#28a745';
        }
    });

    // Regular futures
    const futures = document.querySelectorAll('.future-item:not(.preferred-future-item)');
    futures.forEach(item => {
        if (isDark) {
            item.style.backgroundColor = '#282828';
            item.style.borderLeftColor = '#4a7ab2';
        } else {
            item.style.backgroundColor = '#f9f9f9';
            item.style.borderLeftColor = '#4f92cf';
        }
    });

    // Probability badges
    const badges = document.querySelectorAll('.probability-badge');
    badges.forEach(badge => {
        if (isDark) {
            badge.style.backgroundColor = 'rgba(0, 123, 255, 0.3)';
            badge.style.color = '#8bb8ff';
        } else {
            badge.style.backgroundColor = 'rgba(0, 123, 255, 0.2)';
            badge.style.color = '#0056b3';
        }
    });

    // Alignment badges
    const alignmentBadges = document.querySelectorAll('.alignment-badge');
    alignmentBadges.forEach(badge => {
        if (isDark) {
            badge.style.backgroundColor = 'rgba(40, 167, 69, 0.3)';
            badge.style.color = '#7dd992';
        } else {
            badge.style.backgroundColor = 'rgba(40, 167, 69, 0.2)';
            badge.style.color = '#28a745';
        }
    });
};

// Call updateFutureColors when theme changes
document.addEventListener('DOMContentLoaded', function() {
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', function() {
            // Wait for the theme change to complete
            setTimeout(function() {
                if (window.updateFutureColors) {
                    window.updateFutureColors();
                }
            }, 100);
        });
    }
});