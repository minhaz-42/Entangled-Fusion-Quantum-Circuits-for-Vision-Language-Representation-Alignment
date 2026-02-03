/**
 * Q-FuseVision AI Lab - Main JavaScript
 * Handles UI interactions and API calls
 */

// Auto-hide messages after 5 seconds
document.addEventListener('DOMContentLoaded', function() {
    const messages = document.querySelectorAll('.message');
    messages.forEach(message => {
        setTimeout(() => {
            message.style.opacity = '0';
            message.style.transform = 'translateX(100px)';
            setTimeout(() => message.remove(), 300);
        }, 5000);
    });
});

// Navbar scroll effect
window.addEventListener('scroll', function() {
    const navbar = document.querySelector('.navbar');
    if (navbar) {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    }
});

// Intersection Observer for animations
const observeElements = () => {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll('.animate-on-scroll').forEach(el => {
        observer.observe(el);
    });
};

document.addEventListener('DOMContentLoaded', observeElements);

// Format numbers with commas
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

// Copy to clipboard
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showToast('Copied to clipboard!', 'success');
    }).catch(() => {
        showToast('Failed to copy', 'error');
    });
}

// Toast notifications
function showToast(message, type = 'info') {
    const container = document.querySelector('.messages') || createToastContainer();
    
    const toast = document.createElement('div');
    toast.className = `message ${type}`;
    toast.innerHTML = `
        <i class="fas ${type === 'success' ? 'fa-check-circle' : type === 'error' ? 'fa-exclamation-circle' : 'fa-info-circle'}"></i>
        ${message}
    `;
    
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100px)';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function createToastContainer() {
    const container = document.createElement('div');
    container.className = 'messages';
    document.body.appendChild(container);
    return container;
}

// API helper
async function apiCall(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
            ...options,
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        throw error;
    }
}

// Check experiment status
async function checkExperimentStatus(experimentId, callback) {
    try {
        const data = await apiCall(`/api/status/${experimentId}/`);
        callback(data);
    } catch (error) {
        console.error('Failed to check status:', error);
    }
}

// Check Ollama status
async function checkOllamaStatus(callback) {
    try {
        const data = await apiCall('/api/ollama/');
        callback(data);
    } catch (error) {
        console.error('Failed to check Ollama status:', error);
    }
}

// Particle animation
function createParticles(container, count = 50) {
    for (let i = 0; i < count; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.cssText = `
            left: ${Math.random() * 100}%;
            top: ${Math.random() * 100}%;
            animation-delay: ${Math.random() * 20}s;
            animation-duration: ${15 + Math.random() * 20}s;
        `;
        container.appendChild(particle);
    }
}

// Initialize particles on pages with particle containers
document.addEventListener('DOMContentLoaded', function() {
    const particleContainer = document.querySelector('.particle-container');
    if (particleContainer) {
        createParticles(particleContainer, 50);
    }
});

// Smooth scroll to element
function scrollToElement(selector) {
    const element = document.querySelector(selector);
    if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// Debounce function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Export functions for global use
window.QFuseVision = {
    formatNumber,
    copyToClipboard,
    showToast,
    apiCall,
    checkExperimentStatus,
    checkOllamaStatus,
    createParticles,
    scrollToElement,
    debounce,
};
