/**
 * Q-FuseVision AI Lab — Main JavaScript v2
 * 3D effects, micro-animations, and UI interactions
 */

/* ═══════════════  AUTO-HIDE MESSAGES  ═══════════════ */
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.message').forEach(msg => {
        setTimeout(() => {
            msg.style.opacity = '0';
            msg.style.transform = 'translateX(80px) scale(0.95)';
            setTimeout(() => msg.remove(), 400);
        }, 5000);
    });
});

/* ═══════════════  NAVBAR SCROLL  ═══════════════ */
window.addEventListener('scroll', () => {
    const nav = document.querySelector('.nav');
    if (nav) nav.classList.toggle('scrolled', window.scrollY > 50);
});

/* ═══════════════  3D TILT EFFECT  ═══════════════ */
function initTiltCards() {
    document.querySelectorAll('[data-tilt]').forEach(card => {
        const intensity = parseFloat(card.dataset.tilt) || 8;

        card.addEventListener('mousemove', e => {
            const rect = card.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width - 0.5;
            const y = (e.clientY - rect.top) / rect.height - 0.5;

            card.style.transform =
                `perspective(${getComputedStyle(document.documentElement).getPropertyValue('--perspective').trim() || '1200px'}) ` +
                `rotateY(${x * intensity}deg) ` +
                `rotateX(${-y * intensity}deg) ` +
                `scale3d(1.02, 1.02, 1.02)`;
        });

        card.addEventListener('mouseleave', () => {
            card.style.transform =
                'perspective(1200px) rotateY(0deg) rotateX(0deg) scale3d(1, 1, 1)';
        });
    });
}

document.addEventListener('DOMContentLoaded', initTiltCards);

/* ═══════════════  STAGGERED ENTRANCE  ═══════════════ */
function initScrollReveal() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('revealed');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });

    document.querySelectorAll('.reveal').forEach((el, i) => {
        el.style.transitionDelay = `${i * 0.07}s`;
        observer.observe(el);
    });
}

document.addEventListener('DOMContentLoaded', initScrollReveal);

/* ═══════════════  FLOATING 3D ORBS  ═══════════════ */
function createOrbs(container, count = 6) {
    const colors = [
        'rgba(108, 138, 255, 0.12)',
        'rgba(168, 85, 247, 0.10)',
        'rgba(244, 114, 182, 0.08)',
        'rgba(52, 211, 153, 0.07)',
    ];

    for (let i = 0; i < count; i++) {
        const orb = document.createElement('div');
        orb.className = 'floating-orb';
        const size = 120 + Math.random() * 300;
        orb.style.cssText = `
            position: absolute;
            width: ${size}px;
            height: ${size}px;
            border-radius: 50%;
            background: ${colors[i % colors.length]};
            filter: blur(${40 + Math.random() * 40}px);
            left: ${Math.random() * 100}%;
            top: ${Math.random() * 100}%;
            animation: orbFloat${i % 3} ${18 + Math.random() * 15}s ease-in-out infinite;
            animation-delay: ${-Math.random() * 10}s;
            pointer-events: none;
        `;
        container.appendChild(orb);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const bg = document.querySelector('.bg-orbs');
    if (bg) createOrbs(bg, 6);
});

/* ═══════════════  TOAST SYSTEM  ═══════════════ */
function showToast(message, type = 'info') {
    let container = document.querySelector('.messages');
    if (!container) {
        container = document.createElement('div');
        container.className = 'messages';
        document.body.appendChild(container);
    }

    const icons = {
        success: 'fa-check-circle',
        error:   'fa-exclamation-circle',
        warning: 'fa-exclamation-triangle',
        info:    'fa-info-circle',
    };

    const toast = document.createElement('div');
    toast.className = `message ${type}`;
    toast.innerHTML = `<i class="fas ${icons[type] || icons.info}"></i>${message}`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(80px) scale(0.95)';
        setTimeout(() => toast.remove(), 400);
    }, 3500);
}

/* ═══════════════  HELPERS  ═══════════════ */
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showToast('Copied to clipboard!', 'success');
    }).catch(() => {
        showToast('Failed to copy', 'error');
    });
}

async function apiCall(url, options = {}) {
    const response = await fetch(url, {
        headers: { 'Content-Type': 'application/json', ...options.headers },
        ...options,
    });
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    return response.json();
}

async function checkExperimentStatus(experimentId, callback) {
    try { callback(await apiCall(`/api/status/${experimentId}/`)); }
    catch (e) { console.error('Status check failed:', e); }
}

async function checkOllamaStatus(callback) {
    try { callback(await apiCall('/api/ollama/')); }
    catch (e) { console.error('Ollama check failed:', e); }
}

function scrollToElement(selector) {
    document.querySelector(selector)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function debounce(func, wait) {
    let t;
    return (...args) => { clearTimeout(t); t = setTimeout(() => func(...args), wait); };
}

/* ═══════════════  GLOBAL EXPORT  ═══════════════ */
window.QFuseVision = {
    formatNumber, copyToClipboard, showToast, apiCall,
    checkExperimentStatus, checkOllamaStatus, createOrbs,
    scrollToElement, debounce, initTiltCards, initScrollReveal,
};
