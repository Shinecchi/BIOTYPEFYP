/**
 * app.js — BioType Dashboard WebSocket Client
 *
 * Captures live keystroke events from the browser (keydown/keyup),
 * sends them to the FastAPI server via WebSocket, and updates
 * the trust gauge + decision badge in real-time.
 */

// -------------------------------------------------------------------
// DOM References
// -------------------------------------------------------------------
const enrollInput    = document.getElementById('enroll-input');
const enrollBtn      = document.getElementById('enroll-btn');
const enrollCount    = document.getElementById('enroll-event-count');
const enrollStatus   = document.getElementById('enroll-status');
const enrollSection  = document.getElementById('enroll-section');
const verifySection  = document.getElementById('verify-section');
const verifyInput    = document.getElementById('verify-input');
const reEnrollBtn    = document.getElementById('re-enroll-btn');
const connBadge      = document.getElementById('conn-badge');
const trustPct       = document.getElementById('trust-pct');
const gaugeArc       = document.getElementById('gauge-arc');
const decisionBadge  = document.getElementById('decision-badge');
const decisionIcon   = document.getElementById('decision-icon');
const decisionText   = document.getElementById('decision-text');
const statWindows    = document.getElementById('stat-windows');
const statDistance   = document.getElementById('stat-distance');
const statEnrolled   = document.getElementById('stat-enrolled');
const eventLog       = document.getElementById('event-log');
const clearLogBtn    = document.getElementById('clear-log-btn');

// -------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------
const GAUGE_CIRCUMFERENCE = 553;  // 2π × r (r=88)
const WS_URL = `ws://${location.host}/ws/verify`;
const MIN_ENROLL_EVENTS = 30;     // require at least 30 events before enabling enroll button

// -------------------------------------------------------------------
// State
// -------------------------------------------------------------------
let enrollEvents  = [];   // captured during enrollment phase
let verifyEvents  = [];   // currently buffered verify events (sent incrementally)
let socket        = null;
let isEnrolled    = false;
let verifyStartTs = null; // high-precision base timestamp for this verify session

// -------------------------------------------------------------------
// Utility: high-precision timestamp (seconds, relative)
// -------------------------------------------------------------------
function nowSeconds() {
  return performance.now() / 1000;
}

// -------------------------------------------------------------------
// WebSocket Setup
// -------------------------------------------------------------------
function connectWebSocket() {
  socket = new WebSocket(WS_URL);

  socket.onopen = () => {
    connBadge.textContent = '● ONLINE';
    connBadge.className   = 'badge badge-online';
    console.log('[WS] Connected');
  };

  socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.enrolled) {
      updateGauge(data.trust, data.decision, data.distance, data.windows);
    }
  };

  socket.onclose = () => {
    connBadge.textContent = '● OFFLINE';
    connBadge.className   = 'badge badge-offline';
    setTimeout(connectWebSocket, 3000); // auto-reconnect
  };

  socket.onerror = (err) => console.error('[WS] Error:', err);
}

// -------------------------------------------------------------------
// Gauge + Decision Update
// -------------------------------------------------------------------
function updateGauge(trust, decision, distance, windows) {
  const pct     = Math.round(trust * 100);
  const offset  = GAUGE_CIRCUMFERENCE * (1 - trust);

  // Trust number
  trustPct.textContent = pct + '%';
  gaugeArc.style.strokeDashoffset = offset;

  // Color based on trust level
  let color;
  if (trust >= 0.75)      color = 'var(--green)';
  else if (trust >= 0.45) color = 'var(--amber)';
  else                    color = 'var(--red)';

  document.documentElement.style.setProperty('--trust-color', color);
  gaugeArc.style.stroke = color;
  trustPct.style.color  = color;

  // Decision badge
  decisionBadge.className = 'decision-badge';
  if (decision === 'ACCESS_GRANTED') {
    decisionBadge.classList.add('decision-granted');
    decisionIcon.textContent = '✓';
    decisionText.textContent = 'ACCESS GRANTED';
  } else if (decision === 'CHALLENGE') {
    decisionBadge.classList.add('decision-challenge');
    decisionIcon.textContent = '⚠';
    decisionText.textContent = 'CHALLENGE';
  } else {
    decisionBadge.classList.add('decision-revoked');
    decisionIcon.textContent = '✕';
    decisionText.textContent = 'ACCESS REVOKED';
  }

  // Stats
  statWindows .textContent  = windows;
  statDistance.textContent  = distance.toFixed(4);

  // Log entry
  addLogEntry(trust, decision, distance);
}

function addLogEntry(trust, decision, distance) {
  // Remove the "empty" placeholder
  const empty = eventLog.querySelector('.log-empty');
  if (empty) empty.remove();

  const now  = new Date();
  const time = now.toTimeString().slice(0, 8);
  const cls  = decision === 'ACCESS_GRANTED' ? 'granted'
             : decision === 'CHALLENGE'      ? 'challenge'
             : 'revoked';
  const label = decision === 'ACCESS_GRANTED' ? 'GRANTED'
              : decision === 'CHALLENGE'       ? 'CHALLENGE'
              : 'REVOKED';

  const entry = document.createElement('div');
  entry.className = `log-entry ${cls}`;
  entry.innerHTML = `
    <span class="log-time">${time}</span>
    <span class="log-trust">${Math.round(trust * 100)}%</span>
    <span class="log-dist">d=${distance.toFixed(3)}</span>
    <span class="log-decision">${label}</span>
  `;

  eventLog.appendChild(entry);
  // Keep last 200 entries
  while (eventLog.children.length > 200) eventLog.removeChild(eventLog.firstChild);
  eventLog.scrollTop = eventLog.scrollHeight;
}

// -------------------------------------------------------------------
// Enrollment Phase
// -------------------------------------------------------------------
enrollInput.addEventListener('keydown', (e) => {
  enrollEvents.push({ key: normalizeKey(e), event_type: 'down', ts: nowSeconds() });
  updateEnrollCount();
});

enrollInput.addEventListener('keyup', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    // ENTER submits enrollment
    e.preventDefault();
    if (!enrollBtn.disabled) triggerEnroll();
    return;
  }
  enrollEvents.push({ key: normalizeKey(e), event_type: 'up', ts: nowSeconds() });
  updateEnrollCount();
});

function updateEnrollCount() {
  enrollCount.textContent = `${enrollEvents.length} events captured`;
  enrollBtn.disabled = enrollEvents.length < MIN_ENROLL_EVENTS;
}

enrollBtn.addEventListener('click', triggerEnroll);

async function triggerEnroll() {
  enrollBtn.disabled = true;
  setEnrollStatus('Enrolling… building your biometric profile', '');

  try {
    const res = await fetch('/enroll', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ events: enrollEvents }),
    });
    const data = await res.json();

    if (data.success) {
      setEnrollStatus(`✓ ${data.message}`, 'success');
      isEnrolled = true;
      statEnrolled.textContent = 'Yes';

      // Switch panels
      enrollSection.classList.add('card-disabled');
      verifySection.classList.remove('card-disabled');
      verifySection.classList.add('card-active');
      verifyInput.disabled = false;
      verifyInput.focus();

      // Update decision badge to show ready
      decisionBadge.className = 'decision-badge decision-idle';
      decisionIcon.textContent = '◉';
      decisionText.textContent = 'ENROLLED — MONITORING';

    } else {
      setEnrollStatus(`✕ ${data.message}`, 'error');
      enrollBtn.disabled = false;
    }
  } catch (err) {
    setEnrollStatus('Server error — is the server running?', 'error');
    enrollBtn.disabled = false;
  }
}

function setEnrollStatus(msg, cls) {
  enrollStatus.textContent = msg;
  enrollStatus.className = `status-msg ${cls}`;
}

// -------------------------------------------------------------------
// Verification Phase — WebSocket Keystroke Streaming
// -------------------------------------------------------------------
let pendingVerifyEvents    = [];
let verifyFlushTimer       = null;

verifyInput.addEventListener('keydown', (e) => {
  pendingVerifyEvents.push({ key: normalizeKey(e), event_type: 'down', ts: nowSeconds() });
  scheduleFlush();
});

verifyInput.addEventListener('keyup', (e) => {
  pendingVerifyEvents.push({ key: normalizeKey(e), event_type: 'up', ts: nowSeconds() });
  scheduleFlush();
});

function scheduleFlush() {
  // Debounce: send batches every 300ms to avoid flooding the server
  if (verifyFlushTimer) return;
  verifyFlushTimer = setTimeout(() => {
    flushVerifyEvents();
    verifyFlushTimer = null;
  }, 300);
}

function flushVerifyEvents() {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  if (pendingVerifyEvents.length === 0) return;

  socket.send(JSON.stringify({ events: pendingVerifyEvents }));
  pendingVerifyEvents = [];
}

// -------------------------------------------------------------------
// Re-enroll
// -------------------------------------------------------------------
reEnrollBtn.addEventListener('click', () => {
  enrollEvents = [];
  updateEnrollCount();
  enrollInput.value   = '';
  verifyInput.value   = '';
  verifyInput.disabled = true;
  pendingVerifyEvents  = [];
  isEnrolled = false;
  statEnrolled.textContent = 'No';

  enrollSection.classList.remove('card-disabled');
  verifySection.classList.remove('card-active');
  verifySection.classList.add('card-disabled');
  setEnrollStatus('', '');

  trustPct.textContent = '—';
  gaugeArc.style.strokeDashoffset = GAUGE_CIRCUMFERENCE;
  decisionBadge.className = 'decision-badge decision-idle';
  decisionIcon.textContent = '◉';
  decisionText.textContent = 'AWAITING ENROLLMENT';
  statWindows.textContent  = '0';
  statDistance.textContent = '—';

  enrollInput.focus();
});

// -------------------------------------------------------------------
// Clear Log
// -------------------------------------------------------------------
clearLogBtn.addEventListener('click', () => {
  eventLog.innerHTML = '<div class="log-empty">Authentication events will appear here…</div>';
});

// -------------------------------------------------------------------
// Utility: normalize browser key names to match Python event_schema
// -------------------------------------------------------------------
function normalizeKey(e) {
  const k = e.key;
  if (k === ' ')         return 'space';
  if (k === 'Backspace') return 'backspace';
  if (k === 'Delete')    return 'delete';
  if (k === 'Enter')     return 'enter';
  if (k === 'Tab')       return 'tab';
  if (k.startsWith('Shift'))   return 'key.shift';
  if (k.startsWith('Control')) return 'key.ctrl';
  if (k.startsWith('Alt'))     return 'key.alt';
  if (k.startsWith('Meta'))    return 'key.meta';
  if (k.length === 1)    return k.toLowerCase();
  return k.toLowerCase();
}

// -------------------------------------------------------------------
// Boot
// -------------------------------------------------------------------
connectWebSocket();
enrollInput.focus();
