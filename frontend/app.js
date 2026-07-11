// ── State ─────────────────────────────────────────────────────────────────────
let ws = null;
let isRecording = false;
let audioContext = null;
let analyser = null;
let dataArray = null;
let animationId = null;
let idleAnimId = null;
let idlePhase = 0;

let timerInterval = null;
let secondsElapsed = 0;
let totalWords = 0;
let latencies = [];
let fullTranscriptText = "";
let currentLiveCard = null;

// Reconnect/Session persistence state
let currentMeetingId = null;
let wasRecordingBeforeDisconnect = false;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const micBtn = document.getElementById('micBtn');
const micStatusText = document.getElementById('micStatusText');
const transcriptArea = document.getElementById('transcriptArea');
const emptyState = document.getElementById('emptyState');
const autoScrollToggle = document.getElementById('autoScrollToggle');

const timerDisplay = document.getElementById('timer');
const statDuration = document.getElementById('statDuration');
const statWords = document.getElementById('statWords');
const statLatency = document.getElementById('statLatency');

const wsStatusDot = document.getElementById('wsStatusDot');
const wsStatusText = document.getElementById('wsStatusText');
const dbStatusEl = document.getElementById('dbStatus');
const dbStatusText = document.getElementById('dbStatusText');

const dashWsDot = document.getElementById('dashWsDot');
const dashWsText = document.getElementById('dashWsText');

const sourceLangSelect = document.getElementById('sourceLang');
const targetLangSelect = document.getElementById('targetLang');
const envSelect = document.getElementById('envSelect');
const hdrSourceLang = document.getElementById('hdrSourceLang');
const hdrTargetLang = document.getElementById('hdrTargetLang');

const exportTxtBtn = document.getElementById('exportTxtBtn');
const toggleLeftBtn = document.getElementById('toggleLeftBtn');
const toggleRightBtn = document.getElementById('toggleRightBtn');
const appContainer = document.querySelector('.app-container');

const canvas = document.getElementById('waveform');
const canvasCtx = canvas.getContext('2d');

// ── Waveform helpers (defined FIRST before any calls) ─────────────────────────
const NUM_BARS = 38;

/** HSL transition: pink-purple (300°) → indigo-blue (220°) */
function barColor(i, alpha) {
    const t = i / (NUM_BARS - 1);
    const hue = Math.round(300 - t * 80);   // 300 → 220
    const sat = Math.round(80 + t * 5);    // 80  → 85
    const lit = Math.round(62 - t * 8);    // 62  → 54
    return `hsla(${hue},${sat}%,${lit}%,${alpha})`;
}

/** Draw a rounded-capsule bar centred on (cx, cy) */
function drawCapsule(x, halfH, barW, colorTop, colorBot) {
    const W = canvas;
    const H = canvas.height;
    const cy = H / 2;
    const y = cy - halfH;
    const h = halfH * 2;
    const r = barW / 2;

    const grad = canvasCtx.createLinearGradient(x, y, x, y + h);
    grad.addColorStop(0, colorTop);
    grad.addColorStop(0.5, colorTop.replace(/,[\d.]+\)$/, ',1.0)'));
    grad.addColorStop(1, colorBot);
    canvasCtx.fillStyle = grad;

    canvasCtx.beginPath();
    if (canvasCtx.roundRect) {
        canvasCtx.roundRect(x, y, barW, h, r);
    } else {
        // Fallback for older browsers
        canvasCtx.arc(x + r, y + r, r, Math.PI, 1.5 * Math.PI);
        canvasCtx.arc(x + barW - r, y + r, r, 1.5 * Math.PI, 0);
        canvasCtx.arc(x + barW - r, y + h - r, r, 0, 0.5 * Math.PI);
        canvasCtx.arc(x + r, y + h - r, r, 0.5 * Math.PI, Math.PI);
        canvasCtx.closePath();
    }
    canvasCtx.fill();
}

function drawSilentWaveform() {
    const W = canvas.width;
    const H = canvas.height;
    canvasCtx.clearRect(0, 0, W, H);

    const slotW = W / NUM_BARS;
    const barW = slotW * 0.55;

    for (let i = 0; i < NUM_BARS; i++) {
        // Continuous dots pulsing slightly in opacity
        const sine = Math.sin((i / NUM_BARS) * Math.PI * 3 + idlePhase);
        const alpha = 0.15 + ((sine + 1) / 2) * 0.4; // Opacity between 0.15 and 0.55

        const cx = slotW * i + slotW / 2;
        const cy = H / 2;
        const r = barW / 2;

        canvasCtx.beginPath();
        canvasCtx.arc(cx, cy, r, 0, 2 * Math.PI);
        canvasCtx.fillStyle = barColor(i, alpha);
        canvasCtx.fill();
    }
}

function startIdleAnimation() {
    function tick() {
        if (isRecording) return; // hand off to active loop
        idlePhase += 0.04;
        drawSilentWaveform();
        idleAnimId = requestAnimationFrame(tick);
    }
    cancelAnimationFrame(idleAnimId);
    tick();
}

function drawWaveform() {
    if (!isRecording || !analyser) return;
    animationId = requestAnimationFrame(drawWaveform);

    analyser.getByteFrequencyData(dataArray);

    const W = canvas.width;
    const H = canvas.height;
    canvasCtx.clearRect(0, 0, W, H);

    const slotW = W / NUM_BARS;
    const barW = slotW * 0.60;
    const maxH = H / 2 - 2;

    for (let i = 0; i < NUM_BARS; i++) {
        // Sample from frequency bins, weighting toward lower-mid (speech) range
        const binIdx = Math.floor((i / NUM_BARS) * (dataArray.length * 0.75));
        const norm = dataArray[binIdx] / 255.0;

        // Slight gaussian envelope so centre bars are taller when loud
        const env = Math.exp(-Math.pow((i / NUM_BARS - 0.5) * 2.2, 2) * 0.5);
        const halfH = Math.max(3, (norm * 0.8 + norm * norm * 0.2) * maxH * (0.55 + env * 0.65));

        const x = slotW * i + (slotW - barW) / 2;
        const top = barColor(i, 0.90);
        const bot = barColor(i, 0.65);
        drawCapsule(x, halfH, barW, top, bot);
    }
}

// ── Timer ─────────────────────────────────────────────────────────────────────
function formatTime(sec) {
    const h = Math.floor(sec / 3600).toString().padStart(2, '0');
    const m = Math.floor((sec % 3600) / 60).toString().padStart(2, '0');
    const s = (sec % 60).toString().padStart(2, '0');
    return `${h}:${m}:${s}`;
}
function updateTimer() {
    secondsElapsed++;
    const t = formatTime(secondsElapsed);
    timerDisplay.innerText = t;
    statDuration.innerText = t;
}

// ── WebSocket ─────────────────────────────────────────────────────────────────
function connectWebSocket() {
    if (ws && ws.readyState === WebSocket.OPEN) return; // already open

    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = location.host || 'localhost:8000';
    // Token is injected into the HTML by the server at serve-time.
    // It rotates on every server restart and never appears in source code.
    const TOKEN = window.__WS_TOKEN__ || "";
    let wsUrl = `${protocol}//${host}/ws/translate?token=${TOKEN}`;
    if (currentMeetingId) {
        wsUrl += `&meeting_id=${currentMeetingId}`;
    }

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        wsStatusDot.className = 'indicator-dot online';
        wsStatusText.innerText = 'Connected';
        if (dashWsDot) {
            dashWsDot.className = 'indicator-dot online';
            dashWsDot.style.boxShadow = '0 0 8px rgba(34, 197, 94, 0.6)';
        }
        if (dashWsText) dashWsText.innerText = 'Connected';
        dbStatusText.innerText = 'Ready to sync';
        dbStatusEl.className = 'db-status';
        sendConfig();

        if (wasRecordingBeforeDisconnect) {
            wasRecordingBeforeDisconnect = false;
            setTimeout(() => {
                if (!isRecording) startRecording();
            }, 300);
        }
    };

    ws.onmessage = (event) => {
        if (event.data instanceof Blob) return; // TTS audio – skip for now
        let data;
        try { data = JSON.parse(event.data); } catch { return; }

        if (data.type === 'meeting_created') {
            currentMeetingId = data.meeting_id;
            const titleEl = document.getElementById('meetingTitle');
            const sidebarTitleEl = document.getElementById('sidebarMeetingTitle');
            if (titleEl && data.title) {
                titleEl.innerHTML = `${data.title} <span class="badge-live sm">LIVE</span> <span id="timer">${timerDisplay.innerText}</span>`;
            }
            if (sidebarTitleEl && data.title) {
                sidebarTitleEl.innerText = data.title;
            }
        }
        else if (data.type === 'subtitle') handleSubtitle(data);
        else if (data.type === 'subtitle_update') handleSubtitleUpdate(data);
        else if (data.type === 'done') handleDone(data);
        else if (data.type === 'speaker_renamed') handleSpeakerRenamed(data);
        else if (data.type === 'speakers_merged') handleSpeakersMerged(data);
    };

    ws.onclose = (event) => {
        wsStatusDot.className = 'indicator-dot offline';
        wsStatusText.innerText = 'Disconnected';
        if (dashWsDot) {
            dashWsDot.className = 'indicator-dot offline';
            dashWsDot.style.boxShadow = '0 0 8px rgba(239, 68, 68, 0.6)';
        }
        if (dashWsText) dashWsText.innerText = 'Disconnected';
        
        if (isRecording) {
            wasRecordingBeforeDisconnect = true;
            stopRecording();
        } else {
            wasRecordingBeforeDisconnect = false;
        }

        // 4003 = auth rejected (stale token after server restart).
        // Retrying with the same cached token is futile — reload the page
        // to get the fresh token from the server-rendered HTML.
        if (event.code === 4003) {
            console.warn('[WS] Auth rejected (token expired). Reloading page to get fresh token...');
            wsStatusText.innerText = 'Token expired — reloading...';
            setTimeout(() => location.reload(), 1000);
            return;
        }

        // Normal disconnect (network blip, server restart in progress) —
        // auto-reconnect after 3 seconds.
        setTimeout(connectWebSocket, 3000);
    };
    ws.onerror = (e) => {
        console.error('[WS Error]', e);
        // onerror is always followed by onclose, so the reconnect
        // timer above will fire — no separate retry needed here.
    };
}

function sendConfig() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({
        action: 'config',
        source_lang: sourceLangSelect.value,
        target_lang: targetLangSelect.value,
        environment: envSelect.value
    }));
    hdrSourceLang.innerText = sourceLangSelect.options[sourceLangSelect.selectedIndex].text;
    hdrTargetLang.innerText = targetLangSelect.options[targetLangSelect.selectedIndex].text;
}

sourceLangSelect.addEventListener('change', sendConfig);
targetLangSelect.addEventListener('change', sendConfig);
envSelect.addEventListener('change', sendConfig);

// ── Speaker count helper ──────────────────────────────────────────────────────
function updateSpeakerCount() {
    const countEl = document.getElementById('speakerCount');
    if (!countEl) return;
    const uniqueSpeakers = new Set();
    document.querySelectorAll('.t-card').forEach(card => {
        const spkId = card.dataset.speakerId;
        if (spkId) uniqueSpeakers.add(spkId);
    });
    countEl.innerText = uniqueSpeakers.size || 1;
}

// ── Transcript cards ───────────────────────────────────────────────────────────
function handleSubtitle(data) {
    if (emptyState) emptyState.style.display = 'none';

    if (!currentLiveCard || currentLiveCard.dataset.utteranceId !== data.utterance_id) {
        if (currentLiveCard) currentLiveCard.classList.remove('live-card');

        currentLiveCard = document.createElement('div');
        currentLiveCard.className = 't-card live-card';
        // data-utterance-id is the single ID used by handleSubtitleUpdate
        // to locate this card. No second copy needed.
        currentLiveCard.dataset.utteranceId = data.utterance_id || '';

        const now = new Date();
        const timeStr = [now.getHours(), now.getMinutes(), now.getSeconds()]
            .map(n => String(n).padStart(2, '0')).join(':');

        // Speaker label comes from the backend's speaker ID service.
        // "unknown" means either nobody has self-introduced yet in this
        // meeting, or this utterance didn't clear the match threshold.
        const speakerId = data.speaker_id || 'unknown';
        const speakerName = data.speaker && data.speaker !== 'unknown'
            ? data.speaker
            : 'Unknown Speaker';
        // Badge initial: first letter of the real name, or "?" for unknown.
        const speakerInitial = (data.speaker && data.speaker !== 'unknown')
            ? data.speaker.charAt(0).toUpperCase()
            : '?';

        currentLiveCard.dataset.speakerId = speakerId;
        currentLiveCard.dataset.speaker = speakerName;

        currentLiveCard.innerHTML = `
            <div class="t-time">${timeStr}</div>
            <div class="t-content">
                <div class="t-speaker">
                    <div class="spk-badge">${speakerInitial}</div>
                    <span class="spk-name">${speakerName}</span>
                    <button class="edit-spk-btn" title="Edit speaker name"><i class="fa-solid fa-pen"></i></button>
                    <span class="lang-tag">${data.src_lang || 'Auto'}</span>
                </div>
                <div class="t-source source-text">${data.source_text || '...'}</div>
                <div class="t-target target-text">${data.text || '...'}</div>
                <div class="t-metrics">
                    <span class="metric-item"><i class="fa-solid fa-microphone"></i> ASR: <strong class="metric-asr">${data.stt_time_ms || 0}ms</strong></span>
                    <span class="metric-separator">|</span>
                    <span class="metric-item"><i class="fa-solid fa-language"></i> NMT: <strong class="metric-nmt">${data.trans_time_ms || 0}ms</strong></span>
                    <span class="metric-separator">|</span>
                    <span class="metric-item"><i class="fa-solid fa-bolt"></i> Total: <strong class="metric-total">${data.total_time_ms || 0}ms</strong></span>
                </div>
            </div>`;
        transcriptArea.appendChild(currentLiveCard);

        // ── Click pencil icon to rename speaker ─────────────────────────────
        const editBtn = currentLiveCard.querySelector('.edit-spk-btn');
        editBtn.addEventListener('click', () => {
            const card = editBtn.closest('.t-card');
            const spkId = card?.dataset.speakerId;
            const oldName = card?.dataset.speaker;
            if (!spkId || spkId === 'unknown' || !oldName || oldName === 'Unknown Speaker') return;
            const newName = prompt(`Rename speaker "${oldName}" to:`, oldName);
            if (newName && newName.trim() && newName.trim() !== oldName) {
                ws.send(JSON.stringify({
                    action: 'rename_speaker',
                    speaker_id: spkId,
                    new_name: newName.trim()
                }));
            }
        });
    } else {
        if (data.speaker_id) currentLiveCard.dataset.speakerId = data.speaker_id;
        if (data.speaker) {
            currentLiveCard.dataset.speaker = data.speaker;
            const nameEl = currentLiveCard.querySelector('.spk-name');
            if (nameEl) nameEl.innerText = data.speaker;
            const badgeEl = currentLiveCard.querySelector('.spk-badge');
            if (badgeEl) badgeEl.innerText = data.speaker.charAt(0).toUpperCase();
        }
        currentLiveCard.querySelector('.source-text').innerText = data.source_text || '...';
        currentLiveCard.querySelector('.target-text').innerText = data.text || '...';
        if (data.src_lang) currentLiveCard.querySelector('.lang-tag').innerText = data.src_lang;
        if (data.stt_time_ms) currentLiveCard.querySelector('.metric-asr').innerText = data.stt_time_ms + 'ms';
        if (data.trans_time_ms) currentLiveCard.querySelector('.metric-nmt').innerText = data.trans_time_ms + 'ms';
        if (data.total_time_ms) currentLiveCard.querySelector('.metric-total').innerText = data.total_time_ms + 'ms';
    }

    updateSpeakerCount();

    if (autoScrollToggle.checked) transcriptArea.scrollTop = transcriptArea.scrollHeight;
}

/**
 * Handles subtitle_update: replaces the draft translation on an existing card
 * with the more accurate window-based translation. Uses a brief CSS fade
 * so the update feels smooth rather than jarring.
 */
function handleSubtitleUpdate(data) {
    if (!data.text || !data.utterance_id) return;

    // Find the card created for this utterance_id
    const card = transcriptArea.querySelector(
        `[data-utterance-id="${data.utterance_id}"]`
    );
    if (!card) return;  // card was cleared or doesn't exist — safe to ignore

    const textEl = card.querySelector('.target-text');
    if (!textEl) return;

    // Replace text with the accurate translation + brief fade animation
    textEl.classList.add('text-updating');
    textEl.innerText = data.text;
    textEl.addEventListener(
        'animationend',
        () => textEl.classList.remove('text-updating'),
        { once: true }
    );

    // Update the latency metrics
    const nmtEl = card.querySelector('.metric-nmt');
    const totalEl = card.querySelector('.metric-total');
    if (nmtEl && data.trans_time_ms) nmtEl.innerText = data.trans_time_ms + 'ms';
    if (totalEl && data.total_time_ms) totalEl.innerText = data.total_time_ms + 'ms';
}

function handleDone(data) {
    if (!currentLiveCard) return;
    currentLiveCard.classList.remove('live-card');

    const srcText = currentLiveCard.querySelector('.source-text').innerText;
    const tgtText = currentLiveCard.querySelector('.target-text').innerText;
    const timeStr = currentLiveCard.querySelector('.t-time').innerText;
    const speakerName = currentLiveCard.dataset.speaker || 'Unknown Speaker';
    fullTranscriptText += `[${timeStr}] ${speakerName}\nSource: ${srcText}\nTranslated: ${tgtText}\n\n`;

    const words = srcText.split(/\s+/).filter(w => w.length > 0).length;
    totalWords += words;
    statWords.innerText = totalWords;

    if (data.latency_ms) {
        latencies.push(data.latency_ms);
        const avg = latencies.reduce((a, b) => a + b, 0) / latencies.length;
        statLatency.innerText = (avg / 1000).toFixed(2) + 's';
    }

    dbStatusEl.classList.add('syncing');
    dbStatusText.innerText = 'Saved to DB';
    setTimeout(() => {
        dbStatusEl.classList.remove('syncing');
        dbStatusText.innerText = 'Synced';
    }, 2000);

    currentLiveCard = null;
    updateSpeakerCount();
}

/**
 * Handles speaker_renamed: updates all transcript cards that had the speaker
 * ID with the new one — badges, labels, and data attributes.
 */
function handleSpeakerRenamed(data) {
    if (!data.success || !data.speaker_id || !data.new_name) return;

    const speakerId = data.speaker_id;
    const newName = data.new_name;
    const newInitial = newName.charAt(0).toUpperCase();

    // Update every card that has this speaker ID
    transcriptArea.querySelectorAll('.t-card').forEach(card => {
        if (card.dataset.speakerId === speakerId) {
            const oldName = card.dataset.speaker;
            card.dataset.speaker = newName;
            const nameEl = card.querySelector('.spk-name');
            const badgeEl = card.querySelector('.spk-badge');
            if (nameEl) nameEl.innerText = newName;
            if (badgeEl) badgeEl.innerText = newInitial;

            // Also fix the in-memory transcript export text
            if (oldName) {
                fullTranscriptText = fullTranscriptText.replaceAll(oldName, newName);
            }
        }
    });
    updateSpeakerCount();
}

/**
 * Handles speakers_merged: updates all transcript cards that had the source
 * speaker ID to the target speaker ID and target speaker name.
 */
function handleSpeakersMerged(data) {
    if (!data.success || !data.source_id || !data.target_id || !data.target_name) return;

    const sourceId = data.source_id;
    const targetId = data.target_id;
    const targetName = data.target_name;
    const targetInitial = targetName.charAt(0).toUpperCase();

    // Update every card that has the source speaker ID
    transcriptArea.querySelectorAll('.t-card').forEach(card => {
        if (card.dataset.speakerId === sourceId) {
            const oldName = card.dataset.speaker;
            card.dataset.speakerId = targetId;
            card.dataset.speaker = targetName;
            const nameEl = card.querySelector('.spk-name');
            const badgeEl = card.querySelector('.spk-badge');
            if (nameEl) nameEl.innerText = targetName;
            if (badgeEl) badgeEl.innerText = targetInitial;

            if (oldName) {
                fullTranscriptText = fullTranscriptText.replaceAll(oldName, targetName);
            }
        }
    });
    updateSpeakerCount();
}

// ── Recording ─────────────────────────────────────────────────────────────────
async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: { channelCount: 1, sampleRate: 16000 }
        });

        // --- Audio Context & Analyser ---
        // Do NOT force 16000 here — browsers silently ignore it and the analyser gets zero data.
        // We let the browser run its native rate (44100/48000) for the visualizer,
        // and resample to 16000 only when sending bytes to the backend.
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        if (audioContext.state === 'suspended') await audioContext.resume();

        const source = audioContext.createMediaStreamSource(stream);

        analyser = audioContext.createAnalyser();
        analyser.fftSize = 64;
        analyser.smoothingTimeConstant = 0.75;
        dataArray = new Uint8Array(analyser.frequencyBinCount); // 32 bins

        // Must connect into the graph or Chrome suspends the analyser
        const silentGain = audioContext.createGain();
        silentGain.gain.value = 0;
        source.connect(analyser);
        analyser.connect(silentGain);
        silentGain.connect(audioContext.destination);

        // --- Resampling ScriptProcessor: downsample to 16000 for the backend ---
        const processor = audioContext.createScriptProcessor(4096, 1, 1);
        source.connect(processor);
        processor.connect(audioContext.destination);

        const nativeRate = audioContext.sampleRate;  // e.g. 48000
        const targetRate = 16000;
        const ratio = nativeRate / targetRate;   // e.g. 3.0

        processor.onaudioprocess = (e) => {
            if (!isRecording || !ws || ws.readyState !== WebSocket.OPEN) return;
            const input = e.inputBuffer.getChannelData(0);
            // Simple decimation: pick every `ratio`-th sample
            const outputLen = Math.floor(input.length / ratio);
            const output = new Float32Array(outputLen);
            for (let i = 0; i < outputLen; i++) {
                output[i] = input[Math.round(i * ratio)];
            }
            ws.send(output.buffer);
        };

        window._micStream = stream;
        window._processor = processor;

        // Set state BEFORE starting animation loop
        isRecording = true;
        micBtn.classList.add('recording');
        micStatusText.innerText = 'Listening...';
        if (!timerInterval) timerInterval = setInterval(updateTimer, 1000);

        // Now start the animation — isRecording is true so the loop won't exit immediately
        drawWaveform();

    } catch (err) {
        console.error('Microphone error:', err);
        alert('Microphone access denied or unavailable: ' + err.message);
    }
}

function stopRecording() {
    isRecording = false;
    micBtn.classList.remove('recording');
    micStatusText.innerText = 'Paused';

    cancelAnimationFrame(animationId);
    animationId = null;
    // Resume gentle idle animation
    startIdleAnimation();

    clearInterval(timerInterval);
    timerInterval = null;

    window._processor?.disconnect();
    window._processor = null;
    window._micStream?.getTracks().forEach(t => t.stop());
    window._micStream = null;

    audioContext?.close();
    audioContext = null;
    analyser = null;
}

micBtn.addEventListener('click', async () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        connectWebSocket();
        // Give WebSocket time to open before recording
        await new Promise(r => setTimeout(r, 600));
    }
    if (isRecording) stopRecording();
    else startRecording();
});

// ── Export ────────────────────────────────────────────────────────────────────
exportTxtBtn.addEventListener('click', () => {
    if (!fullTranscriptText) { alert('No transcript yet.'); return; }
    const blob = new Blob([fullTranscriptText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = Object.assign(document.createElement('a'), { href: url, download: `ispeak_${Date.now()}.txt` });
    a.click();
    URL.revokeObjectURL(url);
});

// ── Summarize Meeting ─────────────────────────────────────────────────────────
const summarizeBtn = document.getElementById('summarizeBtn');
const summaryPanel = document.getElementById('summaryPanel');
const summaryText = document.getElementById('summaryText');
const copySummaryBtn = document.getElementById('copySummaryBtn');
const downloadSummaryBtn = document.getElementById('downloadSummaryBtn');

// New Tab Elements
const tabLiveTranscript = document.getElementById('tabLiveTranscript');
const tabMeetingSummary = document.getElementById('tabMeetingSummary');
const transcriptToolbar = document.getElementById('transcriptToolbar');
const mainSummaryArea = document.getElementById('mainSummaryArea');

const summaryTabPlaceholder = document.getElementById('summaryTabPlaceholder');
const summaryReportLayout = document.getElementById('summaryReportLayout');
const mainSummaryContent = document.getElementById('mainSummaryContent');
const generateSummaryTabBtn = document.getElementById('generateSummaryTabBtn');
const copySummaryTabBtn = document.getElementById('copySummaryTabBtn');
const downloadSummaryTabBtn = document.getElementById('downloadSummaryTabBtn');

let rawSummaryMarkdown = '';

// A lightweight markdown to HTML converter for professional layout
function parseSummaryMarkdown(markdown) {
    const lines = markdown.split('\n');
    let html = '';
    let inList = false;

    for (let i = 0; i < lines.length; i++) {
        let line = lines[i].trimEnd();
        
        // Handle Headers
        if (line.startsWith('### ')) {
            if (inList) { html += '</ul>'; inList = false; }
            html += `<h3>${parseInlineMarkdown(line.slice(4))}</h3>`;
            continue;
        }
        if (line.startsWith('## ') || line.startsWith('# ')) {
            if (inList) { html += '</ul>'; inList = false; }
            const headerText = line.startsWith('## ') ? line.slice(3) : line.slice(2);
            html += `<h2>${parseInlineMarkdown(headerText)}</h2>`;
            continue;
        }

        // Handle List Items
        const listMatch = line.match(/^(\s*)([*+-])\s+(.*)$/);
        if (listMatch) {
            const indent = listMatch[1].length;
            const content = parseInlineMarkdown(listMatch[3]);
            if (!inList) {
                html += '<ul>';
                inList = true;
            }
            if (indent > 0) {
                html += `<li class="nested-li">${content}</li>`;
            } else {
                html += `<li>${content}</li>`;
            }
            continue;
        }

        // Close list if we hit a blank line or paragraph
        if (line.trim() === '') {
            if (inList) { html += '</ul>'; inList = false; }
            continue;
        }

        // Default Paragraph
        if (inList) { html += '</ul>'; inList = false; }
        html += `<p>${parseInlineMarkdown(line.trim())}</p>`;
    }

    if (inList) { html += '</ul>'; }
    return html;
}

function parseInlineMarkdown(text) {
    // Convert **bold** to <strong>bold</strong>
    return text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
}

// Unified function to request summarization and update UI
async function executeSummarization() {
    if (!currentMeetingId) { alert('No active meeting to summarize.'); return; }

    // Set loading state on both buttons
    summarizeBtn.disabled = true;
    generateSummaryTabBtn.disabled = true;
    const origSidebarHTML = summarizeBtn.innerHTML;
    const origTabHTML = generateSummaryTabBtn.innerHTML;

    summarizeBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin text-purple"></i> <span>Summarizing...</span>';
    generateSummaryTabBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin text-purple"></i> Summarizing...';

    try {
        const response = await fetch(`/api/meetings/${currentMeetingId}/summarize`, { method: 'POST' });
        const data = await response.json();

        if (data.error) {
            alert('Summarization failed: ' + data.error);
        } else if (data.summary) {
            rawSummaryMarkdown = data.summary;
            const htmlContent = parseSummaryMarkdown(data.summary);

            // Update sidebar widget
            summaryText.innerHTML = htmlContent;
            summaryPanel.style.display = 'block';

            // Update main tab page
            mainSummaryContent.innerHTML = htmlContent;
            summaryTabPlaceholder.style.display = 'none';
            summaryReportLayout.style.display = 'flex';
        }
    } catch (err) {
        alert('Summarization request failed: ' + err.message);
    } finally {
        summarizeBtn.disabled = false;
        generateSummaryTabBtn.disabled = false;
        summarizeBtn.innerHTML = origSidebarHTML;
        generateSummaryTabBtn.innerHTML = origTabHTML;
    }
}

summarizeBtn.addEventListener('click', executeSummarization);
generateSummaryTabBtn.addEventListener('click', executeSummarization);

// Tab switching handlers
tabLiveTranscript.addEventListener('click', (e) => {
    e.preventDefault();
    tabLiveTranscript.classList.add('active');
    tabMeetingSummary.classList.remove('active');

    transcriptToolbar.style.display = 'flex';
    transcriptArea.style.display = 'block';
    mainSummaryArea.style.display = 'none';
});

tabMeetingSummary.addEventListener('click', (e) => {
    e.preventDefault();
    tabMeetingSummary.classList.add('active');
    tabLiveTranscript.classList.remove('active');

    transcriptToolbar.style.display = 'none';
    transcriptArea.style.display = 'none';
    mainSummaryArea.style.display = 'flex';

    // Update layout depending on if a summary is already generated
    if (rawSummaryMarkdown) {
        mainSummaryContent.innerHTML = parseSummaryMarkdown(rawSummaryMarkdown);
        summaryTabPlaceholder.style.display = 'none';
        summaryReportLayout.style.display = 'flex';
    } else {
        summaryTabPlaceholder.style.display = 'flex';
        summaryReportLayout.style.display = 'none';
    }
});

// Copy summary actions
function copySummaryToClipboard() {
    if (!rawSummaryMarkdown) return;
    navigator.clipboard.writeText(rawSummaryMarkdown).then(() => {
        const origHTML = copySummaryBtn.innerHTML;
        const origTabHTML = copySummaryTabBtn.innerHTML;

        copySummaryBtn.innerHTML = '<i class="fa-solid fa-check"></i> Copied!';
        copySummaryTabBtn.innerHTML = '<i class="fa-solid fa-check"></i> Copied!';

        setTimeout(() => {
            copySummaryBtn.innerHTML = origHTML;
            copySummaryTabBtn.innerHTML = origTabHTML;
        }, 1500);
    });
}
copySummaryBtn.addEventListener('click', copySummaryToClipboard);
copySummaryTabBtn.addEventListener('click', copySummaryToClipboard);

// Download summary actions
function downloadSummaryFile() {
    if (!rawSummaryMarkdown) { alert('No summary available to download.'); return; }
    const blob = new Blob([rawSummaryMarkdown], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = Object.assign(document.createElement('a'), {
        href: url,
        download: `meeting_summary_${currentMeetingId}.txt`
    });
    a.click();
    URL.revokeObjectURL(url);
}
downloadSummaryBtn.addEventListener('click', downloadSummaryFile);
downloadSummaryTabBtn.addEventListener('click', downloadSummaryFile);


// ── View Mode Toggles ─────────────────────────────────────────────────────────
const viewSingleBtn = document.getElementById('viewSingleBtn');
const viewSplitBtn = document.getElementById('viewSplitBtn');

viewSingleBtn.addEventListener('click', () => {
    transcriptArea.classList.remove('split-view');
    viewSingleBtn.classList.add('active');
    viewSplitBtn.classList.remove('active');
});

viewSplitBtn.addEventListener('click', () => {
    transcriptArea.classList.add('split-view');
    viewSplitBtn.classList.add('active');
    viewSingleBtn.classList.remove('active');
});

// ── Sidebar toggles ───────────────────────────────────────────────────────────
toggleLeftBtn.addEventListener('click', () => appContainer.classList.toggle('hide-left'));
toggleRightBtn.addEventListener('click', () => appContainer.classList.toggle('hide-right'));


// ── Boot ──────────────────────────────────────────────────────────────────────
startIdleAnimation();   // start gentle idle wave immediately
connectWebSocket();
