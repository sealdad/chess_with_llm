/**
 * Chess with LLM — Control Panel
 * Frontend JavaScript
 */

// State
let currentMode = 'battle';
let currentLanguage = 'zh-TW';
let llmModelName = 'AI'; // Updated from /health endpoint
let _awaitingRobotMove = false; // True while AI is thinking — blocks state overwrite
let gameActive = false;
let voiceEnabled = false;
let pollingInterval = null;
let currentTab = 'main';
// (auto-detect is now server-side — no client timer needed)

// Voice state
let voiceWs = null;
let voiceListeningMode = 'push_to_talk';
let voiceStatus = 'idle';  // idle, listening, processing, speaking
let mediaStream = null;
let mediaRecorder = null;
let audioContext = null;
let audioChunks = [];
let ttsAudioQueue = [];
let isTtsPlaying = false;
let voiceOutputEnabled = true;
let currentTtsAudio = null;     // Reference to playing Audio element for interruption
let _streamingMsgId = 0;        // Counter for streaming message IDs
let _currentStreamingEl = null;  // Currently streaming chat message element
let _watchCurrentSender = null;  // Watch mode: current side label for chat
let _gameWsAudioQueue = [];     // Queue of base64 audio blobs from game WS (sentence-level)
let _gameWsAudioPlaying = false; // Whether game WS audio is currently playing

// VAD (Voice Activity Detection) state
let vadAnalyser = null;         // AnalyserNode for volume monitoring
let vadSource = null;           // MediaStreamAudioSourceNode
let vadAudioCtx = null;         // AudioContext for VAD
let vadInterval = null;         // setInterval ID for volume polling
let vadSpeechDetected = false;  // Currently hearing speech
let vadSilenceStart = 0;        // Timestamp when silence began
let vadSpeechStart = 0;         // Timestamp when speech began
let vadRecorder = null;         // MediaRecorder capturing current utterance
let vadChunks = [];             // Audio chunks for current utterance
let VAD_THRESHOLD = 80;         // Volume threshold (0-255 byte frequency range)
let VAD_SILENCE_MS = 1500;      // Silence duration to end utterance
let VAD_MIN_SPEECH_MS = 2000;   // Minimum speech duration to avoid noise
let vadPeakVolume = 0;          // Track peak volume during current utterance

// Simulation mode state
let simulationMode = false;
let simSelectedSquare = null;
let simLegalMoves = [];
let simHumanColor = 'white'; // the color the human plays
let simLastMove = null; // {from: 'e2', to: 'e4'}
let simIsCheck = false; // true when the human's king is in check
let simCachedRenderArgs = null; // cached args for renderSimBoard re-render
let teachHighlightSquares = []; // squares to highlight during teach instruction
let teachHighlightFrom = null;  // source square (arrow start)
let teachHighlightTo = null;    // dest square (arrow end)
let teachDemoGhost = null;      // {piece, fromSq, toSq, progress} for demo animation
let teachInputLocked = false;   // true during teach opening — blocks sim board clicks
let _teachUnlockAfterAudio = false; // deferred unlock: wait for audio queue to drain
let _teachStepIndex = 0;
let _teachTotalSteps = 0;
let _promotionResolve = null;

// Service URLs (auto-detect hostname so the UI works from any machine)
const VISION_URL = window.VISION_URL || `${location.protocol}//${location.hostname}:8001`;
const ROBOT_URL = window.ROBOT_URL || `${location.protocol}//${location.hostname}:8002`;

// DOM Elements
const elements = {
    // Status
    statusAgent: document.getElementById('status-agent'),
    statusVision: document.getElementById('status-vision'),
    statusRobot: document.getElementById('status-robot'),
    connectionStatus: document.getElementById('connection-status'),

    // Game info
    chessBoard: document.getElementById('chess-board'),
    gameStatus: document.getElementById('game-status'),
    whoseTurn: document.getElementById('whose-turn'),
    moveNumber: document.getElementById('move-number'),
    currentMode: document.getElementById('current-mode'),
    lastHumanMove: document.getElementById('last-human-move'),
    lastRobotMove: document.getElementById('last-robot-move'),

    // Buttons
    btnStart: document.getElementById('btn-start'),
    btnStop: document.getElementById('btn-stop'),
    btnHumanMoved: document.getElementById('btn-human-moved'),
    inputManualMove: document.getElementById('input-manual-move'),
    btnSubmitMove: document.getElementById('btn-submit-move'),
    btnSend: document.getElementById('btn-send'),
    btnChatMic: document.getElementById('btn-chat-mic'),
    btnSpeakTest: document.getElementById('btn-speak-test'),

    // Manual test buttons
    btnCameraFrame: document.getElementById('btn-camera-frame'),
    btnDepthFrame: document.getElementById('btn-depth-frame'),
    btnDetectApriltag: document.getElementById('btn-detect-apriltag'),
    btnCaptureBoard: document.getElementById('btn-capture-board'),
    // Live Streaming (Main View) buttons removed from Vision tab
    btnSaveTraining: document.getElementById('btn-save-training'),
    trainingLabel: document.getElementById('training-label'),
    trainingCount: document.getElementById('training-count'),
    btnRobotHome: document.getElementById('btn-robot-home'),
    btnRobotSleep: document.getElementById('btn-robot-sleep'),
    btnRobotWork: document.getElementById('btn-robot-work'),
    btnSetWork: document.getElementById('btn-set-work'),
    btnRobotVision: document.getElementById('btn-robot-vision'),
    btnSetVision: document.getElementById('btn-set-vision'),
    btnRobotCaptureZone: document.getElementById('btn-robot-capture-zone'),
    btnSetCaptureZone: document.getElementById('btn-set-capture-zone'),
    btnRobotPromotionQueen: document.getElementById('btn-robot-promotion-queen'),
    btnSetPromotionQueen: document.getElementById('btn-set-promotion-queen'),
    btnGripperOpen: document.getElementById('btn-gripper-open'),
    btnGripperClose: document.getElementById('btn-gripper-close'),
    btnGetPositions: document.getElementById('btn-get-positions'),
    btnGripperSet: document.getElementById('btn-gripper-set'),
    // Cartesian Jog and Joint Jog removed from Robot tab
    gripperSlider: document.getElementById('gripper-slider'),

    // Vision output
    visionImage: document.getElementById('vision-image'),
    visionOutput: document.getElementById('vision-output'),

    // Robot output
    robotOutput: document.getElementById('robot-output'),

    // Robot connection
    robotConnectionIndicator: document.getElementById('robot-connection-indicator'),
    robotConnectionText: document.getElementById('robot-connection-text'),
    btnRobotConnect: document.getElementById('btn-robot-connect'),
    btnRobotMock: document.getElementById('btn-robot-mock'),
    btnRobotDisconnect: document.getElementById('btn-robot-disconnect'),

    // Waypoint controls
    waypointName: document.getElementById('waypoint-name'),
    waypointList: document.getElementById('waypoint-list'),
    btnSaveWaypoint: document.getElementById('btn-save-waypoint'),
    btnGotoWaypoint: document.getElementById('btn-goto-waypoint'),
    btnDeleteWaypoint: document.getElementById('btn-delete-waypoint'),
    btnRefreshWaypoints: document.getElementById('btn-refresh-waypoints'),

    // Inputs
    chatInput: document.getElementById('chat-input'),
    chatHistory: document.getElementById('chat-history'),
    voiceEnabled: document.getElementById('voice-enabled'),

    // Voice controls
    voiceOutputEnabled: document.getElementById('voice-output-enabled'),
    voiceListeningMode: document.getElementById('voice-listening-mode'),
    btnMic: document.getElementById('btn-mic'),
    voiceStatusIndicator: document.getElementById('voice-status-indicator'),
    voiceTranscript: document.getElementById('voice-transcript'),
};

// API Functions
async function fetchJSON(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: { 'Content-Type': 'application/json' },
            ...options,
        });
        if (!response.ok) {
            const text = await response.text().catch(() => '');
            console.error(`API ${response.status}: ${url}`, text.substring(0, 100));
            return { success: false, error: `Server error (${response.status})` };
        }
        return await response.json();
    } catch (error) {
        console.error(`API error: ${url}`, error);
        return { success: false, error: error.message };
    }
}

async function checkHealth() {
    const health = await fetchJSON('/health');
    if (health.status === 'healthy' || health.status === 'ok' || health.success) {
        elements.statusAgent.classList.add('online');
        elements.connectionStatus.textContent = 'Connected';
        elements.connectionStatus.classList.add('connected');
    } else {
        elements.statusAgent.classList.remove('online');
    }
    // Vision & Robot service status
    if (elements.statusVision) {
        elements.statusVision.classList.toggle('online', health.vision_service === 'healthy');
    }
    if (elements.statusRobot) {
        elements.statusRobot.classList.toggle('online', health.robot_service === 'healthy');
    }
    // Store LLM model name for chat display — only when no game is active,
    // otherwise we'd clobber the engine the user picked for this game (LLM 2/3).
    if (health.llm_model && !gameActive) {
        llmModelName = health.llm_model;
    }
}

async function getGameState() {
    const result = await fetchJSON('/game/state');
    if (result.success && result.game_state) {
        updateGameDisplay(result.game_state);
        return result.game_state;
    }
    return null;
}

async function startGame() {
    // Hide game-over overlay from previous game
    _gameOverDismissed = false;
    const gameOverOverlay = document.getElementById('game-over-overlay');
    if (gameOverOverlay) gameOverOverlay.style.display = 'none';

    const robotColor = document.querySelector('input[name="robot-color"]:checked').value;
    const simulation = document.getElementById('simulation-mode').checked;
    simulationMode = simulation;
    simHumanColor = robotColor === 'black' ? 'white' : 'black';

    if (simulation) {
        document.body.classList.add('simulation-mode');
    }

    // Build config based on mode
    const gameMode = currentMode;
    const config = {
        robot_color: robotColor,
        game_mode: gameMode,
        simulation: simulation,
    };

    if (gameMode === 'battle') {
        const diffSelect = document.getElementById('difficulty-select');
        const gameEngineSelect = document.getElementById('game-engine-select');
        const moveSourceSelect = document.getElementById('move-source-select');
        config.difficulty = diffSelect ? diffSelect.value : 'intermediate';
        // Game tab engine selector takes priority, fallback to Agent tab
        config.move_source = gameEngineSelect ? gameEngineSelect.value : (moveSourceSelect ? moveSourceSelect.value : 'stockfish');
        // Update chat label to show the correct model name
        if (config.move_source === 'llm3') {
            llmModelName = document.getElementById('setting-llm-model-3')?.value || 'LLM 3';
        } else if (config.move_source === 'llm2') {
            llmModelName = document.getElementById('setting-llm-model-2')?.value || 'LLM 2';
        } else if (config.move_source === 'llm') {
            llmModelName = document.getElementById('setting-llm-model')?.value || 'LLM 1';
        } else {
            llmModelName = 'Stockfish';
        }
        // Character: use game-tab selector if set, fallback to agent-tab fields
        const gameCharSelect = document.getElementById('game-character-select');
        if (gameCharSelect && gameCharSelect.value) {
            const opt = gameCharSelect.selectedOptions[0];
            config.character = gameCharSelect.value + (opt?.dataset.desc ? ': ' + opt.dataset.desc : '');
        } else {
            const charInput = document.getElementById('character-input');
            const charDesc = document.getElementById('character-desc');
            const charTitle = charInput ? charInput.value.trim() : '';
            const charDescription = charDesc ? charDesc.value.trim() : '';
            config.character = charTitle + (charDescription ? ': ' + charDescription : '');
        }
    } else if (gameMode === 'teach') {
        const lessonSelect = document.getElementById('lesson-select');
        config.lesson_id = lessonSelect ? lessonSelect.value : '';
        if (!config.lesson_id) {
            addChatMessage('System', 'Please select a lesson first.');
            return;
        }
    } else if (gameMode === 'watch') {
        config.simulation = true;
        config.white_engine = document.getElementById('watch-white-engine')?.value || 'stockfish';
        config.black_engine = document.getElementById('watch-black-engine')?.value || 'stockfish';
        config.white_difficulty = document.getElementById('watch-white-difficulty')?.value || 'intermediate';
        config.black_difficulty = document.getElementById('watch-black-difficulty')?.value || 'intermediate';
        const wwc = document.getElementById('watch-white-character');
        const wbc = document.getElementById('watch-black-character');
        config.white_character = wwc?.value ? wwc.value + (wwc.selectedOptions[0]?.dataset.desc ? ': ' + wwc.selectedOptions[0].dataset.desc : '') : '';
        config.black_character = wbc?.value ? wbc.value + (wbc.selectedOptions[0]?.dataset.desc ? ': ' + wbc.selectedOptions[0].dataset.desc : '') : '';
        config.move_delay = parseFloat(document.getElementById('watch-move-delay')?.value || '3');
    }

    const result = await fetchJSON('/game/start', {
        method: 'POST',
        body: JSON.stringify(config),
    });

    if (result.success) {
        gameActive = true;
        // Show mode-appropriate controls near start/stop
        const moveBar = document.getElementById('move-submit-bar');
        if (gameMode === 'teach') {
            if (moveBar) moveBar.style.display = 'none';
            addChatMessage('System', 'Teach mode started! Follow the lesson instructions.');
            // Fetch initial teach state
            const teachState = await fetchJSON('/teach/state');
            if (teachState.active && teachState.current_step) {
                updateTeachPanel({
                    lesson_title: teachState.lesson_title,
                    step_index: teachState.step_index,
                    total_steps: teachState.total_steps,
                    instruction: teachState.current_step.instruction,
                });
                // Don't add Coach chat here — _teach_3phase_instruction
                // will stream the LLM-generated instruction via voice_text_chunk
            }
        } else if (gameMode === 'watch') {
            if (moveBar) moveBar.style.display = 'none';
            simHumanColor = 'white'; // Board orientation
            addChatMessage('System', 'Watch mode started! AI vs AI.');
        } else {
            if (moveBar) moveBar.style.display = 'flex';
            addChatMessage('System', simulation ? `Game started! You play ${simHumanColor}.` : `Game started! Robot plays ${robotColor}.`);
        }
        startPolling();
        if (!simulation) startMainYoloStream();
        await getGameState();
    } else {
        addChatMessage('System', `Failed to start game: ${result.error || 'Unknown error'}`);
    }
}

async function stopGame() {
    interruptAgent();
    const result = await fetchJSON('/game/stop', { method: 'POST' });
    if (result.success) {
        gameActive = false;
        simulationMode = false;
        simSelectedSquare = null;
        simLegalMoves = [];
        simLastMove = null;
        teachInputLocked = false;
        _teachUnlockAfterAudio = false;
        document.body.classList.remove('simulation-mode');
        stopPolling();
        stopMainYoloStream();
        addChatMessage('System', 'Game stopped.');
        elements.gameStatus.textContent = 'No game';
        // Hide teach panel and move-submit bar
        const panel = document.getElementById('teach-panel');
        if (panel) panel.style.display = 'none';
        const moveBar = document.getElementById('move-submit-bar');
        if (moveBar) moveBar.style.display = 'none';
    }
}

async function notifyHumanMoved() {
    addChatMessage('System', 'Detecting your move...');
    const result = await fetchJSON('/game/human_moved', { method: 'POST', body: JSON.stringify({}) });
    if (result.success) {
        await getGameState();
    } else {
        addChatMessage('System', `Error: ${result.error || 'Failed to notify'}`);
    }
}

async function teachManualDetect() {
    // Manual "I Made My Move" in teach mode — triggers one-shot vision capture
    addChatMessage('System', 'Detecting your move...');
    const detectStatus = document.getElementById('teach-detect-status');
    if (detectStatus) detectStatus.textContent = 'Capturing board...';
    const result = await fetchJSON('/teach/human_moved', { method: 'POST', body: JSON.stringify({}) });
    if (detectStatus) detectStatus.textContent = '';
    if (!result.success && !result.correct) {
        addChatMessage('System', result.error || result.message || 'Detection failed. Try again.');
        return;
    }
    // The result is handled the same as auto-detect — update teach panel
    handleTeachMoveResult(result);
}

async function submitManualMove() {
    const moveStr = (elements.inputManualMove.value || '').trim().toLowerCase();
    if (!moveStr) {
        addChatMessage('System', 'Please enter a move (e.g. e2e4).');
        elements.inputManualMove.focus();
        return;
    }
    elements.btnSubmitMove.disabled = true;
    try {
        // In teach mode, route to teach/check_move
        if (currentMode === 'teach' && gameActive) {
            addChatMessage('You', moveStr);
            const result = await teachCheckMove(moveStr);
            elements.inputManualMove.value = '';
            await getGameState();
        } else {
            addChatMessage('System', `Submitting move: ${moveStr}...`);
            const result = await fetchJSON('/game/submit_move', {
                method: 'POST',
                body: JSON.stringify({ move: moveStr }),
            });
            if (result.success) {
                addChatMessage('System', result.message || `Move ${moveStr} applied.`);
                elements.inputManualMove.value = '';
                await getGameState();
            } else {
                addChatMessage('System', `Error: ${result.error || 'Invalid move'}`);
                elements.inputManualMove.select();
            }
        }
    } catch (e) {
        addChatMessage('System', `Submit failed: ${e}`);
    } finally {
        elements.btnSubmitMove.disabled = false;
    }
}

async function validateBoard() {
    addChatMessage('System', 'Validating board...');
    const result = await fetchJSON('/game/validate_board', { method: 'POST' });
    if (result.success) {
        const status = result.valid ? 'Board looks good' : 'Board has issues';
        let msg = `${status} (${result.piece_count} pieces detected)`;
        if (result.issues && result.issues.length) {
            msg += '\n' + result.issues.join('\n');
        }
        if (result.vision_warnings && result.vision_warnings.length) {
            msg += '\nWarnings: ' + result.vision_warnings.join(', ');
        }
        addChatMessage('System', msg);
    } else {
        addChatMessage('System', `Validation failed: ${result.error || 'Unknown error'}`);
    }
}

async function setMode(mode) {
    currentMode = mode;
    updateModeButtons();

    // Toggle battle/teach/watch config panels
    const battleConfig = document.getElementById('battle-config');
    const teachConfig = document.getElementById('teach-config');
    const watchConfig = document.getElementById('watch-config');
    const gameOptions = document.getElementById('game-options');
    if (battleConfig) battleConfig.style.display = mode === 'battle' ? '' : 'none';
    if (teachConfig) teachConfig.style.display = mode === 'teach' ? '' : 'none';
    if (watchConfig) watchConfig.style.display = mode === 'watch' ? '' : 'none';

    // Hide LLM Black/White + Simulation checkbox in Watch mode (forced sim, no sides)
    if (gameOptions) {
        const radios = gameOptions.querySelectorAll('input[name="robot-color"]');
        const simCheckbox = document.getElementById('simulation-mode');
        radios.forEach(r => { r.closest('label').style.display = mode === 'watch' ? 'none' : ''; });
        if (simCheckbox) {
            simCheckbox.closest('label').style.display = mode === 'watch' ? 'none' : '';
            if (mode === 'watch') simCheckbox.checked = true;
        }
    }

    elements.currentMode.textContent = mode === 'battle' ? 'Battle' : mode === 'teach' ? 'Teach' : 'Watch';
}

let _lessonsData = []; // cached lesson list for event handlers

function _lessonTitle(lesson) {
    if (currentLanguage === 'zh-CN' && lesson.title_zh_cn) return lesson.title_zh_cn;
    if (currentLanguage.includes('zh') && lesson.title_zh) return lesson.title_zh;
    return lesson.title;
}

function _lessonDesc(lesson) {
    if (currentLanguage === 'zh-CN' && lesson.description_zh_cn) return lesson.description_zh_cn;
    if (currentLanguage.includes('zh') && lesson.description_zh) return lesson.description_zh;
    return lesson.description;
}

async function loadLessons() {
    const result = await fetchJSON('/teach/lessons');
    if (!result.lessons) return;
    _lessonsData = result.lessons;

    // Populate both lesson selects (Game tab + Agent tab)
    const selects = [
        document.getElementById('lesson-select'),
        document.getElementById('agent-lesson-list'),
    ].filter(Boolean);

    for (const select of selects) {
        const prevValue = select.value;
        select.innerHTML = '';
        if (_lessonsData.length === 0) {
            select.innerHTML = '<option value="">No lessons available</option>';
            continue;
        }
        const placeholder = document.createElement('option');
        placeholder.value = '';
        placeholder.textContent = currentLanguage.includes('zh') ? '選擇課程...' : 'Select a lesson...';
        select.appendChild(placeholder);

        for (const lesson of _lessonsData) {
            const opt = document.createElement('option');
            opt.value = lesson.lesson_id;
            const diff = lesson.difficulty.charAt(0).toUpperCase() + lesson.difficulty.slice(1);
            opt.textContent = `[${diff}] ${_lessonTitle(lesson)} (${lesson.step_count})`;
            select.appendChild(opt);
        }
        // Restore previous selection if still exists
        if (prevValue && [...select.options].some(o => o.value === prevValue)) {
            select.value = prevValue;
        }
    }

    // Trigger description updates
    _onGameLessonChange();
    _onAgentLessonChange();
}

function _onGameLessonChange() {
    const select = document.getElementById('lesson-select');
    const descEl = document.getElementById('lesson-description');
    if (!select || !descEl) return;
    const sel = _lessonsData.find(l => l.lesson_id === select.value);
    descEl.textContent = sel ? _lessonDesc(sel) : '';
}

function _onAgentLessonChange() {
    const select = document.getElementById('agent-lesson-list');
    const descEl = document.getElementById('agent-lesson-desc');
    if (!select || !descEl) return;
    const sel = _lessonsData.find(l => l.lesson_id === select.value);
    descEl.textContent = sel ? _lessonDesc(sel) : (currentLanguage.includes('zh') ? '選擇課程查看說明。' : 'Select a lesson to see its description.');
}

// ── Lesson Management (Agent Tab) ───────────────────────────────────
let _generatedLesson = null; // Holds unsaved generated lesson

async function generateLesson() {
    const topicInput = document.getElementById('lesson-topic');
    const diffSelect = document.getElementById('lesson-difficulty');
    const detailEl = document.getElementById('agent-lesson-detail');
    if (!detailEl) return;

    const topic = topicInput ? topicInput.value.trim() : '';
    const difficulty = diffSelect ? diffSelect.value : 'beginner';

    detailEl.textContent = 'Generating lesson with AI... (this may take 15-30 seconds)';

    const result = await fetchJSON('/agent/generate_lesson', {
        method: 'POST',
        body: JSON.stringify({ topic, difficulty, language: currentLanguage }),
    });

    if (result.success && result.lesson) {
        _generatedLesson = result.lesson;
        _displayLessonDetail(result.lesson);
        if (topicInput) topicInput.value = result.lesson.title || topic;
    } else {
        let msg = 'Generation failed: ' + (result.error || 'Unknown error');
        if (result.raw) msg += '\n\nRaw LLM response:\n' + result.raw;
        detailEl.textContent = msg;
        _generatedLesson = null;
    }
}

async function randomLesson() {
    const topicInput = document.getElementById('lesson-topic');
    if (topicInput) topicInput.value = '';
    await generateLesson();
}

async function saveGeneratedLesson() {
    if (!_generatedLesson) {
        const detailEl = document.getElementById('agent-lesson-detail');
        if (detailEl) detailEl.textContent = 'Nothing to save. Generate a lesson first.';
        return;
    }

    const result = await fetchJSON('/agent/save_lesson', {
        method: 'POST',
        body: JSON.stringify({ lesson: _generatedLesson }),
    });

    if (result.success) {
        _generatedLesson = null;
        await loadLessons(); // Reload both dropdowns
        const detailEl = document.getElementById('agent-lesson-detail');
        if (detailEl) detailEl.textContent = 'Lesson saved: ' + result.lesson_id;
    }
}

async function deleteSelectedLesson() {
    const select = document.getElementById('agent-lesson-list');
    if (!select || !select.value) return;
    const lessonId = select.value;

    const result = await fetchJSON(`/agent/lesson/${encodeURIComponent(lessonId)}`, {
        method: 'DELETE',
    });

    if (result.success) {
        await loadLessons();
        const detailEl = document.getElementById('agent-lesson-detail');
        if (detailEl) detailEl.textContent = 'Lesson deleted.';
    }
}

async function showLessonDetail(lessonId) {
    const detailEl = document.getElementById('agent-lesson-detail');
    if (!detailEl || !lessonId) {
        if (detailEl) detailEl.textContent = 'Select a lesson or generate a new one.';
        return;
    }

    const result = await fetchJSON(`/teach/lesson/${encodeURIComponent(lessonId)}`);
    if (result.lesson_id) {
        _displayLessonDetail(result);
        // Also fill topic/difficulty fields
        const topicInput = document.getElementById('lesson-topic');
        const diffSelect = document.getElementById('lesson-difficulty');
        if (topicInput) topicInput.value = result.title || '';
        if (diffSelect) diffSelect.value = result.difficulty || 'beginner';
    } else {
        detailEl.textContent = 'Could not load lesson details.';
    }
}

function _displayLessonDetail(lesson) {
    const detailEl = document.getElementById('agent-lesson-detail');
    if (!detailEl) return;

    const zh = currentLanguage.includes('zh');
    const title = (zh && lesson.title_zh) ? lesson.title_zh : lesson.title;
    const desc = (zh && lesson.description_zh) ? lesson.description_zh : lesson.description;
    const steps = lesson.steps || [];

    let text = `${title}\n${desc}\nDifficulty: ${lesson.difficulty}\n${steps.length} steps\n`;
    text += '─'.repeat(40) + '\n';

    steps.forEach((s, i) => {
        const instr = (zh && s.instruction_zh) ? s.instruction_zh : s.instruction;
        const expl = (zh && s.explanation_zh) ? s.explanation_zh : s.explanation;
        text += `\nStep ${i + 1}: ${s.expected_move || '(any)'}\n`;
        text += `  ${instr}\n`;
        if (expl) text += `  → ${expl}\n`;
    });

    detailEl.textContent = text;
}

// ── Battle Character Management ─────────────────────────────────────
async function loadCharacters() {
    const result = await fetchJSON(`/agent/characters?language=${encodeURIComponent(currentLanguage)}`);
    const chars = result.characters || [];

    // Populate all character selects
    const selects = [
        { el: document.getElementById('character-select'), placeholder: '-- New Character --' },
        { el: document.getElementById('game-character-select'), placeholder: 'Default Character' },
        { el: document.getElementById('watch-white-character'), placeholder: 'Default' },
        { el: document.getElementById('watch-black-character'), placeholder: 'Default' },
    ];
    for (const { el, placeholder } of selects) {
        if (!el) continue;
        const prevValue = el.value;
        el.innerHTML = '';
        const ph = document.createElement('option');
        ph.value = '';
        ph.textContent = placeholder;
        el.appendChild(ph);
        for (const c of chars) {
            const opt = document.createElement('option');
            opt.value = c.title;
            opt.textContent = c.title;
            opt.dataset.desc = c.description || '';
            el.appendChild(opt);
        }
        if (prevValue && [...el.options].some(o => o.value === prevValue)) el.value = prevValue;
    }
}

function onCharacterSelected() {
    const select = document.getElementById('character-select');
    const titleInput = document.getElementById('character-input');
    const descArea = document.getElementById('character-desc');
    if (!select || !titleInput || !descArea) return;
    const opt = select.selectedOptions[0];
    if (select.value === '') {
        titleInput.value = '';
        descArea.value = '';
    } else {
        titleInput.value = select.value;
        descArea.value = opt ? (opt.dataset.desc || '') : '';
    }
    _updateCharCount();
    // Save selected character with language
    localStorage.setItem('battle_character', JSON.stringify({
        title: titleInput.value,
        description: descArea.value,
        language: currentLanguage,
    }));
}

async function saveCharacter() {
    const titleInput = document.getElementById('character-input');
    const descArea = document.getElementById('character-desc');
    if (!titleInput) return;
    const title = titleInput.value.trim();
    if (!title) return;
    await fetchJSON('/agent/characters/save', {
        method: 'POST',
        body: JSON.stringify({
            language: currentLanguage,
            title: title,
            description: (descArea ? descArea.value : '').substring(0, 200),
        }),
    });
    await loadCharacters();
    // Re-select the saved character
    const select = document.getElementById('character-select');
    if (select) select.value = title;
}

async function deleteCharacter() {
    const select = document.getElementById('character-select');
    if (!select || !select.value) return;
    const title = select.value;
    await fetchJSON(`/agent/characters/${encodeURIComponent(title)}?language=${encodeURIComponent(currentLanguage)}`, {
        method: 'DELETE',
    });
    await loadCharacters();
    onCharacterSelected();
}

async function generateCharacterDesc() {
    const titleInput = document.getElementById('character-input');
    const descArea = document.getElementById('character-desc');
    if (!titleInput || !descArea) return;
    const title = titleInput.value.trim();
    if (!title) { descArea.value = ''; return; }

    descArea.value = 'Generating...';
    const result = await fetchJSON('/agent/generate_character', {
        method: 'POST',
        body: JSON.stringify({ title: title }),
    });
    if (result.description) {
        descArea.value = result.description.substring(0, 200);
    } else {
        descArea.value = `A ${title.toLowerCase()} who plays chess with attitude.`;
    }
    _updateCharCount();
}

async function randomCharacter() {
    const titleInput = document.getElementById('character-input');
    const descArea = document.getElementById('character-desc');
    if (!titleInput || !descArea) return;

    titleInput.value = '...';
    descArea.value = 'Generating...';
    const result = await fetchJSON('/agent/random_character', { method: 'POST' });
    if (result.title && result.description) {
        titleInput.value = result.title;
        descArea.value = result.description.substring(0, 200);
    } else {
        // Fallback random characters
        const chars = [
            { t: 'Zen Master', d: 'Calm and wise. Speaks in metaphors about life and chess. Never gets angry, always finds peace in any position.' },
            { t: 'Sarcastic Pirate', d: 'Arrr! Mocks every move with pirate slang. Celebrates captures like finding treasure. Calls pawns "little scallywags".' },
            { t: 'Dramatic Opera Singer', d: 'Every move is a grand performance! Narrates the game like an Italian opera. Gasps at captures, weeps at losses.' },
            { t: 'Robot Butler', d: 'Extremely formal and polite. Addresses the player as "sir" or "madam". Apologizes before capturing pieces.' },
            { t: 'Street Rapper', d: 'Drops rhymes about every move. Trash talks with flow. Celebrates checkmate with a freestyle verse.' },
        ];
        const pick = chars[Math.floor(Math.random() * chars.length)];
        titleInput.value = pick.t;
        descArea.value = pick.d;
    }
    _updateCharCount();
}

function _updateCharCount() {
    const descArea = document.getElementById('character-desc');
    const countEl = document.getElementById('character-desc-count');
    if (descArea && countEl) {
        countEl.textContent = `${descArea.value.length} / 200`;
    }
}

async function teachHint() {
    const result = await fetchJSON('/teach/hint', { method: 'POST' });
    const feedback = document.getElementById('teach-feedback');
    if (feedback && result.message) {
        feedback.textContent = result.message;
        addChatMessage('Coach', result.message);
    }
}

async function teachNextStep() {
    // Disable button while processing
    const btnNext = document.getElementById('btn-teach-next');
    if (btnNext) btnNext.disabled = true;

    const result = await fetchJSON('/teach/next_step', { method: 'POST' });
    if (result.complete) {
        addChatMessage('Coach', result.message);
        const panel = document.getElementById('teach-panel');
        if (panel) panel.style.display = 'none';
        return;
    }
    // Don't add Coach chat — _teach_3phase_instruction streams instruction via voice
    updateTeachPanel(result);
    await getGameState();
}

async function teachStop() {
    await fetchJSON('/teach/stop', { method: 'POST' });
    // Same cleanup as stopGame
    gameActive = false;
    simulationMode = false;
    simSelectedSquare = null;
    simLegalMoves = [];
    simLastMove = null;
    teachInputLocked = false;
    _teachUnlockAfterAudio = false;
    _teachStepIndex = 0;
    _teachTotalSteps = 0;
    document.body.classList.remove('simulation-mode');
    stopPolling();
    stopMainYoloStream();
    elements.gameStatus.textContent = 'No game';
    const panel = document.getElementById('teach-panel');
    if (panel) panel.style.display = 'none';
    const moveBar = document.getElementById('move-submit-bar');
    if (moveBar) moveBar.style.display = 'none';
}

async function teachCheckMove(move) {
    const result = await fetchJSON('/teach/check_move', {
        method: 'POST',
        body: JSON.stringify({ move: move }),
    });
    handleTeachMoveResult(result);
    return result;
}

function handleTeachMoveResult(result) {
    // Shared handler for auto-detect and manual move check results
    var feedback = document.getElementById('teach-feedback');
    var conclusion = document.getElementById('teach-conclusion');
    var btnNext = document.getElementById('btn-teach-next');
    var btnHint = document.getElementById('btn-teach-hint');
    var btnMoved = document.getElementById('btn-teach-moved');
    var detectStatus = document.getElementById('teach-detect-status');

    if (result.message && !result.correct) {
        // Only show Coach message for wrong moves — correct move conclusion
        // is already streamed via voice_text_chunk as "Robot" message
        addChatMessage('Coach', result.message);
    }

    if (result.correct) {
        // Clear teach highlights — step done
        teachHighlightSquares = [];
        teachHighlightFrom = null;
        teachHighlightTo = null;
        teachInputLocked = false;
        // Step completed — show conclusion
        if (feedback) feedback.textContent = '';
        if (conclusion) {
            conclusion.textContent = result.explanation || result.message || 'Correct!';
            conclusion.style.display = '';
        }
        if (btnHint) btnHint.style.display = 'none';
        if (btnMoved) btnMoved.style.display = 'none';
        if (detectStatus) detectStatus.textContent = '';
        if (result.detected_move) {
            addChatMessage('System', 'Move detected: ' + result.detected_move);
        }
        // Last step — auto-advance to lesson complete; otherwise show Next Step
        if (_teachStepIndex + 1 >= _teachTotalSteps) {
            if (btnNext) btnNext.style.display = 'none';
            teachNextStep();
        } else {
            if (btnNext) { btnNext.style.display = ''; btnNext.disabled = false; }
        }
    } else {
        // Wrong move — guide student
        if (feedback) feedback.textContent = result.message || 'Not quite right. Try again!';
        if (detectStatus) detectStatus.textContent = 'Waiting for your move...';
    }
}

function updateTeachPanel(data) {
    var panel = document.getElementById('teach-panel');
    if (!panel) return;

    panel.style.display = '';
    var title = document.getElementById('teach-title');
    var stepInfo = document.getElementById('teach-step-info');
    var instruction = document.getElementById('teach-instruction');
    var feedback = document.getElementById('teach-feedback');
    var conclusion = document.getElementById('teach-conclusion');
    var btnNext = document.getElementById('btn-teach-next');
    var btnHint = document.getElementById('btn-teach-hint');
    var btnMoved = document.getElementById('btn-teach-moved');
    var detectStatus = document.getElementById('teach-detect-status');

    _teachStepIndex = data.step_index || 0;
    _teachTotalSteps = data.total_steps || 0;
    if (title && data.lesson_title) title.textContent = data.lesson_title;
    if (stepInfo) stepInfo.textContent = 'Step ' + (_teachStepIndex + 1) + ' / ' + (_teachTotalSteps || '?');
    if (instruction && data.instruction) instruction.textContent = data.instruction;

    // Reset state for new step — hide conclusion, hide Next
    // Note: do NOT clear teachHighlight* here — they are set by WS teach_highlight
    // event during _teach_demo_move() and would be wiped when this REST response arrives.
    // Highlights are cleared in handleTeachMoveResult (on correct) instead.
    if (feedback) feedback.textContent = '';
    if (conclusion) { conclusion.textContent = ''; conclusion.style.display = 'none'; }
    if (btnNext) { btnNext.style.display = 'none'; btnNext.disabled = false; }
    if (btnHint) btnHint.style.display = '';
    // Show "I Made My Move" in physical mode only
    if (btnMoved) btnMoved.style.display = simulationMode ? 'none' : '';
    if (detectStatus) detectStatus.textContent = simulationMode ? '' : 'Waiting for your move...';
}

async function setLanguage(lang) {
    const result = await fetchJSON('/agent/language', {
        method: 'POST',
        body: JSON.stringify({ language: lang }),
    });

    if (result.success !== false) {
        currentLanguage = lang;
        localStorage.setItem('language', lang);
        updateLanguageButtons();
        // Reload characters and lessons for the new language
        await loadCharacters();
        // Auto-select first character in new language (so fields aren't blank)
        const charSelect = document.getElementById('character-select');
        if (charSelect && charSelect.options.length > 1) {
            charSelect.selectedIndex = 1; // skip "-- New Character --"
        }
        onCharacterSelected();
        await loadLessons();
        _onAgentLessonChange();
        // Notify voice WebSocket of language change
        if (voiceWs && voiceWs.readyState === WebSocket.OPEN) {
            voiceWs.send(JSON.stringify({ type: 'set_language', language: lang }));
        }
        addChatMessage('System', result.greeting || `Language changed to: ${lang}`);
    }
}

function updateLanguageButtons() {
    document.querySelectorAll('.lang-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.lang === currentLanguage);
    });
    const langSelect = document.getElementById('language-select');
    const customLangRow = document.getElementById('custom-lang-row');
    const customLangInput = document.getElementById('custom-language');
    if (langSelect) {
        // Check if current language is one of the preset options
        const isPreset = [...langSelect.options].some(o => o.value === currentLanguage && o.value !== 'custom');
        if (isPreset) {
            langSelect.value = currentLanguage;
            if (customLangRow) customLangRow.style.display = 'none';
        } else {
            langSelect.value = 'custom';
            if (customLangRow) customLangRow.style.display = 'flex';
            if (customLangInput) customLangInput.value = currentLanguage;
        }
    }
}

async function sendChat(message) {
    if (!message.trim()) return;

    addChatMessage('You', message);
    elements.chatInput.value = '';

    const result = await fetchJSON('/agent/chat', {
        method: 'POST',
        body: JSON.stringify({ message: message }),
    });

    // Response text is shown via WS voice_text_chunk (with TTS audio)
    // Don't show raw error strings — the LLM's helpful response is already streamed via WS
    if (result.game_state) {
        updateGameDisplay(result.game_state);
    }
}

async function testTTS() {
    const testText = currentLanguage.includes('zh') ? '你好！準備好下棋了嗎？' : 'Hello! Ready to play chess?';
    const result = await fetchJSON('/agent/chat', {
        method: 'POST',
        body: JSON.stringify({ message: testText }),
    });
    if (!result.text) {
        showToast('TTS test failed: ' + (result.error || 'No response'), 'error', 3000);
    }
}

// Tab Switching
function switchTab(tabName) {
    currentTab = tabName;

    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });

    // Update tab panels
    document.querySelectorAll('.tab-panel').forEach(panel => {
        panel.classList.toggle('active', panel.id === `tab-${tabName}`);
    });
}

// Output Functions
function showVisionResult(text, imageBase64 = null) {
    if (elements.visionOutput) {
        elements.visionOutput.textContent = text;
    }
    if (elements.visionImage && imageBase64) {
        elements.visionImage.src = 'data:image/jpeg;base64,' + imageBase64;
    }
}

function showRobotResult(text) {
    if (elements.robotOutput) {
        elements.robotOutput.textContent = text;
    }
}


async function testCameraFrame() {
    addChatMessage('System', 'Capturing RGB frame...');
    try {
        const response = await fetch(`${VISION_URL}/camera/frame?annotate=true`);
        if (response.ok) {
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            if (elements.visionImage) {
                elements.visionImage.src = url;
            }
            showVisionResult('RGB frame captured successfully');
            addChatMessage('System', 'RGB frame captured.');
        } else {
            showVisionResult(`Error: ${response.status} ${response.statusText}`);
            addChatMessage('System', 'RGB capture failed.');
        }
    } catch (error) {
        showVisionResult(`Error: ${error.message}`);
        addChatMessage('System', `Camera error: ${error.message}`);
    }
}

async function testDepthFrame() {
    addChatMessage('System', 'Capturing depth frame...');
    try {
        const response = await fetch(`${VISION_URL}/camera/depth?colormap=true`);
        if (response.ok) {
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            if (elements.visionImage) {
                elements.visionImage.src = url;
            }
            showVisionResult('Depth frame captured successfully');
            addChatMessage('System', 'Depth frame captured.');
        } else {
            showVisionResult(`Error: ${response.status} ${response.statusText}`);
            addChatMessage('System', 'Depth capture failed.');
        }
    } catch (error) {
        showVisionResult(`Error: ${error.message}`);
        addChatMessage('System', `Depth error: ${error.message}`);
    }
}

function startStreamRgb() {
    addChatMessage('System', 'Starting RGB stream...');
    if (elements.visionImage) {
        elements.visionImage.src = `${VISION_URL}/camera/stream/rgb`;
    }
    showVisionResult('RGB streaming... Click "Stop Stream" to stop.');
}

function startStreamDepth() {
    addChatMessage('System', 'Starting depth stream...');
    if (elements.visionImage) {
        elements.visionImage.src = `${VISION_URL}/camera/stream/depth`;
    }
    showVisionResult('Depth streaming... Click "Stop Stream" to stop.');
}

function stopStream() {
    if (elements.visionImage) {
        elements.visionImage.src = '';
    }
    showVisionResult('Stream stopped.');
    addChatMessage('System', 'Stream stopped.');
}

function startMainYoloStream() {
    if (elements.visionImage) {
        elements.visionImage.src = `${VISION_URL}/camera/stream/yolo`;
    }
}

function stopMainYoloStream() {
    if (elements.visionImage && elements.visionImage.src.includes('/stream/yolo')) {
        elements.visionImage.src = '';
    }
}

// --- Vision tab debug camera ---
function startVcamStream(type) {
    const img = document.getElementById('vcam-image');
    if (!img) return;
    if (type === 'rgb') img.src = `${VISION_URL}/camera/stream/rgb`;
    else if (type === 'depth') img.src = `${VISION_URL}/camera/stream/depth`;
    else if (type === 'yolo') img.src = `${VISION_URL}/camera/stream/yolo`;
}

function stopVcamStream() {
    const img = document.getElementById('vcam-image');
    if (img) img.src = '';
}

function startManualChessStream(type) {
    const img = document.getElementById('mc-camera-img');
    if (!img) return;
    if (type === 'yolo') {
        img.src = `${VISION_URL}/camera/stream/yolo`;
    } else {
        img.src = `${VISION_URL}/camera/stream/rgb`;
    }
}

function stopManualChessStream() {
    const img = document.getElementById('mc-camera-img');
    if (img) img.src = '';
}

async function testDetectApriltag() {
    addChatMessage('System', 'Detecting AprilTags...');
    try {
        const response = await fetch(`${VISION_URL}/detect/apriltag?return_image=true`, {
            method: 'POST',
        });
        const result = await response.json();

        if (result.success) {
            const detections = result.detections || [];
            let text = `Found ${detections.length} AprilTag(s)\n\n`;
            detections.forEach(det => {
                text += `Tag ID: ${det.tag_id}\n`;
                text += `  Center: (${det.center[0].toFixed(1)}, ${det.center[1].toFixed(1)})\n`;
                if (det.pose_valid && det.tvec) {
                    text += `  Position: X=${(det.tvec[0]*100).toFixed(1)}cm, Y=${(det.tvec[1]*100).toFixed(1)}cm, Z=${(det.tvec[2]*100).toFixed(1)}cm\n`;
                }
                text += '\n';
            });
            showVisionResult(text, result.image_base64);
            addChatMessage('System', `Detected ${detections.length} AprilTag(s).`);
        } else {
            showVisionResult(`Error: ${result.error}`);
            addChatMessage('System', `AprilTag detection failed: ${result.error}`);
        }
    } catch (error) {
        showVisionResult(`Error: ${error.message}`);
        addChatMessage('System', `AprilTag error: ${error.message}`);
    }
}

async function testCaptureBoard() {
    addChatMessage('System', 'Capturing board state...');
    try {
        const response = await fetch(`${VISION_URL}/capture`, {
            method: 'POST',
        });
        const result = await response.json();

        if (result.success) {
            let text = `FEN: ${result.fen}\n\n`;
            text += `Valid: ${result.is_valid}\n`;
            if (result.warnings && result.warnings.length > 0) {
                text += `Warnings: ${result.warnings.join(', ')}\n`;
            }
            text += `\n${result.ascii_board || ''}`;
            showVisionResult(text);
            addChatMessage('System', 'Board captured successfully.');
        } else {
            showVisionResult(`Error: ${result.error}`);
            addChatMessage('System', `Board capture failed: ${result.error}`);
        }
    } catch (error) {
        showVisionResult(`Error: ${error.message}`);
        addChatMessage('System', `Board capture error: ${error.message}`);
    }
}

// Training Image Capture
async function saveTrainingImage() {
    const label = elements.trainingLabel ? elements.trainingLabel.value.trim() : '';
    const params = label ? `?label=${encodeURIComponent(label)}` : '';

    addChatMessage('System', 'Saving training image...');
    try {
        // Save the image on the server
        const saveResp = await fetch(`${VISION_URL}/capture/save_image${params}`, {
            method: 'POST',
        });
        const result = await saveResp.json();

        if (result.success) {
            // Also fetch the live frame to show in camera view
            try {
                const frameResp = await fetch(`${VISION_URL}/camera/frame`);
                if (frameResp.ok) {
                    const blob = await frameResp.blob();
                    const url = URL.createObjectURL(blob);
                    if (elements.visionImage) {
                        elements.visionImage.src = url;
                    }
                }
            } catch (e) {
                // Preview failed but save succeeded, that's fine
            }

            showVisionResult(`Saved: ${result.filename}\nTotal images: ${result.count}`);
            if (elements.trainingCount) {
                elements.trainingCount.textContent = `Images saved: ${result.count}`;
            }
            addChatMessage('System', `Training image saved: ${result.filename} (total: ${result.count})`);
        } else {
            showVisionResult(`Error: ${result.error}`);
            addChatMessage('System', `Save failed: ${result.error}`);
        }
    } catch (error) {
        showVisionResult(`Error: ${error.message}`);
        addChatMessage('System', `Save error: ${error.message}`);
    }
}

// Robot Connection Functions
function updateRobotConnectionUI(connected, mock = false, connecting = false) {
    if (elements.robotConnectionIndicator) {
        elements.robotConnectionIndicator.classList.remove('connected', 'connecting', 'mock');
        if (connected) {
            elements.robotConnectionIndicator.classList.add('connected');
            if (mock) {
                elements.robotConnectionIndicator.classList.add('mock');
            }
        } else if (connecting) {
            elements.robotConnectionIndicator.classList.add('connecting');
        }
    }
    if (elements.robotConnectionText) {
        if (connecting) {
            elements.robotConnectionText.textContent = 'Connecting...';
        } else if (connected) {
            elements.robotConnectionText.textContent = mock ? 'Connected (Mock)' : 'Connected';
        } else {
            elements.robotConnectionText.textContent = 'Disconnected';
        }
    }
}

async function checkRobotConnection() {
    try {
        const response = await fetch(`${ROBOT_URL}/connection`);
        const result = await response.json();
        updateRobotConnectionUI(result.connected, result.mock);
        return result.connected;
    } catch (error) {
        updateRobotConnectionUI(false);
        return false;
    }
}

async function connectRobot(mock = false) {
    const modeText = mock ? 'mock mode' : 'robot';
    addChatMessage('System', `Connecting to ${modeText}...`);
    updateRobotConnectionUI(false, false, true);
    showRobotResult('Connecting to robot...');

    try {
        const response = await fetch(`${ROBOT_URL}/connect`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mock: mock }),
        });
        const result = await response.json();

        updateRobotConnectionUI(result.connected, result.mock);

        if (result.success) {
            showRobotResult(`Connected: ${result.message}`);
            addChatMessage('System', result.message);
        } else {
            showRobotResult(`Connection failed: ${result.error}`);
            addChatMessage('System', `Connection failed: ${result.error}`);
        }
    } catch (error) {
        updateRobotConnectionUI(false);
        showRobotResult(`Connection error: ${error.message}`);
        addChatMessage('System', `Connection error: ${error.message}`);
    }
}

async function disconnectRobot() {
    addChatMessage('System', 'Disconnecting...');
    showRobotResult('Disconnecting (moving to sleep pose)...');

    try {
        const response = await fetch(`${ROBOT_URL}/disconnect`, {
            method: 'POST',
        });
        const result = await response.json();

        updateRobotConnectionUI(result.connected);

        if (result.success) {
            showRobotResult(`Disconnected: ${result.message}`);
            addChatMessage('System', 'Robot disconnected.');
        } else {
            showRobotResult(`Disconnect failed: ${result.error}`);
            addChatMessage('System', `Robot disconnect failed: ${result.error}`);
        }
    } catch (error) {
        showRobotResult(`Disconnect error: ${error.message}`);
        addChatMessage('System', `Robot disconnect error: ${error.message}`);
    }
}

async function testRobotHome() {
    addChatMessage('System', 'Moving robot to home position...');
    try {
        const response = await fetch(`${ROBOT_URL}/arm/home`, {
            method: 'POST',
        });
        const result = await response.json();
        showRobotResult(JSON.stringify(result, null, 2));
        addChatMessage('System', result.success ? 'Robot at home position.' : `Failed: ${result.error}`);
    } catch (error) {
        showRobotResult(`Error: ${error.message}`);
        addChatMessage('System', `Robot error: ${error.message}`);
    }
}

async function testRobotSleep() {
    addChatMessage('System', 'Moving robot to sleep position...');
    try {
        const response = await fetch(`${ROBOT_URL}/arm/sleep`, {
            method: 'POST',
        });
        const result = await response.json();
        showRobotResult(JSON.stringify(result, null, 2));
        addChatMessage('System', result.success ? 'Robot at sleep position.' : `Failed: ${result.error}`);
    } catch (error) {
        showRobotResult(`Error: ${error.message}`);
        addChatMessage('System', `Robot error: ${error.message}`);
    }
}

async function goToWork() {
    addChatMessage('System', 'Moving robot to work position...');
    try {
        const response = await fetch(`${ROBOT_URL}/arm/work`, {
            method: 'POST',
        });
        const result = await response.json();
        showRobotResult(JSON.stringify(result, null, 2));
        addChatMessage('System', result.success ? 'Robot at work position.' : `Failed: ${result.error}`);
    } catch (error) {
        showRobotResult(`Error: ${error.message}`);
        addChatMessage('System', `Robot error: ${error.message}`);
    }
}

async function setWorkPosition() {
    addChatMessage('System', 'Saving current position as work position...');
    try {
        const response = await fetch(`${ROBOT_URL}/arm/work/set`, {
            method: 'POST',
        });
        const result = await response.json();
        showRobotResult(JSON.stringify(result, null, 2));
        addChatMessage('System', result.success ? result.message : `Failed: ${result.error}`);
    } catch (error) {
        showRobotResult(`Error: ${error.message}`);
        addChatMessage('System', `Robot error: ${error.message}`);
    }
}

async function goToVision() {
    addChatMessage('System', 'Moving robot to vision position...');
    try {
        const response = await fetch(`${ROBOT_URL}/arm/vision`, {
            method: 'POST',
        });
        const result = await response.json();
        showRobotResult(JSON.stringify(result, null, 2));
        addChatMessage('System', result.success ? 'Robot at vision position.' : `Failed: ${result.error}`);
    } catch (error) {
        showRobotResult(`Error: ${error.message}`);
        addChatMessage('System', `Robot error: ${error.message}`);
    }
}

async function setVisionPosition() {
    addChatMessage('System', 'Saving current position as vision position...');
    try {
        const response = await fetch(`${ROBOT_URL}/arm/vision/set`, {
            method: 'POST',
        });
        const result = await response.json();
        showRobotResult(JSON.stringify(result, null, 2));
        addChatMessage('System', result.success ? result.message : `Failed: ${result.error}`);
    } catch (error) {
        showRobotResult(`Error: ${error.message}`);
        addChatMessage('System', `Robot error: ${error.message}`);
    }
}

async function goToCaptureZone() {
    addChatMessage('System', 'Moving robot to capture zone...');
    try {
        const response = await fetch(`${ROBOT_URL}/arm/capture_zone`, {
            method: 'POST',
        });
        const result = await response.json();
        showRobotResult(JSON.stringify(result, null, 2));
        addChatMessage('System', result.success ? 'Robot at capture zone.' : `Failed: ${result.error}`);
    } catch (error) {
        showRobotResult(`Error: ${error.message}`);
        addChatMessage('System', `Robot error: ${error.message}`);
    }
}

async function setCaptureZone() {
    addChatMessage('System', 'Saving current position as capture zone...');
    try {
        const response = await fetch(`${ROBOT_URL}/capture_zone/set`, {
            method: 'POST',
        });
        const result = await response.json();
        showRobotResult(JSON.stringify(result, null, 2));
        addChatMessage('System', result.success ? result.message : `Failed: ${result.error}`);
    } catch (error) {
        showRobotResult(`Error: ${error.message}`);
        addChatMessage('System', `Robot error: ${error.message}`);
    }
}

async function goToPromotionQueen() {
    addChatMessage('System', 'Moving robot to promotion queen position...');
    try {
        const response = await fetch(`${ROBOT_URL}/arm/promotion_queen`, {
            method: 'POST',
        });
        const result = await response.json();
        showRobotResult(JSON.stringify(result, null, 2));
        addChatMessage('System', result.success ? 'Robot at promotion queen pos.' : `Failed: ${result.error}`);
    } catch (error) {
        showRobotResult(`Error: ${error.message}`);
        addChatMessage('System', `Robot error: ${error.message}`);
    }
}

async function setPromotionQueen() {
    addChatMessage('System', 'Saving current position as promotion queen position...');
    try {
        const response = await fetch(`${ROBOT_URL}/promotion_queen/set`, {
            method: 'POST',
        });
        const result = await response.json();
        showRobotResult(JSON.stringify(result, null, 2));
        addChatMessage('System', result.success ? result.message : `Failed: ${result.error}`);
    } catch (error) {
        showRobotResult(`Error: ${error.message}`);
        addChatMessage('System', `Robot error: ${error.message}`);
    }
}

async function testGripperOpen() {
    addChatMessage('System', 'Opening gripper...');
    try {
        const response = await fetch(`${ROBOT_URL}/gripper/open`, {
            method: 'POST',
        });
        const result = await response.json();
        showRobotResult(JSON.stringify(result, null, 2));
        addChatMessage('System', result.success ? 'Gripper opened.' : `Failed: ${result.error}`);
    } catch (error) {
        showRobotResult(`Error: ${error.message}`);
        addChatMessage('System', `Gripper error: ${error.message}`);
    }
}

async function testGripperClose() {
    addChatMessage('System', 'Closing gripper...');
    try {
        const response = await fetch(`${ROBOT_URL}/gripper/close`, {
            method: 'POST',
        });
        const result = await response.json();
        showRobotResult(JSON.stringify(result, null, 2));
        addChatMessage('System', result.success ? 'Gripper closed.' : `Failed: ${result.error}`);
    } catch (error) {
        showRobotResult(`Error: ${error.message}`);
        addChatMessage('System', `Gripper error: ${error.message}`);
    }
}

async function getPositions() {
    try {
        const response = await fetch(`${ROBOT_URL}/arm/positions`);
        const result = await response.json();
        if (result.success) {
            let text = 'Joint Positions:\n';
            for (const [joint, pos] of Object.entries(result.joints || {})) {
                const deg = (pos * 180 / Math.PI).toFixed(1);
                text += `  ${joint}: ${pos.toFixed(3)} rad (${deg}°)\n`;
            }
            if (result.ee_pose) {
                text += '\nEnd-Effector Position:\n';
                text += `  X: ${(result.ee_pose.x * 100).toFixed(1)} cm\n`;
                text += `  Y: ${(result.ee_pose.y * 100).toFixed(1)} cm\n`;
                text += `  Z: ${(result.ee_pose.z * 100).toFixed(1)} cm\n`;
            }
            showRobotResult(text);
        } else {
            showRobotResult(`Error: ${result.error}`);
        }
    } catch (error) {
        showRobotResult(`Error: ${error.message}`);
    }
}

// Jog functions - click to move one step
async function jogCartesian(axis, direction) {
    const stepCm = parseFloat(elements.cartesianStep?.value || 1);
    const stepM = (stepCm / 100) * direction;
    try {
        const response = await fetch(`${ROBOT_URL}/arm/jog/cartesian`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ axis: axis, step: stepM }),
        });
        const result = await response.json();
        showRobotResult(result.message || result.error || JSON.stringify(result));
    } catch (error) {
        showRobotResult(`Error: ${error.message}`);
    }
}

async function jogJoint(joint, direction) {
    const stepDeg = parseFloat(elements.jointStep?.value || 5);
    const stepRad = (stepDeg * Math.PI / 180) * direction;
    try {
        const response = await fetch(`${ROBOT_URL}/arm/jog/joint`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ joint: joint, step: stepRad }),
        });
        const result = await response.json();
        showRobotResult(result.message || result.error || JSON.stringify(result));
    } catch (error) {
        showRobotResult(`Error: ${error.message}`);
    }
}

async function setGripper() {
    const pos = parseFloat(elements.gripperSlider?.value || 50) / 100;
    try {
        const response = await fetch(`${ROBOT_URL}/gripper/set`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ position: pos }),
        });
        const result = await response.json();
        showRobotResult(result.message || result.error || JSON.stringify(result));
    } catch (error) {
        showRobotResult(`Error: ${error.message}`);
    }
}

// Motor Recovery Functions
async function rebootWrist() {
    addChatMessage('System', 'Rebooting wrist_angle motor (ID 4)...');
    showRobotResult('Rebooting wrist motor...');
    try {
        const response = await fetch(`${ROBOT_URL}/motors/reboot/wrist`, {
            method: 'POST',
        });
        const result = await response.json();
        showRobotResult(JSON.stringify(result, null, 2));
        addChatMessage('System', result.success ? 'Wrist motor rebooted.' : `Failed: ${result.error}`);
    } catch (error) {
        showRobotResult(`Error: ${error.message}`);
        addChatMessage('System', `Reboot error: ${error.message}`);
    }
}

async function rebootAllMotors() {
    if (!confirm('This will reboot ALL motors. Robot may collapse! Is the robot in a safe position?')) {
        return;
    }
    addChatMessage('System', 'Rebooting all motors...');
    showRobotResult('Rebooting all motors...');
    try {
        const response = await fetch(`${ROBOT_URL}/motors/reboot`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: 'all', enable: true, smart_reboot: true }),
        });
        const result = await response.json();
        showRobotResult(JSON.stringify(result, null, 2));
        addChatMessage('System', result.success ? 'All motors rebooted.' : `Failed: ${result.error}`);
    } catch (error) {
        showRobotResult(`Error: ${error.message}`);
        addChatMessage('System', `Reboot error: ${error.message}`);
    }
}

async function enableTorque() {
    addChatMessage('System', 'Enabling torque on all motors...');
    showRobotResult('Enabling torque...');
    try {
        const response = await fetch(`${ROBOT_URL}/motors/torque`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: 'all', enable: true }),
        });
        const result = await response.json();
        showRobotResult(JSON.stringify(result, null, 2));
        addChatMessage('System', result.success ? 'Torque enabled.' : `Failed: ${result.error}`);
    } catch (error) {
        showRobotResult(`Error: ${error.message}`);
        addChatMessage('System', `Torque error: ${error.message}`);
    }
}

async function disableTorque() {
    addChatMessage('System', 'Disabling torque (teach mode)...');
    showRobotResult('Disabling torque...');
    try {
        const response = await fetch(`${ROBOT_URL}/motors/torque`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: 'all', enable: false }),
        });
        const result = await response.json();
        showRobotResult(JSON.stringify(result, null, 2));
        addChatMessage('System', result.success ? 'Torque disabled - you can now move robot by hand.' : `Failed: ${result.error}`);
    } catch (error) {
        showRobotResult(`Error: ${error.message}`);
        addChatMessage('System', `Torque error: ${error.message}`);
    }
}

// Gesture Functions
async function loadGestures() {
    try {
        const response = await fetch(`${ROBOT_URL}/arm/gestures`);
        const result = await response.json();
        const list = document.getElementById('gesture-list');
        const status = document.getElementById('gesture-status');
        if (!list) return;
        list.innerHTML = '';
        if (result.success) {
            const gestures = result.gestures || {};
            for (const [name, info] of Object.entries(gestures)) {
                const opt = document.createElement('option');
                opt.value = name;
                const tag = info.builtin ? '[B]' : '[C]';
                opt.textContent = `${tag} ${name} (${info.duration_s}s, ${info.num_frames}f)`;
                list.appendChild(opt);
            }
            if (status) status.textContent = `${Object.keys(gestures).length} gestures loaded`;
            if (result.recording) {
                if (status) status.textContent = 'Recording in progress...';
            }
        }
        onGestureSelected();
    } catch (error) {
        const status = document.getElementById('gesture-status');
        if (status) status.textContent = `Error: ${error.message}`;
    }
}

function onGestureSelected() {
    const list = document.getElementById('gesture-list');
    const selected = list && list.value;
    const btnPlay = document.getElementById('btn-gesture-play');
    const btnDelete = document.getElementById('btn-gesture-delete');
    const btnStopPlay = document.getElementById('btn-gesture-stop-play');
    if (btnPlay) btnPlay.disabled = !selected;
    if (btnDelete) btnDelete.disabled = !selected;
    if (btnStopPlay) btnStopPlay.disabled = !selected;
}

async function startGestureRecording() {
    const nameInput = document.getElementById('gesture-name');
    const name = nameInput ? nameInput.value.trim() : '';
    if (!name) {
        addChatMessage('System', 'Enter a gesture name first.');
        return;
    }
    const status = document.getElementById('gesture-status');
    if (status) status.textContent = 'Starting recording...';
    try {
        const response = await fetch(`${ROBOT_URL}/arm/gesture/record/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: name, fps: 10, description: '' }),
        });
        const result = await response.json();
        if (result.success) {
            if (status) status.textContent = 'Recording... Move arm by hand, then click Stop.';
            const btnRecord = document.getElementById('btn-gesture-record');
            const btnStop = document.getElementById('btn-gesture-stop-record');
            if (btnRecord) btnRecord.disabled = true;
            if (btnStop) btnStop.disabled = false;
            addChatMessage('System', result.message);
        } else {
            if (status) status.textContent = `Error: ${result.error}`;
        }
    } catch (error) {
        if (status) status.textContent = `Error: ${error.message}`;
    }
}

async function stopGestureRecording() {
    const status = document.getElementById('gesture-status');
    if (status) status.textContent = 'Stopping recording...';
    try {
        const response = await fetch(`${ROBOT_URL}/arm/gesture/record/stop`, {
            method: 'POST',
        });
        const result = await response.json();
        const btnRecord = document.getElementById('btn-gesture-record');
        const btnStop = document.getElementById('btn-gesture-stop-record');
        if (btnRecord) btnRecord.disabled = false;
        if (btnStop) btnStop.disabled = true;
        if (result.success) {
            if (status) status.textContent = result.message;
            addChatMessage('System', result.message);
            loadGestures();
        } else {
            if (status) status.textContent = `Error: ${result.error}`;
        }
    } catch (error) {
        if (status) status.textContent = `Error: ${error.message}`;
    }
}

async function playGesture() {
    const list = document.getElementById('gesture-list');
    const name = list ? list.value : '';
    if (!name) return;
    const status = document.getElementById('gesture-status');
    if (status) status.textContent = `Playing "${name}"...`;
    try {
        const response = await fetch(`${ROBOT_URL}/arm/gesture/play`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: name, speed: 1.0 }),
        });
        const result = await response.json();
        if (status) status.textContent = result.success ? result.message : `Error: ${result.error}`;
    } catch (error) {
        if (status) status.textContent = `Error: ${error.message}`;
    }
}

async function stopGesturePlayback() {
    try {
        await fetch(`${ROBOT_URL}/arm/gesture/stop`, { method: 'POST' });
        const status = document.getElementById('gesture-status');
        if (status) status.textContent = 'Playback stopped.';
    } catch (error) {
        console.error('Stop gesture error:', error);
    }
}

async function deleteGesture() {
    const list = document.getElementById('gesture-list');
    const name = list ? list.value : '';
    if (!name) return;
    if (!confirm(`Delete gesture "${name}"?`)) return;
    const status = document.getElementById('gesture-status');
    try {
        const response = await fetch(`${ROBOT_URL}/arm/gesture/${encodeURIComponent(name)}`, {
            method: 'DELETE',
        });
        const result = await response.json();
        if (result.success) {
            if (status) status.textContent = result.message;
            loadGestures();
        } else {
            if (status) status.textContent = `Error: ${result.error}`;
        }
    } catch (error) {
        if (status) status.textContent = `Error: ${error.message}`;
    }
}

// Waypoint Functions
let waypointData = [];  // [{name, tag}, ...]

async function refreshWaypoints() {
    try {
        const response = await fetch(`${ROBOT_URL}/waypoints`);
        const result = await response.json();
        if (result.success && elements.waypointList) {
            waypointData = result.waypoints || [];
            elements.waypointList.innerHTML = '';
            waypointData.forEach(wp => {
                const option = document.createElement('option');
                option.value = wp.name;
                const tagLabel = wp.tag ? ` [${wp.tag}]` : '';
                option.textContent = wp.name + tagLabel;
                elements.waypointList.appendChild(option);
            });
            showRobotResult(`Loaded ${waypointData.length} waypoint(s)`);
        } else {
            showRobotResult(`Error: ${result.error || 'Failed to load waypoints'}`);
        }
    } catch (error) {
        showRobotResult(`Error: ${error.message}`);
    }
}

async function saveWaypoint() {
    const name = elements.waypointName?.value?.trim();
    if (!name) {
        showRobotResult('Please enter a waypoint name');
        addChatMessage('System', 'Please enter a waypoint name');
        return;
    }

    const tagEl = document.getElementById('waypoint-tag');
    const tag = tagEl?.value || null;

    addChatMessage('System', `Saving waypoint "${name}"${tag ? ` [${tag}]` : ''}...`);
    try {
        const body = { name: name };
        if (tag) body.tag = tag;
        const response = await fetch(`${ROBOT_URL}/waypoints/save`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const result = await response.json();
        if (result.success) {
            showRobotResult(`Waypoint "${name}" saved${tag ? ` [${tag}]` : ''}`);
            addChatMessage('System', `Waypoint "${name}" saved.`);
            elements.waypointName.value = '';
            if (tagEl) tagEl.value = '';
            await refreshWaypoints();
        } else {
            showRobotResult(`Error: ${result.error}`);
            addChatMessage('System', `Failed to save waypoint: ${result.error}`);
        }
    } catch (error) {
        showRobotResult(`Error: ${error.message}`);
        addChatMessage('System', `Waypoint error: ${error.message}`);
    }
}

async function gotoWaypoint() {
    const name = elements.waypointList?.value;
    if (!name) {
        showRobotResult('Please select a waypoint');
        addChatMessage('System', 'Please select a waypoint from the list');
        return;
    }

    addChatMessage('System', `Moving to waypoint "${name}"...`);
    try {
        const response = await fetch(`${ROBOT_URL}/waypoints/load/${encodeURIComponent(name)}`, {
            method: 'POST',
        });
        const result = await response.json();
        if (result.success) {
            let text = `Moved to waypoint "${name}"\n`;
            if (result.waypoint && result.waypoint.joints) {
                text += '\nJoint positions:\n';
                for (const [joint, pos] of Object.entries(result.waypoint.joints)) {
                    const deg = (pos * 180 / Math.PI).toFixed(1);
                    text += `  ${joint}: ${deg}°\n`;
                }
            }
            showRobotResult(text);
            addChatMessage('System', `Moved to waypoint "${name}".`);
        } else {
            showRobotResult(`Error: ${result.error}`);
            addChatMessage('System', `Failed to go to waypoint: ${result.error}`);
        }
    } catch (error) {
        showRobotResult(`Error: ${error.message}`);
        addChatMessage('System', `Waypoint error: ${error.message}`);
    }
}

async function deleteWaypoint() {
    const name = elements.waypointList?.value;
    if (!name) {
        showRobotResult('Please select a waypoint to delete');
        addChatMessage('System', 'Please select a waypoint from the list');
        return;
    }

    if (!confirm(`Delete waypoint "${name}"?`)) {
        return;
    }

    addChatMessage('System', `Deleting waypoint "${name}"...`);
    try {
        const response = await fetch(`${ROBOT_URL}/waypoints/${encodeURIComponent(name)}`, {
            method: 'DELETE',
        });
        const result = await response.json();
        if (result.success) {
            showRobotResult(`Waypoint "${name}" deleted`);
            addChatMessage('System', `Waypoint "${name}" deleted.`);
            await refreshWaypoints();
        } else {
            showRobotResult(`Error: ${result.error}`);
            addChatMessage('System', `Failed to delete waypoint: ${result.error}`);
        }
    } catch (error) {
        showRobotResult(`Error: ${error.message}`);
        addChatMessage('System', `Waypoint error: ${error.message}`);
    }
}

// ==================== CALIBRATION FUNCTIONS ====================

let detectedCalibTags = [];  // Store detected tags for selection

function showCalibOutput(text) {
    const output = document.getElementById('calib-output');
    if (output) output.textContent = text;
}

function updateCalibStatus(status) {
    const statusText = document.getElementById('calib-status-text');
    const pointCount = document.getElementById('calib-point-count');
    if (statusText) {
        statusText.textContent = status.is_calibrated ? 'Calibrated' : 'Not calibrated';
        statusText.style.color = status.is_calibrated ? '#28a745' : '#dc3545';
    }
    if (pointCount) {
        pointCount.textContent = `(${status.num_points} points)`;
    }
}

async function detectCalibTags() {
    showCalibOutput('Detecting AprilTags...');
    try {
        const response = await fetch(`${VISION_URL}/calibration/detect_tags`, { method: 'POST' });
        const result = await response.json();

        if (result.success) {
            detectedCalibTags = result.points;
            const tagsList = document.getElementById('calib-tags-list');
            const tagSelect = document.getElementById('calib-tag-select');

            if (result.points.length === 0) {
                tagsList.innerHTML = '<em>No tags detected</em>';
                tagSelect.innerHTML = '<option value="">-- No tags detected --</option>';
            } else {
                let listHtml = '';
                let selectHtml = '<option value="">-- Select a tag --</option>';
                result.points.forEach((p, i) => {
                    const pos = `(${(p.position[0]*100).toFixed(1)}, ${(p.position[1]*100).toFixed(1)}, ${(p.position[2]*100).toFixed(1)}) cm`;
                    listHtml += `<div>Tag ${p.tag_id}: ${pos}</div>`;
                    selectHtml += `<option value="${i}">Tag ${p.tag_id} - ${pos}</option>`;
                });
                tagsList.innerHTML = listHtml;
                tagSelect.innerHTML = selectHtml;
            }

            // Show image in vision preview
            if (result.image_base64) {
                const visionImg = document.getElementById('vision-image');
                if (visionImg) {
                    visionImg.src = 'data:image/jpeg;base64,' + result.image_base64;
                }
            }

            showCalibOutput(`Detected ${result.points.length} tag(s)`);
            addChatMessage('System', `Detected ${result.points.length} AprilTag(s)`);
        } else {
            showCalibOutput(`Error: ${result.error}`);
        }
    } catch (error) {
        showCalibOutput(`Error: ${error.message}`);
    }
}

async function recordCalibPoint() {
    const tagSelect = document.getElementById('calib-tag-select');
    const selectedIdx = tagSelect?.value;

    if (selectedIdx === '' || selectedIdx === null) {
        showCalibOutput('Please select a tag first');
        addChatMessage('System', 'Please select a detected tag first');
        return;
    }

    const tag = detectedCalibTags[parseInt(selectedIdx)];
    if (!tag) {
        showCalibOutput('Invalid tag selection');
        return;
    }

    showCalibOutput(`Recording point for Tag ${tag.tag_id}...`);
    addChatMessage('System', `Recording calibration point for Tag ${tag.tag_id}...`);

    try {
        const response = await fetch(`${ROBOT_URL}/calibration/add_point`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                camera_point: tag.position,
                tag_id: tag.tag_id
            }),
        });
        const result = await response.json();

        if (result.success) {
            updateCalibStatus(result);
            updateCalibPointsList(result.points);
            showCalibOutput(result.message || `Point recorded (${result.num_points} total)`);
            addChatMessage('System', `Calibration point ${result.num_points} recorded`);
        } else {
            showCalibOutput(`Error: ${result.error}`);
        }
    } catch (error) {
        showCalibOutput(`Error: ${error.message}`);
    }
}

function updateCalibPointsList(points) {
    const list = document.getElementById('calib-points-list');
    if (!list) return;

    if (!points || points.length === 0) {
        list.innerHTML = '<em>No points recorded</em>';
        return;
    }

    let html = '';
    points.forEach((p, i) => {
        const tagLabel = p.tag_id !== null && p.tag_id !== undefined ? `Tag ${p.tag_id}` : 'Unknown';
        const cam = `(${(p.camera[0]*100).toFixed(1)}, ${(p.camera[1]*100).toFixed(1)}, ${(p.camera[2]*100).toFixed(1)})`;
        const rob = `(${(p.robot[0]*100).toFixed(1)}, ${(p.robot[1]*100).toFixed(1)}, ${(p.robot[2]*100).toFixed(1)})`;
        html += `<div style="margin-bottom:5px;">
            <strong>Point ${i+1} [${tagLabel}]:</strong><br>
            &nbsp;&nbsp;Cam: ${cam} cm<br>
            &nbsp;&nbsp;Rob: ${rob} cm
        </div>`;
    });
    list.innerHTML = html;
}

async function computeCalibration() {
    showCalibOutput('Computing calibration...');
    addChatMessage('System', 'Computing camera-robot calibration...');

    try {
        const response = await fetch(`${ROBOT_URL}/calibration/compute`, {
            method: 'POST',
        });
        const result = await response.json();

        if (result.success) {
            updateCalibStatus(result);
            showCalibOutput(result.message || 'Calibration computed and saved');
            addChatMessage('System', result.message || 'Calibration computed successfully');
        } else {
            showCalibOutput(`Error: ${result.error}`);
            addChatMessage('System', `Calibration failed: ${result.error}`);
        }
    } catch (error) {
        showCalibOutput(`Error: ${error.message}`);
    }
}

async function clearCalibPoints() {
    if (!confirm('Clear all calibration points?')) return;

    try {
        const response = await fetch(`${ROBOT_URL}/calibration/clear`, {
            method: 'POST',
        });
        const result = await response.json();

        if (result.success) {
            updateCalibStatus(result);
            updateCalibPointsList([]);
            showCalibOutput('Calibration points cleared');
            addChatMessage('System', 'Calibration points cleared');
        } else {
            showCalibOutput(`Error: ${result.error}`);
        }
    } catch (error) {
        showCalibOutput(`Error: ${error.message}`);
    }
}

async function loadCalibration() {
    showCalibOutput('Loading calibration...');

    try {
        const response = await fetch(`${ROBOT_URL}/calibration/load`, {
            method: 'POST',
        });
        const result = await response.json();

        if (result.success) {
            updateCalibStatus(result);
            updateCalibPointsList(result.points);
            showCalibOutput(result.message || 'Calibration loaded');
            addChatMessage('System', 'Calibration loaded from file');
        } else {
            showCalibOutput(`Error: ${result.error}`);
            addChatMessage('System', result.error || 'Failed to load calibration');
        }
    } catch (error) {
        showCalibOutput(`Error: ${error.message}`);
    }
}

async function testTransform() {
    const tagSelect = document.getElementById('calib-tag-select');
    const selectedIdx = tagSelect?.value;

    if (selectedIdx === '' || selectedIdx === null) {
        document.getElementById('transform-result').textContent = 'Select a tag first';
        return;
    }

    const tag = detectedCalibTags[parseInt(selectedIdx)];
    if (!tag) {
        document.getElementById('transform-result').textContent = 'Invalid tag';
        return;
    }

    try {
        const response = await fetch(`${ROBOT_URL}/calibration/transform`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ point: tag.position }),
        });
        const result = await response.json();

        if (result.success) {
            const cam = result.camera_point.map(v => (v*100).toFixed(1)).join(', ');
            const rob = result.robot_point.map(v => (v*100).toFixed(1)).join(', ');
            document.getElementById('transform-result').textContent =
                `Camera: (${cam}) cm\nRobot:  (${rob}) cm`;
            // Store for move-to-tag
            window.lastTransformedPoint = result.robot_point;
        } else {
            document.getElementById('transform-result').textContent = `Error: ${result.error}`;
        }
    } catch (error) {
        document.getElementById('transform-result').textContent = `Error: ${error.message}`;
    }
}

// Move to the last transformed tag position
async function moveToTag() {
    console.log('moveToTag called, lastTransformedPoint:', window.lastTransformedPoint);

    if (!window.lastTransformedPoint) {
        addChatMessage('System', 'Click "Test Transform" first to get robot coordinates');
        showToast('Click "Test Transform" first!', 'warning');
        return;
    }

    const [x, y, zBase] = window.lastTransformedPoint;
    const zOffsetCm = parseFloat(document.getElementById('z-offset')?.value) || 0;
    const z = zBase + zOffsetCm / 100;  // Add offset (convert cm to m)
    const autoOrientation = document.getElementById('auto-orientation')?.checked || false;
    const pitchDeg = parseFloat(document.getElementById('move-pitch')?.value) || 90;
    const toleranceDeg = parseFloat(document.getElementById('pitch-tolerance')?.value) || 0;
    const pitch = pitchDeg * Math.PI / 180;
    const pitchTolerance = toleranceDeg * Math.PI / 180;

    const modeStr = autoOrientation ? 'auto orientation' : `pitch=${pitchDeg}°±${toleranceDeg}°`;
    console.log('Moving to:', x, y, z, `(z offset: +${zOffsetCm}cm)`, modeStr);
    addChatMessage('System', `Moving to (${(x*100).toFixed(1)}, ${(y*100).toFixed(1)}, ${(z*100).toFixed(1)}) cm [+${zOffsetCm}cm], ${modeStr}...`);

    try {
        const payload = { x, y, z, auto_orientation: autoOrientation };
        if (!autoOrientation) {
            payload.pitch = pitch;
            payload.pitch_tolerance = pitchTolerance;
        }
        console.log('Sending to /arm/move_to_xyz:', payload);
        const response = await fetch(`${ROBOT_URL}/arm/move_to_xyz`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        console.log('Response status:', response.status);
        const result = await response.json();
        console.log('Response body:', result);
        if (result.success) {
            addChatMessage('System', result.message);
        } else {
            addChatMessage('System', `Move failed: ${result.error}`);
        }
    } catch (error) {
        console.error('moveToTag error:', error);
        addChatMessage('System', `Error: ${error.message}`);
    }
}

// Move to manual XYZ position
async function moveToXYZ() {
    const x = parseFloat(document.getElementById('move-x')?.value) || 0.25;
    const y = parseFloat(document.getElementById('move-y')?.value) || 0;
    const zBase = parseFloat(document.getElementById('move-z')?.value) || 0.15;
    const zOffsetCm = parseFloat(document.getElementById('z-offset')?.value) || 0;
    const z = zBase + zOffsetCm / 100;  // Add offset (convert cm to m)
    const autoOrientation = document.getElementById('auto-orientation')?.checked || false;
    const pitchDeg = parseFloat(document.getElementById('move-pitch')?.value) || 90;
    const toleranceDeg = parseFloat(document.getElementById('pitch-tolerance')?.value) || 0;
    const pitch = pitchDeg * Math.PI / 180;
    const pitchTolerance = toleranceDeg * Math.PI / 180;

    const modeStr = autoOrientation ? 'auto orientation' : `pitch=${pitchDeg}°±${toleranceDeg}°`;
    addChatMessage('System', `Moving to (${(x*100).toFixed(1)}, ${(y*100).toFixed(1)}, ${(z*100).toFixed(1)}) cm [+${zOffsetCm}cm], ${modeStr}...`);

    try {
        const payload = { x, y, z, auto_orientation: autoOrientation };
        if (!autoOrientation) {
            payload.pitch = pitch;
            payload.pitch_tolerance = pitchTolerance;
        }
        const response = await fetch(`${ROBOT_URL}/arm/move_to_xyz`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const result = await response.json();
        if (result.success) {
            addChatMessage('System', result.message);
        } else {
            addChatMessage('System', `Move failed: ${result.error}`);
        }
    } catch (error) {
        addChatMessage('System', `Error: ${error.message}`);
    }
}

async function refreshCalibStatus() {
    try {
        const response = await fetch(`${ROBOT_URL}/calibration/status`);
        const result = await response.json();
        if (result.success) {
            updateCalibStatus(result);
            updateCalibPointsList(result.points);
        }
    } catch (error) {
        console.log('Could not fetch calibration status');
    }
}

// UI Update Functions
function updateGameDisplay(state) {
    if (!state) return;

    // Skip board overwrite while AI is thinking (preserve optimistic human move)
    if (_awaitingRobotMove && state.whose_turn === 'human') return;

    // Update board display
    if (state.fen) {
        if (state.simulation && simulationMode) {
            simLegalMoves = state.legal_moves || [];
            simIsCheck = !!state.is_check;
            renderSimBoard(state.fen, simLegalMoves, state.whose_turn, state.last_robot_move, state.last_human_move);
        } else {
            renderMainBoard(state.fen);
        }
    }

    // Update game info
    elements.gameStatus.textContent = state.status || 'Unknown';
    if (state.game_mode === 'watch') {
        // Watch mode: show which side is thinking
        const side = state.fen && state.fen.split(' ')[1] === 'w' ? 'White' : 'Black';
        elements.whoseTurn.textContent = state.status === 'playing' ? `${side} (thinking...)` : '-';
    } else {
        elements.whoseTurn.textContent = state.whose_turn === 'robot' ? llmModelName : (state.whose_turn === 'human' ? 'You' : '-');
    }
    const humanLabel = document.getElementById('human-label');
    const robotLabel = document.getElementById('robot-label');
    if (state.game_mode === 'watch') {
        if (humanLabel) humanLabel.textContent = 'White';
        if (robotLabel) robotLabel.textContent = 'Black';
    } else {
        if (humanLabel) humanLabel.textContent = 'You';
        if (robotLabel) robotLabel.textContent = llmModelName;
    }
    elements.moveNumber.textContent = state.move_number || '-';
    if (state.game_mode === 'watch') {
        const wEng = document.getElementById('watch-white-engine')?.value || '?';
        const bEng = document.getElementById('watch-black-engine')?.value || '?';
        const wDiff = document.getElementById('watch-white-difficulty')?.value || '';
        const bDiff = document.getElementById('watch-black-difficulty')?.value || '';
        const wLabel = wEng === 'stockfish' ? `SF(${wDiff})` : wEng.toUpperCase();
        const bLabel = bEng === 'stockfish' ? `SF(${bDiff})` : bEng.toUpperCase();
        elements.currentMode.textContent = `Watch: ${wLabel} vs ${bLabel}`;
    } else {
        const modeLabel = state.game_mode === 'teach' ? 'Teach' : 'Battle';
        const diff = state.difficulty || document.getElementById('difficulty-select')?.value || '';
        const diffLabel = diff ? diff.charAt(0).toUpperCase() + diff.slice(1) : '';
        const src = state.move_source || document.getElementById('move-source-select')?.value || 'stockfish';
        const srcLabel = src !== 'stockfish' ? `, ${src.toUpperCase()}` : '';
        elements.currentMode.textContent = diffLabel ? `${modeLabel} (${diffLabel}${srcLabel})` : modeLabel;
    }

    // Update last moves
    if (state.last_human_move) {
        const hm = state.last_human_move;
        elements.lastHumanMove.textContent = `${hm.from_square || ''} → ${hm.to_square || hm}`;
    }
    if (state.last_robot_move) {
        const rm = state.last_robot_move;
        elements.lastRobotMove.textContent = `${rm.from_square || ''} → ${rm.to_square || rm}`;
    }

    // Teach mode: update teach panel
    if (state.game_mode === 'teach' && state.lesson_id) {
        const panel = document.getElementById('teach-panel');
        if (panel) panel.style.display = '';
        const stepInfo = document.getElementById('teach-step-info');
        if (stepInfo && state.lesson_step !== null && state.lesson_total_steps) {
            stepInfo.textContent = `Step ${state.lesson_step + 1} / ${state.lesson_total_steps}`;
        }
    }

    // Game over overlay (battle mode only)
    if (state.status === 'ended' && state.game_result && (state.game_mode === 'battle' || state.game_mode === 'watch') && !_gameOverDismissed) {
        showGameOverOverlay(state);
    }
}

let _gameOverDismissed = false;

function dismissGameOverOverlay() {
    _gameOverDismissed = true;
    const overlay = document.getElementById('game-over-overlay');
    if (overlay) overlay.style.display = 'none';
}

function showGameOverOverlay(state) {
    const overlay = document.getElementById('game-over-overlay');
    if (!overlay || _gameOverDismissed) return;

    const result = state.game_result;
    const iconEl = document.getElementById('game-over-icon');
    const titleEl = document.getElementById('game-over-title');
    const detailEl = document.getElementById('game-over-detail');

    const difficulty = state.difficulty || 'intermediate';
    const diffLabel = difficulty.charAt(0).toUpperCase() + difficulty.slice(1);
    const opponent = llmModelName || 'AI';

    const isWatch = state.game_mode === 'watch';
    let icon = '', title = '', detail = '';
    if (result === 'checkmate') {
        if (isWatch) {
            // In watch mode: whose_turn after checkmate is the LOSER (can't move)
            // FEN side-to-move = the side that's checkmated
            const loserSide = state.fen ? state.fen.split(' ')[1] : '?';
            const winnerLabel = loserSide === 'w' ? 'Black' : 'White';
            // Get model names from watch config selects
            const wEngine = document.getElementById('watch-white-engine')?.value || 'stockfish';
            const bEngine = document.getElementById('watch-black-engine')?.value || 'stockfish';
            function _engineModel(eng) {
                if (eng === 'stockfish') return 'Stockfish';
                if (eng === 'llm3') return document.getElementById('setting-llm-model-3')?.value || 'LLM 3';
                if (eng === 'llm2') return document.getElementById('setting-llm-model-2')?.value || 'LLM 2';
                return document.getElementById('setting-llm-model')?.value || 'LLM 1';
            }
            const wModel = _engineModel(wEngine);
            const bModel = _engineModel(bEngine);
            const winnerModel = winnerLabel === 'White' ? wModel : bModel;
            icon = '\u{1F3C6}';
            title = `${winnerLabel} Wins!`;
            detail = `${winnerModel} checkmates (${wModel} vs ${bModel})`;
        } else if (state.whose_turn === 'robot') {
            icon = '\u{1F3C6}';
            title = 'You Win!';
            detail = `Checkmate vs ${opponent} (${diffLabel})`;
        } else {
            icon = '\u{1F916}';
            title = `${opponent} Wins`;
            detail = `Checkmate (${diffLabel})`;
        }
    } else if (result === 'stalemate' || result === 'draw') {
        icon = '\u{1F91D}';
        title = 'Draw';
        detail = isWatch ? 'Stalemate (AI vs AI)' : (result === 'stalemate' ? `Stalemate vs ${opponent} (${diffLabel})` : `Draw vs ${opponent} (${diffLabel})`);
    } else if (result === 'resigned') {
        icon = '\u{1F3F3}';
        title = 'Resigned';
        detail = `vs ${opponent} (${diffLabel})`;
    } else {
        return;
    }

    if (iconEl) iconEl.textContent = icon;
    if (titleEl) titleEl.textContent = title;
    if (detailEl) detailEl.textContent = detail;

    overlay.style.display = 'flex';
    overlay.style.animation = 'gameOverIn 0.6s ease forwards';
}

// FEN char → Unicode piece + color class
const FEN_PIECE_MAP = {
    'K': { ch: '\u2654', color: 'white' }, 'Q': { ch: '\u2655', color: 'white' },
    'R': { ch: '\u2656', color: 'white' }, 'B': { ch: '\u2657', color: 'white' },
    'N': { ch: '\u2658', color: 'white' }, 'P': { ch: '\u2659', color: 'white' },
    'k': { ch: '\u265A', color: 'black' }, 'q': { ch: '\u265B', color: 'black' },
    'r': { ch: '\u265C', color: 'black' }, 'b': { ch: '\u265D', color: 'black' },
    'n': { ch: '\u265E', color: 'black' }, 'p': { ch: '\u265F', color: 'black' },
};

function renderMainBoard(fen) {
    const boardEl = elements.chessBoard;
    if (!boardEl) return;
    boardEl.innerHTML = '';

    // Set grid layout
    boardEl.style.display = 'grid';
    boardEl.style.gridTemplateColumns = 'auto repeat(8, 1fr)';
    boardEl.style.gridTemplateRows = 'repeat(8, 1fr) auto';
    boardEl.style.gap = '0';
    boardEl.style.whiteSpace = 'normal';

    const position = fen.split(' ')[0];
    const rows = position.split('/');
    const files = ['a','b','c','d','e','f','g','h'];

    rows.forEach((row, ri) => {
        const rank = 8 - ri;
        // Rank label
        const label = document.createElement('div');
        label.textContent = rank;
        label.style.cssText = 'display:flex; align-items:center; justify-content:center; width:22px; font-weight:bold; font-size:12px;';
        boardEl.appendChild(label);

        let fileIdx = 0;
        for (const ch of row) {
            if (isNaN(ch)) {
                const cell = document.createElement('div');
                const isLight = (fileIdx + rank) % 2 === 1;
                const p = FEN_PIECE_MAP[ch];
                cell.textContent = p ? p.ch : ch;
                cell.style.cssText = `
                    display:flex; align-items:center; justify-content:center;
                    aspect-ratio:1; font-weight:bold; font-size:22px;
                    border: 1px solid #555;
                    background: ${isLight ? '#b58863' : '#f0d9b5'};
                    color: ${p && p.color === 'black' ? '#1a1a1a' : '#fff'};
                    ${p && p.color === 'white' ? 'text-shadow: 0 0 3px #000;' : ''}
                `;
                boardEl.appendChild(cell);
                fileIdx++;
            } else {
                const empty = parseInt(ch);
                for (let j = 0; j < empty; j++) {
                    const cell = document.createElement('div');
                    const isLight = (fileIdx + rank) % 2 === 1;
                    cell.style.cssText = `
                        display:flex; align-items:center; justify-content:center;
                        aspect-ratio:1;
                        border: 1px solid #555;
                        background: ${isLight ? '#b58863' : '#f0d9b5'};
                    `;
                    boardEl.appendChild(cell);
                    fileIdx++;
                }
            }
        }
    });

    // Bottom file labels
    const spacer = document.createElement('div');
    spacer.style.width = '22px';
    boardEl.appendChild(spacer);
    for (const f of files) {
        const lbl = document.createElement('div');
        lbl.textContent = f;
        lbl.style.cssText = 'display:flex; align-items:center; justify-content:center; font-weight:bold; font-size:12px;';
        boardEl.appendChild(lbl);
    }
}

function renderSimBoard(fen, legalMoves, whoseTurn, lastRobotMove, lastHumanMove) {
    const boardEl = elements.chessBoard;
    if (!boardEl) return;

    // Cache args when called with data; restore from cache when called without
    if (fen) {
        simCachedRenderArgs = [fen, legalMoves, whoseTurn, lastRobotMove, lastHumanMove];
    } else if (simCachedRenderArgs) {
        [fen, legalMoves, whoseTurn, lastRobotMove, lastHumanMove] = simCachedRenderArgs;
    } else {
        return; // no data to render
    }

    boardEl.innerHTML = '';
    boardEl.className = 'sim-board';

    boardEl.style.display = 'grid';
    boardEl.style.gridTemplateColumns = 'auto repeat(8, 1fr)';
    boardEl.style.gridTemplateRows = 'repeat(8, 1fr) auto';
    boardEl.style.gap = '0';
    boardEl.style.maxWidth = '';
    boardEl.style.userSelect = 'none';

    const position = fen.split(' ')[0];
    const fenRows = position.split('/');
    const files = ['a','b','c','d','e','f','g','h'];
    const flipped = simHumanColor === 'black';
    const isHumanTurn = currentMode === 'watch' ? false : (whoseTurn === 'human');

    // Parse board into 8x8 array
    const board = [];
    for (let ri = 0; ri < 8; ri++) {
        const row = [];
        for (const ch of fenRows[ri]) {
            if (isNaN(ch)) {
                row.push(ch);
            } else {
                for (let j = 0; j < parseInt(ch); j++) row.push(null);
            }
        }
        board.push(row);
    }

    // Find king in check — use simIsCheck flag from server state
    let kingInCheckSq = null;
    if (simIsCheck) {
        const fenParts = fen.split(' ');
        const activeColor = fenParts[1]; // 'w' or 'b'
        const kingChar = activeColor === 'w' ? 'K' : 'k';
        for (let ri = 0; ri < 8; ri++) {
            for (let fi = 0; fi < 8; fi++) {
                if (board[ri][fi] === kingChar) {
                    kingInCheckSq = files[fi] + (8 - ri);
                }
            }
        }
    }

    // Determine last move squares for highlighting
    // In watch mode, pick whichever move was most recent (stored in simLastMove by robot_done)
    const lastMove = (currentMode === 'watch' && simLastMove) ? (simLastMove.from + simLastMove.to) : (lastRobotMove || lastHumanMove);
    const lastMoveFrom = lastMove ? lastMove.substring(0, 2) : null;
    const lastMoveTo = lastMove ? lastMove.substring(2, 4) : null;

    // Get destinations for selected square
    const selectedDests = new Set();
    const selectedPromotions = {};  // dest -> [promotion chars]
    if (simSelectedSquare && legalMoves) {
        for (const m of legalMoves) {
            if (m.substring(0, 2) === simSelectedSquare) {
                const dest = m.substring(2, 4);
                selectedDests.add(dest);
                if (m.length === 5) {
                    if (!selectedPromotions[dest]) selectedPromotions[dest] = [];
                    selectedPromotions[dest].push(m[4]);
                }
            }
        }
    }

    // Determine which pieces are the human's (clickable)
    const humanIsWhite = simHumanColor === 'white';
    function isHumanPiece(ch) {
        if (!ch) return false;
        return humanIsWhite ? ch === ch.toUpperCase() : ch === ch.toLowerCase();
    }

    // Render ranks
    const displayRanks = flipped ? [1,2,3,4,5,6,7,8] : [8,7,6,5,4,3,2,1];
    const displayFiles = flipped ? [...files].reverse() : files;

    for (let di = 0; di < 8; di++) {
        const rank = displayRanks[di];
        const ri = 8 - rank; // index into board array

        // Rank label
        const label = document.createElement('div');
        label.textContent = rank;
        label.style.cssText = 'display:flex; align-items:center; justify-content:center; width:22px; font-weight:bold; font-size:12px;';
        boardEl.appendChild(label);

        for (let dfi = 0; dfi < 8; dfi++) {
            const fi = flipped ? 7 - dfi : dfi;
            const sq = files[fi] + rank;
            const piece = board[ri][fi];
            const isLight = (fi + rank) % 2 === 1;

            const cell = document.createElement('div');
            cell.dataset.square = sq;
            cell.style.position = 'relative';
            cell.style.display = 'flex';
            cell.style.alignItems = 'center';
            cell.style.justifyContent = 'center';
            cell.style.aspectRatio = '1';
            cell.style.fontSize = '28px';
            cell.style.fontWeight = 'bold';
            cell.style.border = '1px solid #555';
            cell.style.cursor = 'default';
            cell.style.transition = 'background 0.15s';

            // Background color
            let bg = isLight ? '#f0d9b5' : '#b58863';
            if (sq === lastMoveFrom || sq === lastMoveTo) {
                bg = isLight ? '#f7ec5e' : '#cda744'; // yellow tint for last move
            }
            // King in check: red highlight
            if (sq === kingInCheckSq) {
                bg = '#e84040'; // bold red
            }
            // Teach mode: highlight guided squares (blue tint for source/dest)
            if (teachHighlightSquares.indexOf(sq) >= 0) {
                if (sq === teachHighlightFrom) {
                    bg = isLight ? '#aec6f7' : '#5b8fd4'; // blue for source
                } else if (sq === teachHighlightTo) {
                    bg = isLight ? '#7ee08a' : '#3cb84e'; // green for destination
                } else {
                    bg = isLight ? '#c4dbf7' : '#7ba4d9'; // lighter blue for castling rook squares
                }
            }
            if (sq === simSelectedSquare) {
                bg = isLight ? '#aad576' : '#6fbf73'; // green for selected (overrides)
            }
            cell.style.background = bg;

            // Piece rendering
            if (piece) {
                const p = FEN_PIECE_MAP[piece];
                if (p) {
                    cell.textContent = p.ch;
                    cell.style.color = p.color === 'black' ? '#1a1a1a' : '#fff';
                    if (p.color === 'white') cell.style.textShadow = '0 0 3px #000';
                }
            }

            // Legal move dot/ring
            if (selectedDests.has(sq)) {
                const indicator = document.createElement('div');
                if (piece) {
                    // Capture: ring around piece
                    indicator.style.cssText = 'position:absolute; inset:5%; border-radius:50%; border:3px solid rgba(0,0,0,0.3); box-sizing:border-box; pointer-events:none;';
                } else {
                    // Move: dot in center
                    indicator.style.cssText = 'position:absolute; width:30%; height:30%; top:35%; left:35%; border-radius:50%; background:rgba(0,0,0,0.25); pointer-events:none;';
                }
                cell.appendChild(indicator);
                cell.style.cursor = 'pointer';

                // Click to move
                cell.addEventListener('click', ((fromSq, toSq, promos) => (e) => {
                    e.stopPropagation();
                    handleSimDestClick(fromSq, toSq, promos);
                })(simSelectedSquare, sq, selectedPromotions[sq]));
            } else if (isHumanTurn && isHumanPiece(piece)) {
                // Clickable own piece
                cell.style.cursor = 'pointer';
                cell.addEventListener('click', ((square, pc) => (e) => {
                    e.stopPropagation();
                    // Castle by clicking rook while king is selected
                    if (simSelectedSquare && pc && legalMoves) {
                        const selPiece = board[8 - parseInt(simSelectedSquare[1])][files.indexOf(simSelectedSquare[0])];
                        if ((selPiece === 'K' || selPiece === 'k') && (pc === 'R' || pc === 'r')) {
                            const castleMap = { 'h1': 'g1', 'a1': 'c1', 'h8': 'g8', 'a8': 'c8' };
                            const dest = castleMap[square];
                            if (dest && legalMoves.indexOf(simSelectedSquare + dest) >= 0) {
                                handleSimDestClick(simSelectedSquare, dest, null);
                                return;
                            }
                        }
                    }
                    simSelectedSquare = (simSelectedSquare === square) ? null : square;
                    renderSimBoard(fen, legalMoves, whoseTurn, lastRobotMove, lastHumanMove);
                })(sq, piece));
            } else {
                // Click empty/opponent = deselect
                cell.addEventListener('click', () => {
                    if (simSelectedSquare) {
                        simSelectedSquare = null;
                        renderSimBoard(fen, legalMoves, whoseTurn, lastRobotMove, lastHumanMove);
                    }
                });
            }

            boardEl.appendChild(cell);
        }
    }

    // Bottom file labels
    const spacer = document.createElement('div');
    spacer.style.width = '22px';
    boardEl.appendChild(spacer);
    for (const f of displayFiles) {
        const lbl = document.createElement('div');
        lbl.textContent = f;
        lbl.style.cssText = 'display:flex; align-items:center; justify-content:center; font-weight:bold; font-size:12px;';
        boardEl.appendChild(lbl);
    }

    // Show thinking indicator
    if (whoseTurn === 'robot' && gameActive) {
        const status = document.getElementById('auto-detect-status');
        if (status) status.textContent = 'AI thinking...';
    }
}

function _animateTeachDemoMove(fromSq, toSq) {
    // Animate a "ghost" piece sliding from source to destination on the sim board
    const boardEl = elements.chessBoard;
    if (!boardEl) return;

    const fromCell = boardEl.querySelector('[data-square="' + fromSq + '"]');
    const toCell = boardEl.querySelector('[data-square="' + toSq + '"]');
    if (!fromCell || !toCell) return;

    // Get the piece character from the source cell
    const pieceText = fromCell.textContent.trim();
    const pieceColor = fromCell.style.color;
    const pieceShadow = fromCell.style.textShadow;
    if (!pieceText) return;

    // Calculate positions for the sliding animation
    const boardRect = boardEl.getBoundingClientRect();
    const fromRect = fromCell.getBoundingClientRect();
    const toRect = toCell.getBoundingClientRect();

    // Create ghost piece element
    const ghost = document.createElement('div');
    ghost.className = 'teach-demo-ghost';
    ghost.textContent = pieceText;
    ghost.style.cssText = 'position:fixed; font-size:28px; font-weight:bold; pointer-events:none; z-index:50; transition:all 0.8s ease-in-out; opacity:0.85;';
    ghost.style.color = pieceColor;
    ghost.style.textShadow = pieceShadow || '';
    ghost.style.left = (fromRect.left + fromRect.width / 2 - 14) + 'px';
    ghost.style.top = (fromRect.top + fromRect.height / 2 - 14) + 'px';
    document.body.appendChild(ghost);

    // Fade out the original piece
    fromCell.style.opacity = '0.3';

    // Highlight destination
    toCell.style.outline = '3px solid #4ecca3';
    toCell.style.outlineOffset = '-3px';

    // Trigger the slide animation
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            ghost.style.left = (toRect.left + toRect.width / 2 - 14) + 'px';
            ghost.style.top = (toRect.top + toRect.height / 2 - 14) + 'px';
            ghost.style.transform = 'scale(1.15)';
        });
    });

    // Clean up after animation
    setTimeout(() => {
        ghost.style.opacity = '0';
        setTimeout(() => {
            ghost.remove();
            fromCell.style.opacity = '1';
            toCell.style.outline = '';
            toCell.style.outlineOffset = '';
        }, 400);
    }, 1200);
}

function _applyMoveToFen(fen, fromSq, toSq) {
    // Simple FEN manipulation: move piece from one square to another
    try {
        const parts = fen.split(' ');
        const rows = parts[0].split('/');
        const files = 'abcdefgh';

        // Parse board into 8x8 array
        const board = [];
        for (const row of rows) {
            const r = [];
            for (const ch of row) {
                if (ch >= '1' && ch <= '8') {
                    for (let i = 0; i < parseInt(ch); i++) r.push('');
                } else {
                    r.push(ch);
                }
            }
            board.push(r);
        }

        const fc = files.indexOf(fromSq[0]), fr = 8 - parseInt(fromSq[1]);
        const tc = files.indexOf(toSq[0]), tr = 8 - parseInt(toSq[1]);

        const piece = board[fr][fc];
        if (!piece) return null;

        board[tr][tc] = piece;
        board[fr][fc] = '';

        // Handle castling: move the rook too
        if ((piece === 'K' || piece === 'k') && Math.abs(fc - tc) === 2) {
            if (tc === 6) { // Kingside: rook h→f
                board[fr][5] = board[fr][7]; board[fr][7] = '';
            } else if (tc === 2) { // Queenside: rook a→d
                board[fr][3] = board[fr][0]; board[fr][0] = '';
            }
        }

        // Rebuild FEN position
        const newRows = board.map(row => {
            let s = '', empty = 0;
            for (const c of row) {
                if (c === '') { empty++; }
                else { if (empty) { s += empty; empty = 0; } s += c; }
            }
            if (empty) s += empty;
            return s;
        });

        // Flip side to move
        const side = parts[1] === 'w' ? 'b' : 'w';
        return newRows.join('/') + ' ' + side + ' ' + (parts[2]||'-') + ' ' + (parts[3]||'-') + ' 0 ' + (parts[5]||'1');
    } catch (e) {
        return null;
    }
}

async function handleSimDestClick(fromSq, toSq, promotionChars) {
    if (!fromSq) return;

    let uciMove = fromSq + toSq;

    // Check if promotion
    if (promotionChars && promotionChars.length > 0) {
        const chosen = await showPromotionDialog(promotionChars);
        if (!chosen) return; // cancelled
        uciMove += chosen;
    }

    simSelectedSquare = null;
    simLastMove = { from: fromSq, to: toSq };

    // In teach mode, route to check_move instead of submit_move
    if (currentMode === 'teach' && gameActive) {
        // Optimistic: show student's move immediately on the board
        if (simCachedRenderArgs) {
            const curFen = simCachedRenderArgs[0];
            const updatedFen = _applyMoveToFen(curFen, fromSq, toSq);
            if (updatedFen) {
                renderSimBoard(updatedFen, [], 'human', null, uciMove);
            }
        }
        const result = await teachCheckMove(uciMove);
        await getGameState();
        return;
    }

    // Immediately show human's move on the board (before server responds)
    _awaitingRobotMove = true;
    if (simCachedRenderArgs) {
        const curFen = simCachedRenderArgs[0];
        const updatedFen = _applyMoveToFen(curFen, fromSq, toSq);
        if (updatedFen) {
            renderSimBoard(updatedFen, [], 'robot', null, uciMove);
        }
    }
    elements.whoseTurn.textContent = 'AI (thinking...)';

    // Fire submit_move — don't await (takes 15-30s with TTS/gestures)
    // Board updates come via WS events (move_detected, robot_done)
    fetchJSON('/game/submit_move', {
        method: 'POST',
        body: JSON.stringify({ move: uciMove }),
    }).then(result => {
        _awaitingRobotMove = false;
        if (result.success && result.game_state) {
            updateGameDisplay(result.game_state);
        } else if (!result.success && result.error) {
            showToast(result.error + ' — please try again', 'error', 5000);
            getGameState();
        }
    });
}

function showPromotionDialog(promotionChars) {
    return new Promise((resolve) => {
        _promotionResolve = resolve;
        const dialog = document.getElementById('promotion-dialog');
        const picker = dialog.querySelector('.promotion-picker');
        picker.innerHTML = '';

        const isWhite = simHumanColor === 'white';
        const pieces = {
            'q': isWhite ? '\u2655' : '\u265B',
            'r': isWhite ? '\u2656' : '\u265C',
            'b': isWhite ? '\u2657' : '\u265D',
            'n': isWhite ? '\u2658' : '\u265E',
        };

        for (const ch of ['q', 'r', 'b', 'n']) {
            if (!promotionChars.includes(ch)) continue;
            const btn = document.createElement('button');
            btn.textContent = pieces[ch];
            btn.style.cssText = 'width:64px; height:64px; font-size:40px; border:2px solid #555; border-radius:8px; background:#2a2a3e; color:#fff; cursor:pointer;';
            btn.addEventListener('mouseenter', () => { btn.style.background = '#4a4a6e'; });
            btn.addEventListener('mouseleave', () => { btn.style.background = '#2a2a3e'; });
            btn.addEventListener('click', () => {
                dialog.style.display = 'none';
                _promotionResolve = null;
                resolve(ch);
            });
            picker.appendChild(btn);
        }

        dialog.style.display = 'block';
    });
}

function cancelPromotion() {
    const dialog = document.getElementById('promotion-dialog');
    dialog.style.display = 'none';
    if (_promotionResolve) {
        _promotionResolve(null);
        _promotionResolve = null;
    }
}

function updateModeButtons() {
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === currentMode);
    });
}

function addChatMessage(sender, message) {
    if (sender === 'System') return; // Don't show system messages in chat
    const div = document.createElement('div');
    div.className = `chat-message ${sender === 'You' ? 'user' : 'agent'}`;
    div.innerHTML = `<div class="sender">${sender}</div><div>${message}</div>`;
    elements.chatHistory.appendChild(div);
    elements.chatHistory.scrollTop = elements.chatHistory.scrollHeight;
}

function startStreamingChatMessage(sender) {
    _streamingMsgId++;
    const div = document.createElement('div');
    div.className = `chat-message agent`;
    div.id = `streaming-msg-${_streamingMsgId}`;
    div.innerHTML = `<div class="sender">${sender}</div><div class="streaming-text"></div>`;
    elements.chatHistory.appendChild(div);
    _currentStreamingEl = div.querySelector('.streaming-text');
    elements.chatHistory.scrollTop = elements.chatHistory.scrollHeight;
    return _streamingMsgId;
}

function appendToStreamingMessage(chunk) {
    if (!_currentStreamingEl) {
        // No streaming message started yet — start one with appropriate sender
        const sender = (currentMode === 'watch' && _watchCurrentSender) ? _watchCurrentSender : llmModelName;
        startStreamingChatMessage(sender);
    }
    _currentStreamingEl.textContent += chunk;
    elements.chatHistory.scrollTop = elements.chatHistory.scrollHeight;
}

function finalizeStreamingMessage() {
    _currentStreamingEl = null;
}

function stopCurrentTts() {
    if (currentTtsAudio) {
        currentTtsAudio.pause();
        currentTtsAudio.src = '';
        currentTtsAudio = null;
    }
    isTtsPlaying = false;
    ttsAudioQueue = [];
    // Also stop game WS audio queue
    _gameWsAudioQueue = [];
    _gameWsAudioPlaying = false;
    // Notify voice WS echo suppression
    if (voiceWs && voiceWs.readyState === WebSocket.OPEN) {
        voiceWs.send(JSON.stringify({ type: 'echo_state', speaking: false }));
    }
}

/**
 * Queue a base64-encoded audio blob from the game WebSocket.
 * Each blob typically represents one sentence of TTS audio.
 * Playback starts immediately if nothing is currently playing.
 */
function queueGameWsAudio(base64Data, format) {
    if (!voiceOutputEnabled) return;
    _gameWsAudioQueue.push({ data: base64Data, format: format || 'mp3' });
    // Start playing if idle
    if (!_gameWsAudioPlaying) {
        _playNextGameWsAudio();
    }
}

/**
 * Play the next audio blob from the game WS queue.
 * Chains playback so sentences play sequentially without gaps.
 */
function _playNextGameWsAudio() {
    if (_gameWsAudioQueue.length === 0) {
        _gameWsAudioPlaying = false;
        isTtsPlaying = false;
        if (voiceWs && voiceWs.readyState === WebSocket.OPEN) {
            voiceWs.send(JSON.stringify({ type: 'echo_state', speaking: false }));
        }
        // Deferred teach unlock — audio finished playing
        if (_teachUnlockAfterAudio) {
            _teachUnlockAfterAudio = false;
            teachInputLocked = false;
            renderSimBoard();
        }
        return;
    }
    _gameWsAudioPlaying = true;
    isTtsPlaying = true;

    const item = _gameWsAudioQueue.shift();
    try {
        const binary = atob(item.data);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
        }
        const mimeType = item.format === 'mp3' ? 'audio/mpeg' : `audio/${item.format}`;
        const blob = new Blob([bytes], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        currentTtsAudio = audio;

        // Notify echo suppression
        if (voiceWs && voiceWs.readyState === WebSocket.OPEN) {
            voiceWs.send(JSON.stringify({ type: 'echo_state', speaking: true }));
        }

        audio.onended = () => {
            URL.revokeObjectURL(url);
            currentTtsAudio = null;
            _playNextGameWsAudio();  // Play next sentence
        };
        audio.onerror = () => {
            URL.revokeObjectURL(url);
            currentTtsAudio = null;
            _playNextGameWsAudio();  // Skip to next on error
        };
        audio.play().catch(err => {
            console.error('[TTS] game WS audio playback error:', err);
            currentTtsAudio = null;
            _playNextGameWsAudio();
        });
    } catch (err) {
        console.error('[TTS] game WS audio decode error:', err);
        _playNextGameWsAudio();
    }
}

function interruptAgent() {
    stopCurrentTts();
    fetch('/agent/interrupt', { method: 'POST' }).catch(() => {});
}

// Polling
function startPolling() {
    if (pollingInterval) return;
    pollingInterval = setInterval(async () => {
        if (gameActive) {
            await getGameState();
        }
    }, 3000);
}

function stopPolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
    }
}

// --- Auto-detect status display (server-side polling) ---

function updateAutoDetectStatus(text) {
    const el = document.getElementById('auto-detect-status');
    if (!el) return;
    el.textContent = text || '';
}

// =========================
// Pipeline Debug
// =========================
async function runDebugPipeline() {
    const statusEl = document.getElementById('debug-status');
    const stepsEl = document.getElementById('debug-steps');
    statusEl.textContent = 'Running pipeline...';
    stepsEl.innerHTML = '';

    try {
        const response = await fetch(`${VISION_URL}/capture/debug`, { method: 'POST' });
        const result = await response.json();

        if (!result.success && (!result.steps || result.steps.length === 0)) {
            statusEl.textContent = `Error: ${result.error || 'Unknown error'}`;
            return;
        }

        const failedAt = result.failed_at;
        const statusText = failedAt
            ? `Failed at: ${failedAt} (${result.total_duration_ms.toFixed(0)}ms)`
            : `OK — ${Object.keys(result.piece_positions || {}).length} pieces (${result.total_duration_ms.toFixed(0)}ms)`;
        statusEl.textContent = statusText;
        statusEl.style.color = failedAt ? 'var(--danger)' : 'var(--success)';

        result.steps.forEach((step, index) => {
            stepsEl.appendChild(createDebugStepCard(step, index));
        });

        // Render detected pieces on the manual chess board (if present)
        if (result.piece_positions && Object.keys(result.piece_positions).length > 0) {
            const manualBoard = document.getElementById('manual-board');
            if (manualBoard) {
                renderManualBoard(result.piece_positions);
            }
            const manualStatus = document.getElementById('manual-status');
            if (manualStatus) {
                manualStatus.textContent = `Debug pipeline: ${Object.keys(result.piece_positions).length} pieces detected.`;
            }
        }

    } catch (error) {
        statusEl.textContent = `Error: ${error.message}`;
        statusEl.style.color = 'var(--danger)';
    }
}

function createDebugStepCard(step, index) {
    const card = document.createElement('div');
    card.className = 'debug-step-card' + (step.success ? '' : ' failed');

    const icon = step.success ? '\u2705' : '\u274C';
    const duration = step.duration_ms != null ? `${step.duration_ms.toFixed(0)}ms` : '';

    const headerDiv = document.createElement('div');
    headerDiv.className = 'debug-step-header';
    headerDiv.innerHTML = `
        <span class="debug-step-number">[${index + 1}]</span>
        <span class="debug-step-label">${step.step_label}</span>
        <span class="debug-step-icon">${icon}</span>
        <span class="debug-step-time">${duration}</span>
    `;
    headerDiv.addEventListener('click', () => toggleDebugStep(index));
    card.appendChild(headerDiv);

    const bodyDiv = document.createElement('div');
    bodyDiv.className = 'debug-step-body';
    bodyDiv.id = `debug-step-body-${index}`;

    if (step.image_base64) {
        const img = document.createElement('img');
        img.src = `data:image/jpeg;base64,${step.image_base64}`;
        img.className = 'debug-step-image';
        bodyDiv.appendChild(img);
    }

    if (step.error) {
        const errDiv = document.createElement('div');
        errDiv.className = 'debug-step-error';
        errDiv.textContent = step.error;
        bodyDiv.appendChild(errDiv);
    }

    if (step.metadata && Object.keys(step.metadata).length > 0) {
        const pre = document.createElement('pre');
        pre.className = 'debug-step-meta';
        pre.textContent = formatDebugMetadata(step.metadata);
        bodyDiv.appendChild(pre);
    }

    card.appendChild(bodyDiv);
    return card;
}

function toggleDebugStep(index) {
    const body = document.getElementById(`debug-step-body-${index}`);
    if (body) {
        body.style.display = body.style.display === 'none' ? 'block' : 'none';
    }
}

function formatDebugMetadata(meta) {
    if (!meta) return '';
    return Object.entries(meta).map(([k, v]) => {
        if (Array.isArray(v)) {
            if (v.length > 10) return `${k}: [${v.length} items]`;
            return `${k}: ${JSON.stringify(v)}`;
        }
        if (typeof v === 'number') return `${k}: ${v}`;
        return `${k}: ${JSON.stringify(v)}`;
    }).join('\n');
}

// Event Listeners
function setupEventListeners() {
    // Tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });

    // Mode buttons
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.addEventListener('click', () => setMode(btn.dataset.mode));
    });

    // Teach mode buttons
    const btnTeachHint = document.getElementById('btn-teach-hint');
    const btnTeachNext = document.getElementById('btn-teach-next');
    const btnTeachStop = document.getElementById('btn-teach-stop');
    const btnTeachMoved = document.getElementById('btn-teach-moved');
    if (btnTeachHint) btnTeachHint.addEventListener('click', teachHint);
    if (btnTeachNext) btnTeachNext.addEventListener('click', teachNextStep);
    if (btnTeachStop) btnTeachStop.addEventListener('click', teachStop);
    if (btnTeachMoved) btnTeachMoved.addEventListener('click', teachManualDetect);

    // Language buttons (hidden, kept for compatibility)
    document.querySelectorAll('.lang-btn').forEach(btn => {
        btn.addEventListener('click', () => setLanguage(btn.dataset.lang));
    });
    // Language dropdown (Agent tab)
    const langSelect = document.getElementById('language-select');
    const customLangRow = document.getElementById('custom-lang-row');
    if (langSelect) {
        langSelect.addEventListener('change', () => {
            if (langSelect.value === 'custom') {
                if (customLangRow) customLangRow.style.display = 'flex';
            } else {
                if (customLangRow) customLangRow.style.display = 'none';
                setLanguage(langSelect.value);
            }
        });
    }
    const btnSetCustomLang = document.getElementById('btn-set-custom-lang');
    const customLangInput = document.getElementById('custom-language');
    if (btnSetCustomLang && customLangInput) {
        btnSetCustomLang.addEventListener('click', () => {
            const lang = customLangInput.value.trim();
            if (lang) setLanguage(lang);
        });
        customLangInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                const lang = customLangInput.value.trim();
                if (lang) setLanguage(lang);
            }
        });
    }

    // Lesson management (Agent tab)
    const btnLessonGen = document.getElementById('btn-lesson-generate');
    const btnLessonRandom = document.getElementById('btn-lesson-random');
    const btnLessonSave = document.getElementById('btn-lesson-save');
    const btnLessonDelete = document.getElementById('btn-lesson-delete');
    const agentLessonList = document.getElementById('agent-lesson-list');
    if (btnLessonGen) btnLessonGen.addEventListener('click', generateLesson);
    if (btnLessonRandom) btnLessonRandom.addEventListener('click', randomLesson);
    if (btnLessonSave) btnLessonSave.addEventListener('click', saveGeneratedLesson);
    if (btnLessonDelete) btnLessonDelete.addEventListener('click', deleteSelectedLesson);
    if (agentLessonList) agentLessonList.addEventListener('change', () => { _onAgentLessonChange(); showLessonDetail(agentLessonList.value); });
    const gameLessonSelect = document.getElementById('lesson-select');
    if (gameLessonSelect) gameLessonSelect.addEventListener('change', _onGameLessonChange);

    // Character management
    const btnCharGen = document.getElementById('btn-char-generate');
    const btnCharRandom = document.getElementById('btn-char-random');
    const btnCharSave = document.getElementById('btn-char-save');
    const btnCharDelete = document.getElementById('btn-char-delete');
    const charSelect = document.getElementById('character-select');
    const charDescArea = document.getElementById('character-desc');
    if (btnCharGen) btnCharGen.addEventListener('click', generateCharacterDesc);
    if (btnCharRandom) btnCharRandom.addEventListener('click', randomCharacter);
    if (btnCharSave) btnCharSave.addEventListener('click', saveCharacter);
    if (btnCharDelete) btnCharDelete.addEventListener('click', deleteCharacter);
    if (charSelect) charSelect.addEventListener('change', onCharacterSelected);
    if (charDescArea) charDescArea.addEventListener('input', _updateCharCount);

    // Game over dismiss
    const btnGameOverDismiss = document.getElementById('btn-game-over-dismiss');
    if (btnGameOverDismiss) btnGameOverDismiss.addEventListener('click', dismissGameOverOverlay);

    // Game controls
    elements.btnStart.addEventListener('click', startGame);
    // Toggle camera visibility when simulation checkbox changes + save
    const simCheckbox = document.getElementById('simulation-mode');
    if (simCheckbox) {
        // Restore saved state
        const savedSim = localStorage.getItem('simulation_mode');
        if (savedSim !== null) simCheckbox.checked = savedSim === 'true';
        document.body.classList.toggle('simulation-mode', simCheckbox.checked);

        simCheckbox.addEventListener('change', () => {
            document.body.classList.toggle('simulation-mode', simCheckbox.checked);
            localStorage.setItem('simulation_mode', simCheckbox.checked);
        });
    }
    elements.btnStop.addEventListener('click', stopGame);
    if (elements.btnHumanMoved) elements.btnHumanMoved.addEventListener('click', notifyHumanMoved);
    if (elements.btnSubmitMove) elements.btnSubmitMove.addEventListener('click', submitManualMove);
    if (elements.inputManualMove) elements.inputManualMove.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') submitManualMove();
    });
    const btnValidate = document.getElementById('btn-validate-board');
    if (btnValidate) btnValidate.addEventListener('click', validateBoard);

    // Chat — interrupt AI speech when user starts typing or sends
    elements.btnSend.addEventListener('click', () => sendChat(elements.chatInput.value));
    elements.chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendChat(elements.chatInput.value);
    });
    elements.chatInput.addEventListener('focus', () => {});
    elements.chatInput.addEventListener('input', () => {});

    // Chat mic button (hold-to-talk from main panel) — interrupt on press
    if (elements.btnChatMic) {
        elements.btnChatMic.addEventListener('mousedown', (e) => {
            e.preventDefault();
            startChatMicRecording();
        });
        elements.btnChatMic.addEventListener('mouseup', () => {
            stopChatMicRecording();
        });
        elements.btnChatMic.addEventListener('mouseleave', () => {
            stopChatMicRecording();
        });
        // Touch support for mobile
        elements.btnChatMic.addEventListener('touchstart', (e) => {
            e.preventDefault();
            startChatMicRecording();
        });
        elements.btnChatMic.addEventListener('touchend', () => {
            stopChatMicRecording();
        });
    }

    // Voice output toggle
    if (elements.voiceOutputEnabled) {
        elements.voiceOutputEnabled.addEventListener('change', (e) => {
            voiceOutputEnabled = e.target.checked;
            // Notify backend to skip TTS synthesis when disabled
            fetchJSON('/agent/tts_enabled', {
                method: 'POST',
                body: JSON.stringify({ enabled: voiceOutputEnabled }),
            });
        });
    }

    // Voice
    elements.voiceEnabled.addEventListener('change', (e) => {
        voiceEnabled = e.target.checked;
        if (voiceEnabled) {
            connectVoiceWs();
        } else {
            disconnectVoiceWs();
        }
        addChatMessage('System', `Voice ${voiceEnabled ? 'enabled' : 'disabled'}`);
    });
    elements.btnSpeakTest.addEventListener('click', testTTS);

    // Listening mode toggle
    if (elements.voiceListeningMode) {
        elements.voiceListeningMode.addEventListener('change', (e) => {
            voiceListeningMode = e.target.value;
            localStorage.setItem('voice_listening_mode', voiceListeningMode);
            if (voiceWs && voiceWs.readyState === WebSocket.OPEN) {
                voiceWs.send(JSON.stringify({ type: 'set_mode', mode: voiceListeningMode }));
            }
            updateMicButton();
            _toggleVadSettings();
            // Start/stop continuous stream
            if (voiceListeningMode === 'always_on' && voiceEnabled) {
                startContinuousStream();
            } else {
                stopContinuousStream();
            }
        });
    }

    // VAD settings save
    const btnVadSave = document.getElementById('btn-vad-save');
    if (btnVadSave) btnVadSave.addEventListener('click', _saveVadSettings);

    // API Settings save
    const btnSaveSettings = document.getElementById('btn-save-settings');
    if (btnSaveSettings) btnSaveSettings.addEventListener('click', saveAgentSettings);

    // Mic button — push-to-talk or VAD mute/unmute toggle
    if (elements.btnMic) {
        elements.btnMic.addEventListener('mousedown', (e) => {
            e.preventDefault();
            if (voiceListeningMode === 'always_on') {
                // Toggle VAD mute/unmute
                if (vadInterval) { stopContinuousStream(); } else { startContinuousStream(); }
                updateMicButton();
                return;
            }
            if (voiceListeningMode === 'push_to_talk' && voiceEnabled) {
                startPttRecording();
            }
        });
        elements.btnMic.addEventListener('mouseup', (e) => {
            e.preventDefault();
            if (voiceListeningMode === 'push_to_talk') {
                stopPttRecording();
            }
        });
        elements.btnMic.addEventListener('mouseleave', (e) => {
            if (voiceListeningMode === 'push_to_talk') {
                stopPttRecording();
            }
        });
        // Touch support
        elements.btnMic.addEventListener('touchstart', (e) => {
            e.preventDefault();
            if (voiceListeningMode === 'always_on') {
                if (vadInterval) { stopContinuousStream(); } else { startContinuousStream(); }
                updateMicButton();
                return;
            }
            if (voiceListeningMode === 'push_to_talk' && voiceEnabled) {
                startPttRecording();
            }
        });
        elements.btnMic.addEventListener('touchend', (e) => {
            e.preventDefault();
            if (voiceListeningMode === 'push_to_talk') {
                stopPttRecording();
            }
        });
    }

    // Mic toggle button — behavior depends on voice mode (Agent tab setting)
    const micToggleBtn = document.getElementById('mic-toggle-btn');
    if (micToggleBtn) {
        micToggleBtn.addEventListener('click', async () => {
            // Auto-enable voice if not enabled
            if (!voiceEnabled) {
                voiceEnabled = true;
                if (elements.voiceEnabled) elements.voiceEnabled.checked = true;
                connectVoiceWs();
                // Wait briefly for WS to connect
                await new Promise(r => setTimeout(r, 500));
            }

            if (voiceListeningMode === 'always_on') {
                // Always-on: toggle VAD on/off (like enable checkbox)
                if (vadInterval) {
                    stopContinuousStream();
                    micToggleBtn.classList.remove('recording');
                } else {
                    startContinuousStream();
                    micToggleBtn.classList.add('recording');
                }
            } else {
                // Push-to-talk: click start, click stop & send
                if (micToggleBtn.classList.contains('recording')) {
                    stopPttRecording();
                    micToggleBtn.classList.remove('recording');
                } else {
                    startPttRecording();
                    micToggleBtn.classList.add('recording');
                }
            }
        });
    }

    // Manual test buttons
    if (elements.btnCameraFrame) elements.btnCameraFrame.addEventListener('click', testCameraFrame);
    if (elements.btnDepthFrame) elements.btnDepthFrame.addEventListener('click', testDepthFrame);
    if (elements.btnDetectApriltag) elements.btnDetectApriltag.addEventListener('click', testDetectApriltag);
    if (elements.btnCaptureBoard) elements.btnCaptureBoard.addEventListener('click', testCaptureBoard);
    // Live Streaming (Main View) buttons removed from Vision tab
    // Vision tab debug camera
    const btnVcamRgb = document.getElementById('btn-vcam-rgb');
    const btnVcamDepth = document.getElementById('btn-vcam-depth');
    const btnVcamYolo = document.getElementById('btn-vcam-yolo');
    const btnVcamStop = document.getElementById('btn-vcam-stop');
    if (btnVcamRgb) btnVcamRgb.addEventListener('click', () => startVcamStream('rgb'));
    if (btnVcamDepth) btnVcamDepth.addEventListener('click', () => startVcamStream('depth'));
    if (btnVcamYolo) btnVcamYolo.addEventListener('click', () => startVcamStream('yolo'));
    if (btnVcamStop) btnVcamStop.addEventListener('click', stopVcamStream);
    if (elements.btnSaveTraining) elements.btnSaveTraining.addEventListener('click', saveTrainingImage);
    if (elements.btnRobotHome) elements.btnRobotHome.addEventListener('click', testRobotHome);
    if (elements.btnRobotSleep) elements.btnRobotSleep.addEventListener('click', testRobotSleep);
    if (elements.btnRobotWork) elements.btnRobotWork.addEventListener('click', goToWork);
    if (elements.btnSetWork) elements.btnSetWork.addEventListener('click', setWorkPosition);
    if (elements.btnRobotVision) elements.btnRobotVision.addEventListener('click', goToVision);
    if (elements.btnSetVision) elements.btnSetVision.addEventListener('click', setVisionPosition);
    if (elements.btnRobotCaptureZone) elements.btnRobotCaptureZone.addEventListener('click', goToCaptureZone);
    if (elements.btnSetCaptureZone) elements.btnSetCaptureZone.addEventListener('click', setCaptureZone);
    if (elements.btnRobotPromotionQueen) elements.btnRobotPromotionQueen.addEventListener('click', goToPromotionQueen);
    if (elements.btnSetPromotionQueen) elements.btnSetPromotionQueen.addEventListener('click', setPromotionQueen);
    if (elements.btnGripperOpen) elements.btnGripperOpen.addEventListener('click', testGripperOpen);
    if (elements.btnGripperClose) elements.btnGripperClose.addEventListener('click', testGripperClose);
    if (elements.btnGetPositions) elements.btnGetPositions.addEventListener('click', getPositions);
    if (elements.btnGripperSet) elements.btnGripperSet.addEventListener('click', setGripper);

    // Cartesian Jog and Joint Jog buttons removed from Robot tab

    // Waypoint buttons
    if (elements.btnSaveWaypoint) elements.btnSaveWaypoint.addEventListener('click', saveWaypoint);
    if (elements.btnGotoWaypoint) elements.btnGotoWaypoint.addEventListener('click', gotoWaypoint);
    if (elements.btnDeleteWaypoint) elements.btnDeleteWaypoint.addEventListener('click', deleteWaypoint);
    if (elements.btnRefreshWaypoints) elements.btnRefreshWaypoints.addEventListener('click', refreshWaypoints);

    // Robot connection buttons
    if (elements.btnRobotConnect) elements.btnRobotConnect.addEventListener('click', () => connectRobot(false));
    if (elements.btnRobotMock) elements.btnRobotMock.addEventListener('click', () => connectRobot(true));
    if (elements.btnRobotDisconnect) elements.btnRobotDisconnect.addEventListener('click', disconnectRobot);

    // Motor recovery buttons
    const btnRebootWrist = document.getElementById('btn-reboot-wrist');
    const btnRebootAll = document.getElementById('btn-reboot-all');
    const btnTorqueOff = document.getElementById('btn-torque-off');
    const btnTorqueOn = document.getElementById('btn-torque-on');
    if (btnRebootWrist) btnRebootWrist.addEventListener('click', rebootWrist);
    if (btnRebootAll) btnRebootAll.addEventListener('click', rebootAllMotors);
    if (btnTorqueOff) btnTorqueOff.addEventListener('click', disableTorque);
    if (btnTorqueOn) btnTorqueOn.addEventListener('click', enableTorque);

    // Gesture buttons
    const btnGestureRecord = document.getElementById('btn-gesture-record');
    const btnGestureStopRecord = document.getElementById('btn-gesture-stop-record');
    const btnGesturePlay = document.getElementById('btn-gesture-play');
    const btnGestureStopPlay = document.getElementById('btn-gesture-stop-play');
    const btnGestureDelete = document.getElementById('btn-gesture-delete');
    const btnGestureRefresh = document.getElementById('btn-gesture-refresh');
    if (btnGestureRecord) btnGestureRecord.addEventListener('click', startGestureRecording);
    if (btnGestureStopRecord) btnGestureStopRecord.addEventListener('click', stopGestureRecording);
    if (btnGesturePlay) btnGesturePlay.addEventListener('click', playGesture);
    if (btnGestureStopPlay) btnGestureStopPlay.addEventListener('click', stopGesturePlayback);
    if (btnGestureDelete) btnGestureDelete.addEventListener('click', deleteGesture);
    if (btnGestureRefresh) btnGestureRefresh.addEventListener('click', loadGestures);
    const gestureList = document.getElementById('gesture-list');
    if (gestureList) gestureList.addEventListener('change', onGestureSelected);

    // Calibration buttons
    const btnDetectCalibTags = document.getElementById('btn-detect-calib-tags');
    const btnRecordCalibPoint = document.getElementById('btn-record-calib-point');
    const btnComputeCalib = document.getElementById('btn-compute-calib');
    const btnClearCalib = document.getElementById('btn-clear-calib');
    const btnLoadCalib = document.getElementById('btn-load-calib');
    const btnTestTransform = document.getElementById('btn-test-transform');
    if (btnDetectCalibTags) btnDetectCalibTags.addEventListener('click', detectCalibTags);
    if (btnRecordCalibPoint) btnRecordCalibPoint.addEventListener('click', recordCalibPoint);
    if (btnComputeCalib) btnComputeCalib.addEventListener('click', computeCalibration);
    if (btnClearCalib) btnClearCalib.addEventListener('click', clearCalibPoints);
    if (btnLoadCalib) btnLoadCalib.addEventListener('click', loadCalibration);
    if (btnTestTransform) btnTestTransform.addEventListener('click', testTransform);

    // Manual Chess UI removed from Vision tab (buttons no longer in HTML)
    // Capture Board (Vision) button is still in Pipeline Debug section
    const btnManualCapture = document.getElementById('btn-manual-capture');
    if (btnManualCapture) btnManualCapture.addEventListener('click', captureForManualChess);

    // Square Position Teaching buttons
    const btnSqRecord = document.getElementById('btn-sq-record');
    const btnSqGoto = document.getElementById('btn-sq-goto');
    const btnSqDelete = document.getElementById('btn-sq-delete');
    const btnSqLoad = document.getElementById('btn-sq-load');
    const btnSqClear = document.getElementById('btn-sq-clear');
    if (btnSqRecord) btnSqRecord.addEventListener('click', recordSquarePosition);
    if (btnSqGoto) btnSqGoto.addEventListener('click', gotoSquarePosition);
    const btnSqGotoLow = document.getElementById('btn-sq-goto-low');
    if (btnSqGotoLow) btnSqGotoLow.addEventListener('click', gotoSquarePositionLow);
    if (btnSqDelete) btnSqDelete.addEventListener('click', deleteSquarePosition);
    if (btnSqLoad) btnSqLoad.addEventListener('click', loadSquarePositions);
    if (btnSqClear) btnSqClear.addEventListener('click', clearAllSquarePositions);
    const btnSqAutotest = document.getElementById('btn-sq-autotest');
    const btnSqAutotestStop = document.getElementById('btn-sq-autotest-stop');
    if (btnSqAutotest) btnSqAutotest.addEventListener('click', autotestAllSquares);
    if (btnSqAutotestStop) btnSqAutotestStop.addEventListener('click', stopAutotest);

    // Board Surface Z button
    const btnTeachBoardZ = document.getElementById('btn-teach-board-z');
    if (btnTeachBoardZ) btnTeachBoardZ.addEventListener('click', teachBoardSurfaceZ);

    // Teach 4 Corners removed from Robot tab

    // Level 2 waypoint buttons removed

    // Debug pipeline button
    const btnRunDebug = document.getElementById('btn-run-debug');
    if (btnRunDebug) btnRunDebug.addEventListener('click', runDebugPipeline);

    // Move to tag/XYZ buttons
    const btnMoveToTag = document.getElementById('btn-move-to-tag');
    const btnMoveXYZ = document.getElementById('btn-move-xyz');
    if (btnMoveToTag) btnMoveToTag.addEventListener('click', moveToTag);
    if (btnMoveXYZ) btnMoveXYZ.addEventListener('click', moveToXYZ);

    // Robot tab camera stream
    const robotCamImg = document.getElementById('robot-camera-image');
    const btnRobotStreamRgb = document.getElementById('btn-robot-stream-rgb');
    const btnRobotStreamDepth = document.getElementById('btn-robot-stream-depth');
    const btnRobotStreamStop = document.getElementById('btn-robot-stream-stop');
    if (btnRobotStreamRgb) btnRobotStreamRgb.addEventListener('click', () => {
        if (robotCamImg) robotCamImg.src = `${VISION_URL}/camera/stream/rgb`;
    });
    if (btnRobotStreamDepth) btnRobotStreamDepth.addEventListener('click', () => {
        if (robotCamImg) robotCamImg.src = `${VISION_URL}/camera/stream/depth`;
    });
    if (btnRobotStreamStop) btnRobotStreamStop.addEventListener('click', () => {
        if (robotCamImg) robotCamImg.src = '';
    });
}

// ==================== MANUAL CHESS FUNCTIONS ====================

let manualChessPieces = {};  // { "e2": "white_pawn", ... }
let manualSelectedSquare = null;
let gameActive_manual = false;  // Whether a tracker game is active

// Unicode chess piece symbols
const PIECE_UNICODE = {
    'white_king': '\u2654', 'white_queen': '\u2655', 'white_rook': '\u2656',
    'white_bishop': '\u2657', 'white_knight': '\u2658', 'white_pawn': '\u2659',
    'black_king': '\u265A', 'black_queen': '\u265B', 'black_rook': '\u265C',
    'black_bishop': '\u265D', 'black_knight': '\u265E', 'black_pawn': '\u265F',
};

function pieceLabel(pieceData) {
    if (!pieceData) return '';
    // Handle both string and object formats
    const pieceName = typeof pieceData === 'string' ? pieceData : pieceData.piece;
    if (!pieceName) return '';
    return PIECE_UNICODE[pieceName] || pieceName.substring(0, 2).toUpperCase();
}

function renderManualBoard(piecePositions) {
    manualChessPieces = piecePositions || {};
    const board = document.getElementById('manual-board');
    if (!board) return;
    board.innerHTML = '';

    const files = ['a','b','c','d','e','f','g','h'];

    for (let rank = 8; rank >= 1; rank--) {
        // Row label
        const label = document.createElement('div');
        label.textContent = rank;
        label.style.cssText = 'display:flex; align-items:center; justify-content:center; width:20px; font-weight:bold; font-size:11px;';
        board.appendChild(label);

        for (let f = 0; f < 8; f++) {
            const square = files[f] + rank;
            const cell = document.createElement('div');
            const isLight = (f + rank) % 2 === 1;
            const piece = manualChessPieces[square];

            cell.dataset.square = square;
            cell.textContent = pieceLabel(piece);
            const pieceName = piece ? (typeof piece === 'string' ? piece : piece.piece) : null;
            cell.style.cssText = `
                display:flex; align-items:center; justify-content:center;
                aspect-ratio:1; cursor:pointer; font-weight:bold; font-size:18px;
                border: 1px solid #555;
                background: ${isLight ? '#b58863' : '#f0d9b5'};
                color: ${pieceName && pieceName.startsWith('black') ? '#1a1a1a' : '#fff'};
                ${pieceName && pieceName.startsWith('white') ? 'text-shadow: 0 0 3px #000;' : ''}
            `;

            cell.addEventListener('click', () => selectManualSquare(square));
            board.appendChild(cell);
        }
    }

    // Bottom file labels row
    const spacer = document.createElement('div');
    spacer.style.width = '20px';
    board.appendChild(spacer);
    for (const f of files) {
        const lbl = document.createElement('div');
        lbl.textContent = f;
        lbl.style.cssText = 'display:flex; align-items:center; justify-content:center; font-weight:bold; font-size:11px;';
        board.appendChild(lbl);
    }
}

function selectManualSquare(square) {
    manualSelectedSquare = square;
    const piece = manualChessPieces[square];
    const selectedEl = document.getElementById('manual-selected');
    const pickBtn = document.getElementById('btn-manual-pickup');
    const placeBtn = document.getElementById('btn-manual-place');

    // Get piece name (handle both string and object formats)
    const pieceName = piece ? (typeof piece === 'string' ? piece : piece.piece) : null;

    if (pieceName) {
        if (selectedEl) selectedEl.textContent = `${square}: ${pieceName}`;
        if (pickBtn) {
            pickBtn.disabled = false;
            pickBtn.textContent = `Pick Up ${square}`;
        }
    } else {
        if (selectedEl) selectedEl.textContent = `${square}: empty`;
        if (pickBtn) {
            pickBtn.disabled = true;
            pickBtn.textContent = 'Pick Up';
        }
    }
    if (placeBtn) {
        placeBtn.disabled = false;
        placeBtn.textContent = `Place ${square}`;
    }

    // Highlight selected cell
    document.querySelectorAll('#manual-board div[data-square]').forEach(cell => {
        if (cell.dataset.square === square) {
            cell.style.outline = '3px solid #007bff';
            cell.style.outlineOffset = '-3px';
        } else {
            cell.style.outline = 'none';
        }
    });
}

function updateGameTrackerInfo(data) {
    const infoEl = document.getElementById('game-tracker-info');
    const turnEl = document.getElementById('game-turn-info');
    const detectBtn = document.getElementById('btn-game-detect');

    if (!data || !data.active) {
        if (turnEl) turnEl.innerHTML = 'No active game';
        if (detectBtn) detectBtn.disabled = true;
        gameActive_manual = false;
        return;
    }

    gameActive_manual = true;
    if (detectBtn) detectBtn.disabled = false;

    const turn = data.turn || '?';
    const moveCount = data.move_count || 0;
    const fen = data.fen || '';
    const fenShort = fen.split(' ')[0];

    if (turnEl) {
        turnEl.innerHTML = `Turn: <strong>${turn}</strong> | Move #${moveCount + 1} | FEN: ${fenShort}`;
    }
}

function updateMoveHistory(history) {
    const el = document.getElementById('game-move-history');
    if (!el) return;

    if (!history || history.length === 0) {
        el.innerHTML = '<em>No moves yet</em>';
        return;
    }

    let html = '';
    history.forEach((m, i) => {
        const capture = m.is_capture ? ' x' + (m.captured_piece || '') : '';
        html += `<div>${i + 1}. ${m.uci_move} (${m.piece || '?'})${capture}</div>`;
    });
    el.innerHTML = html;
    el.scrollTop = el.scrollHeight;
}

async function initGame() {
    const statusEl = document.getElementById('manual-status');
    const initBtn = document.getElementById('btn-game-init');
    if (initBtn) initBtn.disabled = true;
    if (statusEl) statusEl.textContent = 'Initializing game (capturing board)...';

    try {
        const response = await fetch(`${ROBOT_URL}/game/init`, { method: 'POST' });
        const result = await response.json();

        if (result.success) {
            // Render board from tracker's piece_map initially
            renderManualBoard(result.piece_map);
            updateGameTrackerInfo({
                active: true,
                turn: result.turn,
                move_count: 0,
                fen: result.fen,
            });
            updateMoveHistory([]);

            if (statusEl) {
                statusEl.textContent = `Game initialized: ${result.occupied_count} pieces detected (expected ${result.expected_count}), white=${result.white_side}. Running vision debug...`;
            }

            // Run debug pipeline to show actual vision-detected pieces on board
            await runDebugPipeline();
        } else {
            if (statusEl) statusEl.textContent = `Init failed: ${result.error}`;
        }
    } catch (error) {
        if (statusEl) statusEl.textContent = `Error: ${error.message}`;
    }

    if (initBtn) initBtn.disabled = false;
}

async function detectMove() {
    const statusEl = document.getElementById('manual-status');
    const detectBtn = document.getElementById('btn-game-detect');
    if (detectBtn) detectBtn.disabled = true;
    if (statusEl) statusEl.textContent = 'Detecting move (capturing board)...';

    try {
        const response = await fetch(`${ROBOT_URL}/game/detect_move`, { method: 'POST' });
        const result = await response.json();

        if (result.success) {
            // Update board from tracker's piece_map
            renderManualBoard(result.piece_map);
            updateGameTrackerInfo({
                active: true,
                turn: result.turn,
                move_count: result.move_number,
                fen: result.fen,
            });

            // Refresh full history from server
            const stateResp = await fetch(`${ROBOT_URL}/game/state`);
            const stateData = await stateResp.json();
            updateMoveHistory(stateData.move_history || []);

            const capture = result.is_capture ? ` captures ${result.captured_piece}` : '';
            if (statusEl) {
                statusEl.textContent = `Move ${result.move_number}: ${result.uci_move} (${result.piece})${capture}. ${result.turn}'s turn.`;
            }
        } else {
            let errMsg = `Detection failed: ${result.error}`;
            if (result.emptied) errMsg += ` | emptied: [${result.emptied}] filled: [${result.filled}]`;
            if (result.candidates && result.candidates.length > 0) errMsg += ` | candidates: ${result.candidates.join(', ')}`;
            if (statusEl) statusEl.textContent = errMsg;
        }
    } catch (error) {
        if (statusEl) statusEl.textContent = `Error: ${error.message}`;
    }

    if (detectBtn) detectBtn.disabled = !gameActive_manual;
}

async function resetGame() {
    const statusEl = document.getElementById('manual-status');

    try {
        await fetch(`${ROBOT_URL}/game/reset`, { method: 'POST' });
        renderManualBoard({});
        updateGameTrackerInfo(null);
        updateMoveHistory([]);
        manualSelectedSquare = null;
        if (statusEl) statusEl.textContent = 'Game reset. Click "Init Game" to start.';
    } catch (error) {
        if (statusEl) statusEl.textContent = `Error: ${error.message}`;
    }
}

async function captureForManualChess() {
    const statusEl = document.getElementById('manual-status');
    if (statusEl) statusEl.textContent = 'Capturing board (vision only)...';

    try {
        const response = await fetch(`${VISION_URL}/capture`, { method: 'POST' });
        const result = await response.json();

        if (result.success) {
            const positions = result.piece_positions || {};
            renderManualBoard(positions);
            const count = Object.keys(positions).length;
            if (statusEl) statusEl.textContent = `Vision detected ${count} piece(s). Click a square to select.`;
        } else {
            if (statusEl) statusEl.textContent = `Capture failed: ${result.error}`;
        }
    } catch (error) {
        if (statusEl) statusEl.textContent = `Error: ${error.message}`;
    }
}

async function manualPickUp() {
    if (!manualSelectedSquare) return;
    const pieceData = manualChessPieces[manualSelectedSquare];
    if (!pieceData) return;

    // Handle both string and object formats
    const piece = typeof pieceData === 'string' ? pieceData : pieceData.piece;
    if (!piece) return;

    const statusEl = document.getElementById('manual-status');
    const pickBtn = document.getElementById('btn-manual-pickup');
    if (pickBtn) pickBtn.disabled = true;

    // Extract piece type (strip color prefix)
    let pieceType = piece;
    for (const prefix of ['white_', 'black_']) {
        if (pieceType.startsWith(prefix)) {
            pieceType = pieceType.substring(prefix.length);
            break;
        }
    }

    if (statusEl) statusEl.textContent = `Picking up ${piece} from ${manualSelectedSquare}...`;

    try {
        const response = await fetch(`${ROBOT_URL}/manual_pick`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                square: manualSelectedSquare,
                piece_type: pieceType,
            }),
        });
        const result = await response.json();

        if (result.success) {
            if (statusEl) statusEl.textContent = `Done: ${result.message}`;
        } else {
            if (statusEl) statusEl.textContent = `Failed: ${result.error}`;
        }
    } catch (error) {
        if (statusEl) statusEl.textContent = `Error: ${error.message}`;
    }

    if (pickBtn) pickBtn.disabled = false;
}

async function manualPlace() {
    if (!manualSelectedSquare) return;

    const statusEl = document.getElementById('manual-status');
    const placeBtn = document.getElementById('btn-manual-place');
    if (placeBtn) placeBtn.disabled = true;

    // Use piece type if known on that square, otherwise default to pawn
    const pieceData = manualChessPieces[manualSelectedSquare];
    let pieceType = 'pawn';
    if (pieceData) {
        const piece = typeof pieceData === 'string' ? pieceData : pieceData.piece;
        if (piece) {
            pieceType = piece;
            for (const prefix of ['white_', 'black_']) {
                if (pieceType.startsWith(prefix)) {
                    pieceType = pieceType.substring(prefix.length);
                    break;
                }
            }
        }
    }

    if (statusEl) statusEl.textContent = `Placing on ${manualSelectedSquare}...`;

    try {
        const response = await fetch(`${ROBOT_URL}/manual_place`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                square: manualSelectedSquare,
                piece_type: pieceType,
            }),
        });
        const result = await response.json();

        if (result.success) {
            if (statusEl) statusEl.textContent = `Done: ${result.message}`;
        } else {
            if (statusEl) statusEl.textContent = `Failed: ${result.error}`;
        }
    } catch (error) {
        if (statusEl) statusEl.textContent = `Error: ${error.message}`;
    }

    if (placeBtn) placeBtn.disabled = false;
}

let pickPlaceTestRunning = false;

async function autoPickPlaceTest(ranks) {
    if (pickPlaceTestRunning) return;
    pickPlaceTestRunning = true;

    const btnBlack = document.getElementById('btn-pickplace-black');
    const btnWhite = document.getElementById('btn-pickplace-white');
    const btnStop = document.getElementById('btn-pickplace-stop');
    const statusEl = document.getElementById('manual-status');
    const resultEl = document.getElementById('pickplace-test-result');
    if (btnBlack) btnBlack.style.display = 'none';
    if (btnWhite) btnWhite.style.display = 'none';
    if (btnStop) btnStop.style.display = '';
    if (resultEl) { resultEl.style.display = 'block'; resultEl.textContent = ''; }

    const testSquares = [];
    const files = 'abcdefgh';
    for (const rank of ranks) {
        for (const f of files) testSquares.push(f + rank);
    }

    const failed = [];
    let tested = 0;

    for (const square of testSquares) {
        if (!pickPlaceTestRunning) break;

        // Get piece type from board state
        const pieceData = manualChessPieces[square];
        let pieceType = 'pawn';
        if (pieceData) {
            const piece = typeof pieceData === 'string' ? pieceData : pieceData.piece;
            if (piece) {
                pieceType = piece;
                for (const prefix of ['white_', 'black_']) {
                    if (pieceType.startsWith(prefix)) {
                        pieceType = pieceType.substring(prefix.length);
                        break;
                    }
                }
            }
        }

        tested++;
        selectManualSquare(square);

        // === PICK ===
        if (statusEl) statusEl.textContent = `[${tested}/${testSquares.length}] Picking ${pieceType} from ${square}...`;
        if (resultEl) resultEl.textContent += `${square} (${pieceType}): pick...`;
        resultEl.scrollTop = resultEl.scrollHeight;

        try {
            const pickResp = await fetch(`${ROBOT_URL}/manual_pick`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ square: square, piece_type: pieceType, skip_return_to_work: true }),
            });
            const pickResult = await pickResp.json();
            if (!pickResult.success) {
                failed.push({ square, step: 'pick', error: pickResult.error });
                if (resultEl) resultEl.textContent += `PICK FAIL: ${pickResult.error}\n`;
                continue;
            }
        } catch (e) {
            failed.push({ square, step: 'pick', error: e.message });
            if (resultEl) resultEl.textContent += `PICK ERR: ${e.message}\n`;
            continue;
        }

        if (!pickPlaceTestRunning) break;

        // === PLACE (same square) ===
        if (statusEl) statusEl.textContent = `[${tested}/${testSquares.length}] Placing ${pieceType} on ${square}...`;
        if (resultEl) resultEl.textContent += `place...`;

        try {
            const placeResp = await fetch(`${ROBOT_URL}/manual_place`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ square: square, piece_type: pieceType }),
            });
            const placeResult = await placeResp.json();
            if (!placeResult.success) {
                failed.push({ square, step: 'place', error: placeResult.error });
                if (resultEl) resultEl.textContent += `PLACE FAIL: ${placeResult.error}\n`;
                continue;
            }
        } catch (e) {
            failed.push({ square, step: 'place', error: e.message });
            if (resultEl) resultEl.textContent += `PLACE ERR: ${e.message}\n`;
            continue;
        }

        if (resultEl) resultEl.textContent += `OK\n`;
        resultEl.scrollTop = resultEl.scrollHeight;
    }

    // Summary
    const summary = `\n===== PICK & PLACE TEST DONE =====\nTested: ${tested}, Failed: ${failed.length}\n`;
    if (resultEl) resultEl.textContent += summary;
    if (failed.length > 0) {
        if (resultEl) resultEl.textContent += `Failures:\n${failed.map(f => `  ${f.square} (${f.step}): ${f.error}`).join('\n')}\n`;
    }
    if (statusEl) statusEl.textContent = `Test done. ${failed.length} failed out of ${tested} tested.`;

    pickPlaceTestRunning = false;
    if (btnBlack) btnBlack.style.display = '';
    if (btnWhite) btnWhite.style.display = '';
    if (btnStop) btnStop.style.display = 'none';
}

function stopPickPlaceTest() {
    pickPlaceTestRunning = false;
    const statusEl = document.getElementById('manual-status');
    if (statusEl) statusEl.textContent = 'Pick & Place test stopped.';
}

// ==================== SQUARE POSITION TEACHING ====================

let squarePositions = {};       // { "e2": {x, y, z, pitch}, ... }
let teachSelectedSquare = null; // currently selected square for teaching

function renderSquarePositionBoard() {
    const board = document.getElementById('sq-position-board');
    if (!board) return;
    board.innerHTML = '';

    const files = ['a','b','c','d','e','f','g','h'];

    for (let rank = 8; rank >= 1; rank--) {
        const label = document.createElement('div');
        label.textContent = rank;
        label.style.cssText = 'display:flex; align-items:center; justify-content:center; width:20px; font-weight:bold; font-size:10px;';
        board.appendChild(label);

        for (let f = 0; f < 8; f++) {
            const square = files[f] + rank;
            const cell = document.createElement('div');
            const isLight = (f + rank) % 2 === 1;
            const hasPos = !!squarePositions[square];

            cell.dataset.square = square;
            cell.textContent = hasPos ? '\u2713' : '';
            cell.style.cssText = `
                display:flex; align-items:center; justify-content:center;
                aspect-ratio:1; cursor:pointer; font-weight:bold; font-size:13px;
                border: 1px solid #555;
                background: ${hasPos ? '#2d7d46' : (isLight ? '#b58863' : '#f0d9b5')};
                color: ${hasPos ? '#fff' : '#888'};
            `;

            cell.addEventListener('click', () => selectTeachSquare(square));
            board.appendChild(cell);
        }
    }

    // Bottom file labels
    const spacer = document.createElement('div');
    spacer.style.width = '20px';
    board.appendChild(spacer);
    for (const f of files) {
        const lbl = document.createElement('div');
        lbl.textContent = f;
        lbl.style.cssText = 'display:flex; align-items:center; justify-content:center; font-weight:bold; font-size:10px;';
        board.appendChild(lbl);
    }

    // Update count
    const countEl = document.getElementById('sq-count');
    if (countEl) countEl.textContent = `${Object.keys(squarePositions).length} / 64 recorded`;
}

function selectTeachSquare(square) {
    teachSelectedSquare = square;
    const pos = squarePositions[square];
    const statusEl = document.getElementById('sq-teach-status');
    const btnRecord = document.getElementById('btn-sq-record');
    const btnGoto = document.getElementById('btn-sq-goto');
    const btnGotoLow = document.getElementById('btn-sq-goto-low');
    const btnDelete = document.getElementById('btn-sq-delete');

    if (btnRecord) {
        btnRecord.disabled = false;
        btnRecord.textContent = `Record ${square}`;
    }
    if (btnGoto) btnGoto.disabled = false;
    if (btnGotoLow) btnGotoLow.disabled = !pos;
    if (btnDelete) btnDelete.disabled = !pos;

    const inputX = document.getElementById('sq-pos-x');
    const inputY = document.getElementById('sq-pos-y');
    const inputZ = document.getElementById('sq-pos-z');

    if (pos) {
        if (inputX) inputX.value = pos.x.toFixed(4);
        if (inputY) inputY.value = pos.y.toFixed(4);
        if (inputZ) inputZ.value = pos.z.toFixed(4);
        if (statusEl) statusEl.textContent = `${square}: recorded`;
    } else {
        if (inputX) inputX.value = '';
        if (inputY) inputY.value = '';
        if (inputZ) inputZ.value = '';
        if (statusEl) statusEl.textContent = `${square}: not recorded yet`;
    }

    // Also select on the main manual board (sync selection)
    manualSelectedSquare = square;

    // Highlight on teaching board
    document.querySelectorAll('#sq-position-board div[data-square]').forEach(cell => {
        if (cell.dataset.square === square) {
            cell.style.outline = '3px solid #007bff';
            cell.style.outlineOffset = '-3px';
        } else {
            cell.style.outline = 'none';
        }
    });
}

async function loadSquarePositions() {
    const statusEl = document.getElementById('sq-teach-status');
    try {
        const resp = await fetch(`${ROBOT_URL}/square_positions`);
        const data = await resp.json();
        if (data.success) {
            squarePositions = data.squares || {};
            renderSquarePositionBoard();
            if (statusEl) statusEl.textContent = `Loaded ${data.count} positions`;
        } else {
            if (statusEl) statusEl.textContent = `Load failed: ${data.error}`;
        }
    } catch (e) {
        if (statusEl) statusEl.textContent = `Error: ${e.message}`;
    }
}

async function recordSquarePosition() {
    if (!teachSelectedSquare) return;
    const statusEl = document.getElementById('sq-teach-status');
    if (statusEl) statusEl.textContent = `Recording ${teachSelectedSquare}...`;

    try {
        const resp = await fetch(`${ROBOT_URL}/square_positions/record`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ square: teachSelectedSquare }),
        });
        const data = await resp.json();
        if (data.success) {
            squarePositions[teachSelectedSquare] = {
                x: data.position.x, y: data.position.y,
                z: data.position.z, pitch: data.position.pitch,
            };
            renderSquarePositionBoard();
            selectTeachSquare(teachSelectedSquare);
            if (statusEl) statusEl.textContent = data.message;
        } else {
            if (statusEl) statusEl.textContent = `Failed: ${data.error}`;
        }
    } catch (e) {
        if (statusEl) statusEl.textContent = `Error: ${e.message}`;
    }
}

async function gotoSquarePosition() {
    if (!teachSelectedSquare) return;
    const inputX = document.getElementById('sq-pos-x');
    const inputY = document.getElementById('sq-pos-y');
    const inputZ = document.getElementById('sq-pos-z');
    const x = parseFloat(inputX?.value);
    const y = parseFloat(inputY?.value);
    const z = parseFloat(inputZ?.value);
    if (isNaN(x) || isNaN(y) || isNaN(z)) {
        const statusEl = document.getElementById('sq-teach-status');
        if (statusEl) statusEl.textContent = 'Enter valid X, Y, Z values';
        return;
    }

    const zOffsetInput = document.getElementById('sq-pos-z-offset');
    const Z_OFFSET = parseFloat(zOffsetInput?.value) || 0;
    const moveZ = z + Z_OFFSET;

    const pos = squarePositions[teachSelectedSquare];
    const pitch = (pos && pos.pitch) ? pos.pitch : 1.5708;

    const statusEl = document.getElementById('sq-teach-status');
    if (statusEl) statusEl.textContent = `Moving to (${x.toFixed(4)}, ${y.toFixed(4)}, ${moveZ.toFixed(4)}) [z+50mm]...`;

    try {
        const resp = await fetch(`${ROBOT_URL}/arm/move_to_xyz`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                x: x, y: y, z: moveZ,
                pitch: pitch,
                moving_time: 1.5,
                pitch_tolerance: 0.7854,
            }),
        });
        const data = await resp.json();
        if (statusEl) statusEl.textContent = data.success
            ? `Moved to (${x.toFixed(4)}, ${y.toFixed(4)}, ${z.toFixed(4)})`
            : `Failed: ${data.error}`;
    } catch (e) {
        if (statusEl) statusEl.textContent = `Error: ${e.message}`;
    }
}

async function gotoSquarePositionLow() {
    if (!teachSelectedSquare) return;
    const pos = squarePositions[teachSelectedSquare];
    if (!pos) return;

    const lowZ = pos.z - 0.010;
    const pitch = pos.pitch || 1.5708;
    const statusEl = document.getElementById('sq-teach-status');
    if (statusEl) statusEl.textContent = `Going low to ${teachSelectedSquare} (z=${lowZ.toFixed(4)}, -10mm)...`;

    try {
        const resp = await fetch(`${ROBOT_URL}/arm/move_to_xyz`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                x: pos.x, y: pos.y, z: lowZ,
                pitch: pitch,
                moving_time: 1.5,
                pitch_tolerance: 0.7854,
            }),
        });
        const data = await resp.json();
        if (statusEl) statusEl.textContent = data.success
            ? `Low at ${teachSelectedSquare} (z=${lowZ.toFixed(4)})`
            : `Failed: ${data.error}`;
    } catch (e) {
        if (statusEl) statusEl.textContent = `Error: ${e.message}`;
    }
}

let autotestRunning = false;

async function autotestAllSquares() {
    if (autotestRunning) return;
    autotestRunning = true;

    const btnStart = document.getElementById('btn-sq-autotest');
    const btnStop = document.getElementById('btn-sq-autotest-stop');
    const statusEl = document.getElementById('sq-teach-status');
    const resultEl = document.getElementById('sq-autotest-result');
    if (btnStart) btnStart.style.display = 'none';
    if (btnStop) btnStop.style.display = '';
    if (resultEl) { resultEl.style.display = 'block'; resultEl.textContent = ''; }

    const zOffsetInput = document.getElementById('sq-pos-z-offset');
    const Z_OFFSET = parseFloat(zOffsetInput?.value) || 0;

    const files = 'abcdefgh';
    const ranks = '87654321';
    const failed = [];
    const skipped = [];
    let tested = 0;

    for (const rank of ranks) {
        for (const file of files) {
            if (!autotestRunning) break;
            const square = file + rank;
            const pos = squarePositions[square];
            if (!pos) {
                skipped.push(square);
                continue;
            }

            tested++;
            const moveZ = pos.z + Z_OFFSET;
            const pitch = pos.pitch || 1.5708;
            if (statusEl) statusEl.textContent = `Testing ${square} (${tested}/64)...`;

            // Highlight on board
            selectTeachSquare(square);

            try {
                const resp = await fetch(`${ROBOT_URL}/arm/move_to_xyz`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ x: pos.x, y: pos.y, z: moveZ, pitch: pitch, moving_time: 1.5, pitch_tolerance: 0.7854 }),
                });
                const data = await resp.json();
                if (!data.success) {
                    failed.push({ square, error: data.error || 'unknown' });
                    if (resultEl) resultEl.textContent += `FAIL ${square}: ${data.error}\n`;
                } else {
                    if (resultEl) resultEl.textContent += `OK   ${square}\n`;
                }
            } catch (e) {
                failed.push({ square, error: e.message });
                if (resultEl) resultEl.textContent += `ERR  ${square}: ${e.message}\n`;
            }

            // Wait for move to complete
            await new Promise(r => setTimeout(r, 2000));
        }
        if (!autotestRunning) break;
    }

    // Summary
    const summary = `\n===== AUTO TEST DONE =====\nTested: ${tested}, Skipped: ${skipped.length}, Failed: ${failed.length}\n`;
    if (resultEl) resultEl.textContent += summary;
    if (failed.length > 0) {
        if (resultEl) resultEl.textContent += `IK failures: ${failed.map(f => f.square).join(', ')}\n`;
    }
    if (statusEl) statusEl.textContent = `Test done. ${failed.length} failed out of ${tested} tested.`;

    autotestRunning = false;
    if (btnStart) btnStart.style.display = '';
    if (btnStop) btnStop.style.display = 'none';
}

function stopAutotest() {
    autotestRunning = false;
    const statusEl = document.getElementById('sq-teach-status');
    if (statusEl) statusEl.textContent = 'Auto test stopped.';
}

async function deleteSquarePosition() {
    if (!teachSelectedSquare) return;
    const statusEl = document.getElementById('sq-teach-status');

    try {
        const resp = await fetch(`${ROBOT_URL}/square_positions/${teachSelectedSquare}`, {
            method: 'DELETE',
        });
        const data = await resp.json();
        if (data.success) {
            delete squarePositions[teachSelectedSquare];
            renderSquarePositionBoard();
            selectTeachSquare(teachSelectedSquare);
            if (statusEl) statusEl.textContent = `Deleted ${teachSelectedSquare}`;
        } else {
            if (statusEl) statusEl.textContent = `Failed: ${data.error}`;
        }
    } catch (e) {
        if (statusEl) statusEl.textContent = `Error: ${e.message}`;
    }
}

async function clearAllSquarePositions() {
    if (!confirm('Clear all recorded square positions?')) return;
    const statusEl = document.getElementById('sq-teach-status');

    try {
        const resp = await fetch(`${ROBOT_URL}/square_positions/clear`, { method: 'POST' });
        const data = await resp.json();
        if (data.success) {
            squarePositions = {};
            renderSquarePositionBoard();
            if (statusEl) statusEl.textContent = 'All positions cleared';
        } else {
            if (statusEl) statusEl.textContent = `Failed: ${data.error}`;
        }
    } catch (e) {
        if (statusEl) statusEl.textContent = `Error: ${e.message}`;
    }
}

let boardSurfaceZ = null;

// ==================== BOARD SURFACE Z ====================

async function loadBoardSurfaceZ() {
    const display = document.getElementById('board-z-value');
    try {
        const resp = await fetch(`${ROBOT_URL}/board_surface_z`);
        const data = await resp.json();
        if (data.success && data.board_surface_z !== null) {
            boardSurfaceZ = data.board_surface_z;
            if (display) display.textContent = `Z = ${boardSurfaceZ.toFixed(4)} m`;
        } else {
            boardSurfaceZ = null;
            if (display) display.textContent = 'Not set';
        }
    } catch (e) {
        if (display) display.textContent = `Error: ${e.message}`;
    }
}

async function teachBoardSurfaceZ() {
    const display = document.getElementById('board-z-value');
    if (display) display.textContent = 'Recording...';
    try {
        const resp = await fetch(`${ROBOT_URL}/board_surface_z/record`, { method: 'POST' });
        const data = await resp.json();
        if (data.success) {
            boardSurfaceZ = data.board_surface_z;
            if (display) display.textContent = `Z = ${boardSurfaceZ.toFixed(4)} m`;
            addChatMessage('System', data.message);
        } else {
            if (display) display.textContent = `Failed: ${data.error}`;
        }
    } catch (e) {
        if (display) display.textContent = `Error: ${e.message}`;
    }
}


// ==================== TEACH 4 CORNERS (INTERPOLATION) ====================

var cornerSquares = ['a1', 'a8', 'h1', 'h8'];
var cornersTaught = {};  // e.g. { a1: true, h8: true }

function updateCornerUI() {
    var allDone = true;
    for (var i = 0; i < cornerSquares.length; i++) {
        var sq = cornerSquares[i];
        var el = document.getElementById('corner-' + sq + '-status');
        if (el) {
            el.textContent = cornersTaught[sq] ? '\u2713' : '?';
            el.style.color = cornersTaught[sq] ? '#16a34a' : '#888';
        }
        if (!cornersTaught[sq]) allDone = false;
    }
    var btn = document.getElementById('btn-interpolate');
    if (btn) btn.disabled = !allDone;
}

async function teachCorner(sq) {
    var statusEl = document.getElementById('interpolate-status');
    if (statusEl) statusEl.textContent = 'Recording ' + sq + '...';
    try {
        var resp = await fetch(ROBOT_URL + '/square_positions/record', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ square: sq }),
        });
        var data = await resp.json();
        if (data.success) {
            cornersTaught[sq] = true;
            if (statusEl) statusEl.textContent = sq + ' recorded';
            showToast(sq + ' corner recorded', 'success');
        } else {
            if (statusEl) statusEl.textContent = 'Error: ' + (data.error || data.detail || 'unknown');
            showToast('Failed to record ' + sq, 'error');
        }
    } catch (e) {
        if (statusEl) statusEl.textContent = 'Error: ' + e.message;
    }
    updateCornerUI();
    // Also refresh the full square position board
    await loadSquarePositions();
    renderSquarePositionBoard();
}

async function interpolateFromCorners() {
    var statusEl = document.getElementById('interpolate-status');
    if (statusEl) statusEl.textContent = 'Interpolating...';
    try {
        var resp = await fetch(ROBOT_URL + '/square_positions/interpolate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        });
        var data = await resp.json();
        if (data.success) {
            if (statusEl) statusEl.textContent = data.count + ' squares interpolated!';
            showToast('64 squares interpolated from 4 corners', 'success');
            await loadSquarePositions();
            renderSquarePositionBoard();
        } else {
            if (statusEl) statusEl.textContent = 'Error: ' + (data.error || data.detail || 'unknown');
            showToast('Interpolation failed', 'error');
        }
    } catch (e) {
        if (statusEl) statusEl.textContent = 'Error: ' + e.message;
    }
}

function setupCornerListeners() {
    var btns = document.querySelectorAll('.corner-btn');
    for (var i = 0; i < btns.length; i++) {
        (function (btn) {
            btn.addEventListener('click', function () {
                teachCorner(btn.getAttribute('data-sq'));
            });
        })(btns[i]);
    }
    var interpBtn = document.getElementById('btn-interpolate');
    if (interpBtn) {
        interpBtn.addEventListener('click', interpolateFromCorners);
    }
}

function refreshCornerStatus() {
    // Check which corners already exist in loaded square positions
    // This is called after loadSquarePositions
    if (typeof squarePositions !== 'undefined' && squarePositions) {
        for (var i = 0; i < cornerSquares.length; i++) {
            var sq = cornerSquares[i];
            if (squarePositions[sq]) {
                cornersTaught[sq] = true;
            }
        }
    }
    updateCornerUI();
}


// ==================== TOAST NOTIFICATIONS ====================

/**
 * Show a toast notification.
 * @param {string} message - Text to display
 * @param {'info'|'success'|'warning'|'error'} type - Toast style
 * @param {number} duration - Auto-dismiss ms (0 = sticky)
 */
function showToast(message, type, duration) {
    type = type || 'info';
    duration = duration !== undefined ? duration : 3000;
    var container = document.getElementById('toast-container');
    if (!container) return;

    var toast = document.createElement('div');
    toast.className = 'toast toast-' + type;
    toast.textContent = message;

    // Close button
    var closeBtn = document.createElement('span');
    closeBtn.className = 'toast-close';
    closeBtn.textContent = '\u00d7';
    closeBtn.onclick = function () { removeToast(toast); };
    toast.appendChild(closeBtn);

    container.appendChild(toast);

    // Auto-dismiss
    if (duration > 0) {
        setTimeout(function () { removeToast(toast); }, duration);
    }
}

function removeToast(el) {
    if (!el || !el.parentNode) return;
    el.style.opacity = '0';
    setTimeout(function () {
        if (el.parentNode) el.parentNode.removeChild(el);
    }, 300);
}


// ==================== WEBSOCKET /ws/game ====================

var gameWs = null;
var gameWsReconnectTimer = null;

function connectGameWs() {
    if (gameWs && gameWs.readyState <= 1) return; // already open/connecting
    var protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    var url = protocol + '//' + location.host + '/ws/game';

    gameWs = new WebSocket(url);

    gameWs.onopen = function () {
        console.log('[WS/game] connected');
        if (gameWsReconnectTimer) {
            clearTimeout(gameWsReconnectTimer);
            gameWsReconnectTimer = null;
        }
    };

    gameWs.onmessage = function (evt) {
        try {
            var msg = JSON.parse(evt.data);
            handleGameWsMessage(msg);
        } catch (e) {
            console.warn('[WS/game] bad message', e);
        }
    };

    gameWs.onclose = function () {
        console.log('[WS/game] disconnected, retrying in 3s');
        scheduleGameWsReconnect();
    };

    gameWs.onerror = function () {
        // onclose will fire after this
    };
}

function scheduleGameWsReconnect() {
    if (gameWsReconnectTimer) return;
    gameWsReconnectTimer = setTimeout(function () {
        gameWsReconnectTimer = null;
        connectGameWs();
    }, 3000);
}

function handleGameWsMessage(msg) {
    var type = msg.type;
    var data = msg.data || {};

    if (type === 'game_state' && data.fen) {
        // Update board display
        renderMainBoard(data.fen);
        if (elements.gameStatus) elements.gameStatus.textContent = data.status || '';
        if (elements.whoseTurn) elements.whoseTurn.textContent = data.whose_turn || '';
        if (elements.moveNumber) elements.moveNumber.textContent = data.move_number || '';
    }

    if (type === 'robot_status') {
        var status = data.status || 'idle';
        if (status === 'error') {
            showToast('Robot error: ' + (data.error || 'unknown'), 'error', 5000);
        }
    }

    if (type === 'error') {
        showToast(data.message || 'Unknown error', 'error', 5000);
    }

    if (type === 'pause') {
        if (data.paused) {
            showToast('Game paused: ' + (data.reason || ''), 'warning', 0);
        }
    }

    if (type === 'robot_done') {
        _awaitingRobotMove = false;
        // Track last move for board highlighting (especially watch mode)
        if (data.last_robot_move) {
            const m = data.last_robot_move;
            simLastMove = { from: m.substring(0, 2), to: m.substring(2, 4) };
        }
        if (data.game_state) {
            updateGameDisplay(data.game_state);
        }
        if (!simulationMode) {
            updateAutoDetectStatus('Your turn! Auto-detecting...');
            // Start YOLO stream in the main camera view while scanning
            startMainYoloStream();
        }
    }

    if (type === 'detecting_move') {
        if (!simulationMode) {
            updateAutoDetectStatus('Scanning board... (poll #' + (data.poll || '?') + ')');
            // Ensure YOLO stream is running while scanning
            if (elements.visionImage && !elements.visionImage.src.includes('/stream/yolo')) {
                startMainYoloStream();
            }
        }
    }

    if (type === 'move_detected') {
        updateAutoDetectStatus('');
        if (data.game_state) {
            updateGameDisplay(data.game_state);
        }
        if (data.move) {
            addChatMessage('System', 'Move detected: ' + data.move);
        }
    }

    if (type === 'detection_failed') {
        if (!simulationMode) {
            const candidates = data.candidates || [];
            let msg = 'Detection failed';
            if (candidates.length > 1) {
                msg = 'Ambiguous move (' + candidates.join(', ') + ') — type your move below';
            } else {
                msg += ' — type your move below, or wait for retry.';
            }
            updateAutoDetectStatus(msg);
            // Ensure manual move bar is visible for fallback input
            const moveBar = document.getElementById('move-submit-bar');
            if (moveBar) moveBar.style.display = 'flex';
            if (elements.inputManualMove) {
                elements.inputManualMove.focus();
                if (candidates.length === 1) {
                    elements.inputManualMove.value = candidates[0];
                }
                elements.inputManualMove.style.borderColor = '#e74c3c';
                setTimeout(function() { elements.inputManualMove.style.borderColor = ''; }, 3000);
            }
        }
    }

    // Teach mode auto-detect events
    if (type === 'teach_step_done') {
        handleTeachMoveResult({
            correct: true,
            detected_move: data.detected_move,
            message: data.message,
            explanation: data.explanation,
        });
        if (data.game_state) updateGameDisplay(data.game_state);
    }

    if (type === 'teach_robot_moving') {
        var detectStatus = document.getElementById('teach-detect-status');
        if (detectStatus) detectStatus.textContent = data.message || 'Teacher is making the opponent\'s move...';
    }

    if (type === 'teach_demo_anim') {
        // Animated demo move on the sim chessboard
        const boardEl = elements.chessBoard;
        if (!boardEl) return;
        const phase = data.phase;

        if (phase === 'source') {
            // Highlight the source square (pulse effect)
            teachHighlightSquares = [data.from];
            teachHighlightFrom = data.from;
            teachHighlightTo = null;
            teachDemoGhost = null;
            renderSimBoard();
            // Add pulse animation to source cell
            const srcCell = boardEl.querySelector('[data-square="' + data.from + '"]');
            if (srcCell) srcCell.classList.add('teach-demo-pulse');
        } else if (phase === 'move') {
            // Animate piece sliding from source to destination
            teachHighlightSquares = data.squares || [];
            teachHighlightFrom = data.from;
            teachHighlightTo = data.to;
            _animateTeachDemoMove(data.from, data.to);
        } else if (phase === 'reset') {
            // Reset board back to original (demo was conceptual)
            teachDemoGhost = null;
            renderSimBoard();
        }
    }

    if (type === 'teach_highlight') {
        // Highlight squares on the chessboard during teach instruction
        teachHighlightSquares = data.squares || [];
        teachHighlightFrom = data.from || null;
        teachHighlightTo = data.to || null;
        teachDemoGhost = null;
        renderSimBoard();
    }

    if (type === 'teach_input_lock') {
        console.log('[Teach] input_lock event:', data, 'audioPlaying:', _gameWsAudioPlaying);
        if (data.locked) {
            teachInputLocked = true;
            renderSimBoard();
            // Safety: auto-unlock after 30s in case unlock event is missed
            if (window._teachLockTimeout) clearTimeout(window._teachLockTimeout);
            window._teachLockTimeout = setTimeout(() => {
                if (teachInputLocked) {
                    console.log('[Teach] Safety unlock after 30s timeout');
                    teachInputLocked = false;
                    _teachUnlockAfterAudio = false;
                    renderSimBoard();
                }
            }, 30000);
        } else {
            if (window._teachLockTimeout) clearTimeout(window._teachLockTimeout);
            if (data.after_audio && _gameWsAudioPlaying) {
                // Defer unlock until TTS audio queue finishes playing
                _teachUnlockAfterAudio = true;
            } else {
                teachInputLocked = false;
                _teachUnlockAfterAudio = false;
                renderSimBoard();
            }
        }
    }

    if (type === 'teach_move_wrong') {
        handleTeachMoveResult({
            correct: false,
            detected_move: data.detected_move,
            message: data.message,
        });
    }

    if (type === 'teach_detecting') {
        var detectStatus = document.getElementById('teach-detect-status');
        if (detectStatus) detectStatus.textContent = 'Scanning board... (poll #' + (data.poll || '?') + ')';
    }

    if (type === 'voice_response') {
        if (data.transcription) {
            addChatMessage('You (voice)', data.transcription);
        }
        // Text is shown via voice_text_chunk streaming — don't duplicate here
    }

    if (type === 'watch_side') {
        // Watch mode: track which side is about to speak
        _watchCurrentSender = data.side + ' (' + data.model + ')';
    }

    if (type === 'voice_text_chunk') {
        // Append streaming text chunk to current message
        appendToStreamingMessage(data.chunk || '');
    }

    if (type === 'voice_audio_chunk') {
        // Queue sentence-level TTS audio for synchronized playback
        if (data.data) {
            queueGameWsAudio(data.data, data.format || 'mp3');
        }
    }

    if (type === 'voice_audio_done') {
        // All TTS audio has been sent — no action needed;
        // the queue will drain on its own via _playNextGameWsAudio
    }

    if (type === 'voice_text_done') {
        // Finalize the streaming text message
        finalizeStreamingMessage();
        // Only fall back to REST TTS if backend didn't already stream audio
        if (!data.tts_streamed && voiceOutputEnabled && data.text) {
            playTtsFromText(data.text);
        }
    }

    if (type === 'voice_stop') {
        // Server requested TTS stop (user interrupted)
        stopCurrentTts();
        finalizeStreamingMessage();
    }
}


// =========================
// Voice WebSocket Client
// =========================

function connectVoiceWs() {
    if (voiceWs && voiceWs.readyState === WebSocket.OPEN) return;

    const wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    voiceWs = new WebSocket(`${wsProto}//${location.host}/ws/voice`);

    voiceWs.onopen = () => {
        console.log('[Voice WS] Connected');
        updateVoiceStatus('idle');
        if (elements.btnMic) elements.btnMic.disabled = false;
        // Set current mode
        voiceWs.send(JSON.stringify({ type: 'set_mode', mode: voiceListeningMode }));
        // Start continuous stream if always-on
        if (voiceListeningMode === 'always_on') {
            startContinuousStream();
        }
    };

    voiceWs.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            handleVoiceWsMessage(msg);
        } catch (e) {
            console.error('[Voice WS] Parse error:', e);
        }
    };

    voiceWs.onclose = () => {
        console.log('[Voice WS] Disconnected');
        updateVoiceStatus('idle');
        if (elements.btnMic) elements.btnMic.disabled = true;
        stopContinuousStream();
        // Reconnect if still enabled
        if (voiceEnabled) {
            setTimeout(connectVoiceWs, 3000);
        }
    };

    voiceWs.onerror = (err) => {
        console.error('[Voice WS] Error:', err);
    };
}

function disconnectVoiceWs() {
    stopContinuousStream();
    stopPttRecording();
    if (voiceWs) {
        voiceWs.close();
        voiceWs = null;
    }
    if (elements.btnMic) elements.btnMic.disabled = true;
    updateVoiceStatus('idle');
}

function handleVoiceWsMessage(msg) {
    const type = msg.type;

    if (type === 'transcription') {
        appendVoiceTranscript('You', msg.text);
    } else if (type === 'agent_response') {
        if (msg.text) {
            appendVoiceTranscript(llmModelName, msg.text);
            // Don't addChatMessage here — voice_text_chunk already shows it
        }
        if (msg.move_detected) {
            addChatMessage('System', `Move detected: ${msg.move_detected}`);
        }
    } else if (type === 'audio_chunk') {
        // Queue TTS audio for playback
        queueTtsChunk(msg.data, msg.format || 'mp3');
    } else if (type === 'audio_end') {
        flushTtsPlayback();
    } else if (type === 'voice_status') {
        updateVoiceStatus(msg.status);
    } else if (type === 'listening_mode') {
        voiceListeningMode = msg.mode;
        if (elements.voiceListeningMode) {
            elements.voiceListeningMode.value = msg.mode;
        }
        updateMicButton();
    } else if (type === 'language_changed') {
        currentLanguage = msg.language;
        updateLanguageButtons();
    } else if (type === 'voice_stop') {
        // Server requested TTS stop (user interrupted)
        stopCurrentTts();
        finalizeStreamingMessage();
    } else if (type === 'error') {
        console.error('[Voice WS] Error:', msg.error);
        addChatMessage('System', `Voice error: ${msg.error}`);
    }
}

function updateVoiceStatus(status) {
    voiceStatus = status;
    if (elements.voiceStatusIndicator) {
        const labels = {
            idle: 'Idle',
            listening: 'Listening...',
            processing: 'Processing...',
            speaking: 'Speaking...',
        };
        elements.voiceStatusIndicator.textContent = labels[status] || status;
        elements.voiceStatusIndicator.style.color =
            status === 'listening' ? '#22c55e' :
            status === 'processing' ? '#eab308' :
            status === 'speaking' ? '#3b82f6' : 'var(--text-secondary)';
    }
}

function updateMicButton() {
    if (!elements.btnMic) return;
    if (voiceListeningMode === 'always_on') {
        elements.btnMic.textContent = vadInterval ? 'Mute' : 'Unmute';
        elements.btnMic.disabled = !voiceEnabled;
    } else {
        elements.btnMic.textContent = 'Hold to Talk';
        elements.btnMic.disabled = !voiceEnabled;
    }
}

function appendVoiceTranscript(sender, text) {
    if (!elements.voiceTranscript) return;
    // Clear placeholder
    const em = elements.voiceTranscript.querySelector('em');
    if (em) em.remove();

    const line = document.createElement('div');
    line.style.marginBottom = '4px';
    line.innerHTML = `<strong style="color:${sender === 'You' ? '#22c55e' : '#3b82f6'}">${sender}:</strong> ${text}`;
    elements.voiceTranscript.appendChild(line);
    elements.voiceTranscript.scrollTop = elements.voiceTranscript.scrollHeight;
}

// --- Push-to-Talk Recording ---

async function startPttRecording() {
    if (!voiceWs || voiceWs.readyState !== WebSocket.OPEN) return;
    if (mediaRecorder && mediaRecorder.state === 'recording') return;

    try {
        if (!mediaStream) {
            mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        }

        audioChunks = [];
        mediaRecorder = new MediaRecorder(mediaStream, { mimeType: 'audio/webm;codecs=opus' });

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            if (audioChunks.length === 0) return;
            const blob = new Blob(audioChunks, { type: 'audio/webm' });
            const arrayBuffer = await blob.arrayBuffer();
            const base64 = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));

            if (voiceWs && voiceWs.readyState === WebSocket.OPEN) {
                voiceWs.send(JSON.stringify({
                    type: 'audio_chunk',
                    data: base64,
                    format: 'webm',
                    is_final: true,
                }));
            }
        };

        mediaRecorder.start(100); // Collect chunks every 100ms
        if (elements.btnMic) {
            elements.btnMic.textContent = 'Recording...';
            elements.btnMic.classList.add('danger');
        }
        updateVoiceStatus('listening');
    } catch (err) {
        console.error('[Voice] Mic access failed:', err);
        addChatMessage('System', `Mic error: ${err.message}`);
    }
}

function stopPttRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
    }
    if (elements.btnMic) {
        elements.btnMic.textContent = 'Hold to Talk';
        elements.btnMic.classList.remove('danger');
    }
}

// --- Chat Mic (Hold-to-Talk from main panel) ---

let chatMicRecording = false;

async function startChatMicRecording() {
    if (chatMicRecording) return;

    // Auto-connect voice WS if not connected
    if (!voiceWs || voiceWs.readyState !== WebSocket.OPEN) {
        voiceEnabled = true;
        if (elements.voiceEnabled) elements.voiceEnabled.checked = true;
        connectVoiceWs();
        // Wait for connection before recording
        await new Promise((resolve) => {
            const checkInterval = setInterval(() => {
                if (voiceWs && voiceWs.readyState === WebSocket.OPEN) {
                    clearInterval(checkInterval);
                    resolve();
                }
            }, 100);
            // Timeout after 3 seconds
            setTimeout(() => { clearInterval(checkInterval); resolve(); }, 3000);
        });
        if (!voiceWs || voiceWs.readyState !== WebSocket.OPEN) {
            addChatMessage('System', 'Voice connection failed. Try again.');
            return;
        }
    }

    chatMicRecording = true;
    elements.btnChatMic.classList.add('recording');
    elements.btnChatMic.title = 'Recording... release to send';

    // Reuse the PTT recording infrastructure
    await startPttRecording();
}

function stopChatMicRecording() {
    if (!chatMicRecording) return;
    chatMicRecording = false;
    elements.btnChatMic.classList.remove('recording');
    elements.btnChatMic.title = 'Hold to talk';
    stopPttRecording();
}

// --- Always-On Continuous Stream ---

let continuousStreamInterval = null;
let continuousRecorder = null;

// ── VAD-based always-on listening ──────────────────────────────────────
async function startContinuousStream() {
    if (vadInterval) return;
    if (!voiceWs || voiceWs.readyState !== WebSocket.OPEN) return;

    try {
        if (!mediaStream) {
            mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        }

        // Set up Web Audio analyser for volume detection
        vadAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
        vadSource = vadAudioCtx.createMediaStreamSource(mediaStream);
        vadAnalyser = vadAudioCtx.createAnalyser();
        vadAnalyser.fftSize = 512;
        vadAnalyser.smoothingTimeConstant = 0.3;
        vadSource.connect(vadAnalyser);

        vadSpeechDetected = false;
        vadSilenceStart = 0;
        vadSpeechStart = 0;

        updateVoiceStatus('listening');
        _updateVadMeter(0);

        // Poll volume every 50ms
        const freqData = new Uint8Array(vadAnalyser.frequencyBinCount);
        vadInterval = setInterval(() => {
            if (!vadAnalyser) return;
            vadAnalyser.getByteFrequencyData(freqData);

            // Compute average volume (0-255)
            let sum = 0;
            for (let i = 0; i < freqData.length; i++) sum += freqData[i];
            const avgVolume = sum / freqData.length;

            _updateVadMeter(avgVolume);

            // Echo suppression: ignore mic when TTS is playing
            if (isTtsPlaying || _gameWsAudioPlaying) {
                if (vadSpeechDetected) {
                    // Discard in-progress recording
                    _vadDiscardRecording();
                }
                return;
            }

            const now = Date.now();

            if (avgVolume > VAD_THRESHOLD) {
                // Speech detected
                if (!vadSpeechDetected) {
                    // Speech onset — start recording
                    vadSpeechDetected = true;
                    vadSpeechStart = now;
                    vadSilenceStart = 0;
                    vadPeakVolume = avgVolume;
                    _vadStartRecording();
                    updateVoiceStatus('listening');
                } else {
                    // Continued speech — reset silence timer, track peak
                    vadSilenceStart = 0;
                    if (avgVolume > vadPeakVolume) vadPeakVolume = avgVolume;
                }
            } else {
                // Silence
                if (vadSpeechDetected) {
                    if (!vadSilenceStart) {
                        vadSilenceStart = now;
                    } else if (now - vadSilenceStart >= VAD_SILENCE_MS) {
                        // Silence long enough — end utterance
                        const speechDuration = now - vadSpeechStart;
                        // Require minimum duration AND peak volume (filter background noise)
                        if (speechDuration >= VAD_MIN_SPEECH_MS && vadPeakVolume >= VAD_THRESHOLD * 1.5) {
                            _vadStopAndSend();
                        } else {
                            console.log(`[VAD] Discarding: ${speechDuration}ms, peak=${vadPeakVolume.toFixed(0)}`);
                            _vadDiscardRecording();
                        }
                    }
                }
            }
        }, 50);

    } catch (err) {
        console.error('[Voice] VAD start failed:', err);
        addChatMessage('System', `VAD mic error: ${err.message}`);
    }
}

function stopContinuousStream() {
    if (vadInterval) {
        clearInterval(vadInterval);
        vadInterval = null;
    }
    _vadDiscardRecording();
    if (vadSource) { try { vadSource.disconnect(); } catch(e) {} vadSource = null; }
    if (vadAudioCtx) { try { vadAudioCtx.close(); } catch(e) {} vadAudioCtx = null; }
    vadAnalyser = null;
    _updateVadMeter(0);
    updateVoiceStatus('idle');
}

function _vadStartRecording() {
    if (vadRecorder) return;
    vadChunks = [];
    try {
        vadRecorder = new MediaRecorder(mediaStream, { mimeType: 'audio/webm;codecs=opus' });
        vadRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) vadChunks.push(e.data);
        };
        vadRecorder.start();
    } catch (err) {
        console.error('[VAD] MediaRecorder start error:', err);
        vadRecorder = null;
    }
}

async function _vadStopAndSend() {
    vadSpeechDetected = false;
    vadSilenceStart = 0;
    if (!vadRecorder || vadRecorder.state !== 'recording') return;

    updateVoiceStatus('processing');

    // Stop recorder — audio comes via onstop
    const recorder = vadRecorder;
    vadRecorder = null;

    await new Promise((resolve) => {
        recorder.onstop = async () => {
            if (vadChunks.length === 0) { resolve(); return; }
            const blob = new Blob(vadChunks, { type: 'audio/webm' });
            vadChunks = [];
            const arrayBuffer = await blob.arrayBuffer();
            const base64 = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));

            if (voiceWs && voiceWs.readyState === WebSocket.OPEN) {
                voiceWs.send(JSON.stringify({
                    type: 'audio_chunk',
                    data: base64,
                    format: 'webm',
                    is_final: true,
                }));
            }
            resolve();
        };
        recorder.stop();
    });

    // Back to listening after sending
    updateVoiceStatus('listening');
}

function _vadDiscardRecording() {
    vadSpeechDetected = false;
    vadSilenceStart = 0;
    if (vadRecorder && vadRecorder.state === 'recording') {
        vadRecorder.onstop = null;
        vadRecorder.stop();
    }
    vadRecorder = null;
    vadChunks = [];
}

function _updateVadMeter(volume) {
    const bar = document.getElementById('vad-volume-bar');
    const meter = document.getElementById('vad-volume-meter');
    if (!bar || !meter) return;
    if (vadInterval) {
        meter.style.display = '';
        const pct = Math.min(100, (volume / 128) * 100);
        bar.style.width = pct + '%';
        bar.style.background = volume > VAD_THRESHOLD ? '#22c55e' : '#64748b';
    } else {
        meter.style.display = 'none';
    }
}

function _toggleVadSettings() {
    const vadSettings = document.getElementById('vad-settings');
    if (vadSettings) {
        vadSettings.style.display = voiceListeningMode === 'always_on' ? '' : 'none';
    }
}

// ── API Settings (Agent tab) ────────────────────────────────────────
async function loadAgentSettings() {
    const result = await fetchJSON('/agent/settings');
    if (!result.llm_model) return;
    const fields = {
        'setting-llm-api-key': result.llm_api_key ? '********' : '',
        'setting-llm-model': result.llm_model || '',
        'setting-llm-base-url': result.llm_base_url || '',
        'setting-llm-model-2': result.llm_model_2 || '',
        'setting-llm-api-key-2': result.llm_api_key_2 ? '********' : '',
        'setting-llm-base-url-2': result.llm_base_url_2 || '',
        'setting-llm-model-3': result.llm_model_3 || '',
        'setting-llm-api-key-3': result.llm_api_key_3 ? '********' : '',
        'setting-llm-base-url-3': result.llm_base_url_3 || '',
        'setting-tts-provider': result.tts_provider || 'openai',
        'setting-tts-voice': result.tts_voice || '',
        'setting-tts-url': result.tts_service_url || '',
    };
    for (const [id, val] of Object.entries(fields)) {
        const el = document.getElementById(id);
        if (el) el.value = val;
    }
    const moveSourceEl = document.getElementById('move-source-select');
    const savedMoveSource = localStorage.getItem('move_source');
    if (savedMoveSource && moveSourceEl) moveSourceEl.value = savedMoveSource;
    // Sync game-tab engine selector
    const gameEngineEl = document.getElementById('game-engine-select');
    if (savedMoveSource && gameEngineEl) gameEngineEl.value = savedMoveSource;
}

async function saveAgentSettings() {
    const statusEl = document.getElementById('settings-status');
    const moveSourceEl = document.getElementById('move-source-select');

    const settings = {};
    const fieldMap = {
        'setting-llm-model': 'llm_model',
        'setting-llm-base-url': 'llm_base_url',
        'setting-llm-model-2': 'llm_model_2',
        'setting-llm-base-url-2': 'llm_base_url_2',
        'setting-llm-model-3': 'llm_model_3',
        'setting-llm-base-url-3': 'llm_base_url_3',
        'setting-tts-provider': 'tts_provider',
        'setting-tts-voice': 'tts_voice',
        'setting-tts-url': 'tts_service_url',
    };
    for (const [id, key] of Object.entries(fieldMap)) {
        const el = document.getElementById(id);
        if (el && el.value.trim()) settings[key] = el.value.trim();
    }
    // API Keys: only send if changed (not the masked ********)
    const key1El = document.getElementById('setting-llm-api-key');
    if (key1El && key1El.value.trim() && !key1El.value.includes('*')) {
        settings.llm_api_key = key1El.value.trim();
    }
    const key2El = document.getElementById('setting-llm-api-key-2');
    if (key2El && key2El.value.trim() && !key2El.value.includes('*')) {
        settings.llm_api_key_2 = key2El.value.trim();
    }
    const key3El = document.getElementById('setting-llm-api-key-3');
    if (key3El && key3El.value.trim() && !key3El.value.includes('*')) {
        settings.llm_api_key_3 = key3El.value.trim();
    }

    if (moveSourceEl) localStorage.setItem('move_source', moveSourceEl.value);

    const result = await fetchJSON('/agent/settings', {
        method: 'POST',
        body: JSON.stringify(settings),
    });
    if (statusEl) {
        statusEl.textContent = result.message || 'Saved';
        setTimeout(() => { statusEl.textContent = ''; }, 5000);
    }
}

function _loadVadSettings() {
    try {
        const saved = localStorage.getItem('vad_settings');
        if (saved) {
            const s = JSON.parse(saved);
            VAD_THRESHOLD = s.threshold || 80;
            VAD_SILENCE_MS = s.silence || 1500;
            VAD_MIN_SPEECH_MS = s.minSpeech || 2000;
        }
    } catch (e) {}
    // Sync UI
    const thEl = document.getElementById('vad-threshold-input');
    const silEl = document.getElementById('vad-silence-input');
    const minEl = document.getElementById('vad-min-speech-input');
    if (thEl) thEl.value = VAD_THRESHOLD;
    if (silEl) silEl.value = VAD_SILENCE_MS;
    if (minEl) minEl.value = VAD_MIN_SPEECH_MS;
    _toggleVadSettings();
}

function _saveVadSettings() {
    const thEl = document.getElementById('vad-threshold-input');
    const silEl = document.getElementById('vad-silence-input');
    const minEl = document.getElementById('vad-min-speech-input');
    VAD_THRESHOLD = parseInt(thEl?.value) || 80;
    VAD_SILENCE_MS = parseInt(silEl?.value) || 1500;
    VAD_MIN_SPEECH_MS = parseInt(minEl?.value) || 2000;
    localStorage.setItem('vad_settings', JSON.stringify({
        threshold: VAD_THRESHOLD,
        silence: VAD_SILENCE_MS,
        minSpeech: VAD_MIN_SPEECH_MS,
    }));
    console.log(`[VAD] Settings saved: threshold=${VAD_THRESHOLD}, silence=${VAD_SILENCE_MS}ms, minSpeech=${VAD_MIN_SPEECH_MS}ms`);
}

// --- TTS Audio Playback ---

function queueTtsChunk(base64Data, format) {
    ttsAudioQueue.push(base64Data);
}

function flushTtsPlayback() {
    if (ttsAudioQueue.length === 0) return;
    if (!voiceOutputEnabled) { ttsAudioQueue = []; return; }

    // Notify server that TTS is playing (echo suppression)
    if (voiceWs && voiceWs.readyState === WebSocket.OPEN) {
        voiceWs.send(JSON.stringify({ type: 'echo_state', speaking: true }));
    }

    isTtsPlaying = true;
    const combined = ttsAudioQueue.join('');
    ttsAudioQueue = [];

    try {
        const binary = atob(combined);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
        }
        const blob = new Blob([bytes], { type: 'audio/mpeg' });
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        currentTtsAudio = audio;  // Store reference for interruption
        audio.onended = () => {
            URL.revokeObjectURL(url);
            isTtsPlaying = false;
            currentTtsAudio = null;
            if (voiceWs && voiceWs.readyState === WebSocket.OPEN) {
                voiceWs.send(JSON.stringify({ type: 'echo_state', speaking: false }));
            }
        };
        audio.onerror = () => {
            URL.revokeObjectURL(url);
            isTtsPlaying = false;
            currentTtsAudio = null;
            if (voiceWs && voiceWs.readyState === WebSocket.OPEN) {
                voiceWs.send(JSON.stringify({ type: 'echo_state', speaking: false }));
            }
        };
        audio.play().catch(err => {
            console.error('[Voice] TTS playback error:', err);
            isTtsPlaying = false;
            currentTtsAudio = null;
        });
    } catch (err) {
        console.error('[Voice] TTS decode error:', err);
        isTtsPlaying = false;
    }
}

// --- Play TTS via REST endpoint (for game events without voice WS) ---

async function playTtsFromText(text) {
    if (!text || isTtsPlaying) return;
    isTtsPlaying = true;
    try {
        const response = await fetch('/voice/speak', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text, format: 'mp3' }),
        });
        if (!response.ok) {
            console.error('[TTS] speak failed:', response.status);
            isTtsPlaying = false;
            return;
        }
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        currentTtsAudio = audio;  // Store reference for interruption
        audio.onended = () => { URL.revokeObjectURL(url); isTtsPlaying = false; currentTtsAudio = null; };
        audio.onerror = () => { URL.revokeObjectURL(url); isTtsPlaying = false; currentTtsAudio = null; };
        audio.play().catch(err => {
            console.error('[TTS] playback error:', err);
            isTtsPlaying = false;
            currentTtsAudio = null;
        });
    } catch (err) {
        console.error('[TTS] fetch error:', err);
        isTtsPlaying = false;
    }
}


// Initialize
async function init() {
    setupEventListeners();
    await checkHealth();

    // Load current language setting
    try {
        const langResp = await fetchJSON('/agent/language');
        if (langResp.language) {
            currentLanguage = langResp.language;
            updateLanguageButtons();
        }
    } catch (e) { /* use default */ }

    // Restore saved language preference and sync to server before loading lessons
    const savedLang = localStorage.getItem('language');
    if (savedLang && savedLang !== currentLanguage) {
        currentLanguage = savedLang;
        try {
            await fetchJSON('/agent/language', {
                method: 'POST',
                body: JSON.stringify({ language: savedLang }),
            });
        } catch (e) { /* ignore */ }
    }
    updateLanguageButtons();

    // Initial state
    renderMainBoard('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');

    // Load available lessons, saved characters, and VAD settings
    await loadLessons();
    await loadCharacters();
    _loadVadSettings();
    loadAgentSettings();
    const savedVoiceMode = localStorage.getItem('voice_listening_mode');
    if (savedVoiceMode && elements.voiceListeningMode) {
        voiceListeningMode = savedVoiceMode;
        elements.voiceListeningMode.value = savedVoiceMode;
        _toggleVadSettings();
    }
    const savedChar = localStorage.getItem('battle_character');
    if (savedChar) {
        try {
            const ch = JSON.parse(savedChar);
            // Only restore if same language, otherwise auto-select first for current language
            if (ch.language && ch.language !== currentLanguage) {
                const charSelect = document.getElementById('character-select');
                if (charSelect && charSelect.options.length > 1) {
                    charSelect.selectedIndex = 1;
                }
                onCharacterSelected();
            } else {
                const titleEl = document.getElementById('character-input');
                const descEl = document.getElementById('character-desc');
                if (titleEl && ch.title) titleEl.value = ch.title;
                if (descEl && ch.description) descEl.value = ch.description;
                _updateCharCount();
            }
        } catch (e) {}
    }

    addChatMessage('System', 'Control panel ready. Start a game to begin!');

    // Check robot connection status
    await checkRobotConnection();

    // Load waypoints
    await refreshWaypoints();

    // Check calibration status
    await refreshCalibStatus();

    // Load game state (if active) or render empty board
    try {
        const gameResp = await fetch(`${ROBOT_URL}/game/state`);
        const gameData = await gameResp.json();
        if (gameData.active) {
            renderManualBoard(gameData.piece_map);
            updateGameTrackerInfo(gameData);
            updateMoveHistory(gameData.move_history || []);
        } else {
            renderManualBoard({});
        }
    } catch (e) {
        renderManualBoard({});
    }

    // Load square positions for teaching board
    await loadSquarePositions();
    renderSquarePositionBoard();
    // refreshCornerStatus() removed — Teach 4 Corners UI deleted

    // Level2 waypoints removed

    // Load gestures
    await loadGestures();

    // Load board surface Z
    await loadBoardSurfaceZ();

    // Connect game WebSocket
    connectGameWs();

    // Periodic health check
    setInterval(checkHealth, 10000);
}

// Start
document.addEventListener('DOMContentLoaded', init);
