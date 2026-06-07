(() => {
  "use strict";

  const canvas = document.getElementById("game");
  const ctx = canvas.getContext("2d", { alpha: false });
  const scoreEl = document.getElementById("score");
  const livesEl = document.getElementById("lives");
  const levelEl = document.getElementById("level");
  const restartBtn = document.getElementById("restart");
  const aiPressureEl = document.getElementById("ai-pressure");
  const aiHorizonEl = document.getElementById("ai-horizon");
  const aiMemoryEl = document.getElementById("ai-memory");
  const aiNnSamplesEl = document.getElementById("ai-nn-samples");
  const aiNnLossEl = document.getElementById("ai-nn-loss");
  const pacNnSamplesEl = document.getElementById("pac-nn-samples");
  const pacNnOnlineEl = document.getElementById("pac-nn-online");
  const pacNnRewardEl = document.getElementById("pac-nn-reward");
  const pacNnPretrainEl = document.getElementById("pac-nn-pretrain");
  const classicAiBtn = document.getElementById("classic-ai");
  const allAiBtn = document.getElementById("all-ai");
  const pretrainPacmanBtn = document.getElementById("pretrain-pacman");
  const resetAiMemoryBtn = document.getElementById("reset-ai-memory");
  const aiToggleBtns = [...document.querySelectorAll("[data-ai-toggle]")];

  const TILE = 24;
  const ROWS = 31;
  const COLS = 28;
  const STEP_MS = 115;
  const READY_SECONDS = 5;
  const READY_TICKS = Math.ceil((READY_SECONDS * 1000) / STEP_MS);
  const GHOST_TACTICS = ["chaser", "ambusher", "guardian", "trapper"];
  const PLAYER_MEMORY_LIMIT = 140;
  const PATTERN_PREDICTION_STEPS = 8;
  const LEARNING_DECAY = 0.985;
  const MIN_LEARNED_WEIGHT = 0.05;
  const BEHAVIOR_CLUSTER_LIMIT = 10;
  const BEHAVIOR_CLUSTER_RATE = 0.18;
  const BEHAVIOR_CLUSTER_DECAY = 0.992;
  const BEHAVIOR_CLUSTER_THRESHOLD = 1.1;
  const NN_INPUTS = 22;
  const NN_HIDDEN = 14;
  const NN_OUTPUTS = 4;
  const NN_LEARNING_RATE = 0.045;
  const NN_LOG_INTERVAL = 25;
  const NN_METRIC_DECAY = 0.94;
  const PAC_NN_INPUTS = 36;
  const PAC_NN_HIDDEN = 18;
  const PAC_NN_OUTPUTS = 4;
  const PAC_NN_LEARNING_RATE = 0.035;
  const PAC_NN_GAMMA = 0.9;
  const PAC_NN_LOG_INTERVAL = 25;
  const PAC_NN_METRIC_DECAY = 0.94;
  const PAC_NN_PRETRAIN_TARGET = 10000000;
  const PAC_NN_PRETRAIN_BATCH = 50000;
  const PAC_NN_PRETRAIN_CHUNK = 2000;
  const PAC_NN_PRETRAIN_MIN_CHUNK = 300;
  const PAC_NN_PRETRAIN_BUDGET_MS = 8;
  const PAC_NN_PRETRAIN_RATE = 0.026;
  const ROUTE_DISTANCE_CACHE_LIMIT = 75000;
  const LEARNING_STORAGE_KEY = "mini-maze-muncher.ghost-learning.v1";
  const AI_SETTINGS_STORAGE_KEY = "mini-maze-muncher.ai-settings.v1";
  const LEARNING_SAVE_INTERVAL = 10;
  const BASE_FRIGHTENED_TICKS = 70;
  const MIN_FRIGHTENED_TICKS = 18;
  const TUNNEL_ROW = 14;

  const MAZE_TEMPLATE = [
    "############################",
    "#............##............#",
    "#.####.#####.##.#####.####.#",
    "#o####.#####.##.#####.####o#",
    "#.####.#####.##.#####.####.#",
    "#..........................#",
    "#.####.##.########.##.####.#",
    "#.####.##.########.##.####.#",
    "#......##....##....##......#",
    "######.##### ## #####.######",
    "     #.##### ## #####.#     ",
    "     #.##          ##.#     ",
    "     #.## ##GGGG## ##.#     ",
    "######.## #      # ##.######",
    "      .   #      #   .      ",
    "######.## #      # ##.######",
    "     #.## ######## ##.#     ",
    "     #.##          ##.#     ",
    "     #.## ######## ##.#     ",
    "######.## ######## ##.######",
    "#............##............#",
    "#.####.#####.##.#####.####.#",
    "#.####.#####.##.#####.####.#",
    "#o..##.......P .......##..o#",
    "###.##.##.########.##.##.###",
    "###.##.##.########.##.##.###",
    "#......##....##....##......#",
    "#.##########.##.##########.#",
    "#..........................#",
    "############################",
    "############################"
  ];

  const DIRS = {
    left: { dc: -1, dr: 0 },
    right: { dc: 1, dr: 0 },
    up: { dc: 0, dr: -1 },
    down: { dc: 0, dr: 1 }
  };
  const DIR_NAMES = Object.keys(DIRS);

  const REVERSE = { left: "right", right: "left", up: "down", down: "up" };

  const KEY_TO_DIR = {
    ArrowLeft: "left", KeyA: "left",
    ArrowRight: "right", KeyD: "right",
    ArrowUp: "up", KeyW: "up",
    ArrowDown: "down", KeyS: "down"
  };

  const DEFAULT_AI_SETTINGS = {
    patternMemory: true,
    behaviorClusters: true,
    neuralNetwork: true,
    nnLogs: true,
    autoPacman: false,
    pacmanNn: true,
    pacmanPretrain: true,
    pacmanLogs: true,
    persistentMemory: true,
    adaptiveDifficulty: true,
    groupTactics: true,
    powerPelletNerf: true,
    antiLoopTraps: true,
    tunnelTraps: true,
    speedBursts: true,
    levelScaling: true
  };

  let grid, score, lives, level, pelletsLeft, player, ghosts, frightenedTicks, state, readyTicks, survivalTicks;
  let playerTrail, learnedTurns, behaviorClusters, neuralNetwork, pacmanNetwork;
  let aiSettings = { ...DEFAULT_AI_SETTINGS };
  let neuralLogWindow = { loss: 0, correct: 0, confidence: 0, count: 0 };
  let pacmanLogWindow = { reward: 0, loss: 0, count: 0 };
  let pacmanPretrainJob = null;
  let pacmanPretrainRunId = 0;
  let routeDistanceCache = new Map();
  let learnedMovesSinceSave = 0;
  let loopId = null;
  let tickCounter = 0;

  /**
   * Checks whether an AI feature is enabled.
   *
   * @param {string} name Feature key.
   * @returns {boolean} True when the feature is active.
   */
  function aiEnabled(name) {
    return aiSettings[name] !== false;
  }

  /**
   * Loads saved AI toggle settings from browser storage.
   */
  function loadAiSettings() {
    try {
      const stored = JSON.parse(localStorage.getItem(AI_SETTINGS_STORAGE_KEY) || "{}");
      aiSettings = { ...DEFAULT_AI_SETTINGS, ...stored };
    } catch {
      aiSettings = { ...DEFAULT_AI_SETTINGS };
    }
  }

  /**
   * Saves AI toggle settings into browser storage.
   */
  function saveAiSettings() {
    try {
      localStorage.setItem(AI_SETTINGS_STORAGE_KEY, JSON.stringify(aiSettings));
    } catch {
      // Settings persistence is optional.
    }
  }

  /**
   * Applies one AI setting and updates the controls.
   *
   * @param {string} name Feature key.
   * @param {boolean} value New enabled state.
   */
  function setAiSetting(name, value) {
    aiSettings[name] = value;
    saveAiSettings();
    updateAiControls();
    updatePacmanPretrainingForSettings();
    draw();
  }

  /**
   * Applies a full AI preset.
   *
   * @param {boolean} value Enabled state for every AI feature.
   */
  function setAllAiSettings(value) {
    aiSettings = Object.fromEntries(Object.keys(DEFAULT_AI_SETTINGS).map(name => [name, value]));
    saveAiSettings();
    updateAiControls();
    updatePacmanPretrainingForSettings();
    draw();
  }

  /**
   * Starts or stops Pac-Man warm-start training after AI settings change.
   */
  function updatePacmanPretrainingForSettings() {
    if (!aiEnabled("pacmanNn") || !aiEnabled("pacmanPretrain")) {
      stopPacmanPretraining();
      return;
    }

    maybeWarmStartPacmanPolicy();
  }

  /**
   * Updates toggle pressed states to match the current AI settings.
   */
  function updateAiControls() {
    aiToggleBtns.forEach(button => {
      const name = button.dataset.aiToggle;
      button.setAttribute("aria-pressed", String(aiEnabled(name)));
    });
    updateAiStats();
  }

  /**
   * Formats a sample count for compact status readouts.
   *
   * @param {number} value Count to format.
   * @returns {string} Human-readable count.
   */
  function formatCount(value) {
    return Math.max(0, Math.floor(Number(value) || 0)).toLocaleString("en-US");
  }

  /**
   * Updates the manual Pac-Man pretraining button state.
   */
  function updatePretrainButton() {
    if (!pretrainPacmanBtn) return;

    const samples = pacmanNetwork?.pretrainSamples || 0;
    const progress = Math.min(100, Math.floor((samples / PAC_NN_PRETRAIN_TARGET) * 100));
    const running = isPacmanPretraining();
    pretrainPacmanBtn.disabled = !aiEnabled("pacmanNn") || !aiEnabled("pacmanPretrain");
    pretrainPacmanBtn.setAttribute("aria-pressed", String(running));

    if (!aiEnabled("pacmanNn")) {
      pretrainPacmanBtn.textContent = "Pac-Man NN Off";
    } else if (!aiEnabled("pacmanPretrain")) {
      pretrainPacmanBtn.textContent = "Warm Start Off";
    } else if (running) {
      pretrainPacmanBtn.textContent = `Pause ${progress}%`;
    } else if (samples >= PAC_NN_PRETRAIN_TARGET) {
      pretrainPacmanBtn.textContent = "Add 50k Samples";
    } else {
      pretrainPacmanBtn.textContent = "Pretrain Pac-Man";
    }
  }

  /**
   * Updates the visible AI difficulty and learning metrics.
   */
  function updateAiStats() {
    if (aiPressureEl) aiPressureEl.textContent = playerTrail ? difficultyPressure() : 0;
    if (aiHorizonEl) aiHorizonEl.textContent = playerTrail ? predictionHorizon() : PATTERN_PREDICTION_STEPS;
    if (aiMemoryEl) aiMemoryEl.textContent = `${learnedTurns?.size || 0}/${behaviorClusters?.length || 0}`;
    if (aiNnSamplesEl) aiNnSamplesEl.textContent = formatCount(neuralNetwork?.samples || 0);
    if (aiNnLossEl) aiNnLossEl.textContent = neuralNetwork?.rollingLoss != null ? neuralNetwork.rollingLoss.toFixed(3) : "-";
    if (pacNnSamplesEl) pacNnSamplesEl.textContent = formatCount(pacmanNetwork?.samples || 0);
    if (pacNnOnlineEl) pacNnOnlineEl.textContent = formatCount(pacmanNetwork?.onlineSamples || 0);
    if (pacNnRewardEl) pacNnRewardEl.textContent = pacmanNetwork?.rollingReward != null ? pacmanNetwork.rollingReward.toFixed(3) : "-";
    if (pacNnPretrainEl) {
      const samples = pacmanNetwork?.pretrainSamples || 0;
      const target = Math.max(PAC_NN_PRETRAIN_TARGET, pacmanPretrainJob?.target || 0);
      pacNnPretrainEl.textContent = `${formatCount(samples)}/${formatCount(target)}`;
    }
    updatePretrainButton();
  }

  /**
   * Clears persistent and in-memory ghost learning.
   */
  function clearAiMemory() {
    stopPacmanPretraining();
    try {
      localStorage.removeItem(LEARNING_STORAGE_KEY);
    } catch {
      // Memory clearing still works in RAM when storage is unavailable.
    }
    resetLearning();
    updateAiStats();
  }

  /**
   * Builds the playable grid from the maze template and places actors.
   */
  function parseMaze() {
    grid = [];
    pelletsLeft = 0;
    const ghostStarts = [];
    let playerStart = { c: 13, r: 23 };

    for (let r = 0; r < ROWS; r++) {
      grid[r] = [];
      for (let c = 0; c < COLS; c++) {
        let ch = MAZE_TEMPLATE[r][c] || " ";
        if (ch === "P") {
          playerStart = { c, r };
          ch = " ";
        }
        if (ch === "G") {
          ghostStarts.push({ c, r });
          ch = " ";
        }
        if (ch === "." || ch === "o") pelletsLeft++;
        grid[r][c] = ch;
      }
    }

    player = { c: playerStart.c, r: playerStart.r, dir: "left", nextDir: "left", mouthOpen: true };
    const colors = ["#ff4b6e", "#4bd6ff", "#ff9f40", "#c879ff"];
    ghosts = ghostStarts.map((g, i) => ({
      c: g.c,
      r: g.r,
      startC: g.c,
      startR: g.r,
      dir: ["left", "right", "up", "down"][i % 4],
      color: colors[i % colors.length],
      id: i,
      eaten: false
    }));
    routeDistanceCache = new Map();
  }

  /**
   * Resets score, lives, level state, actors, and starts a fresh game loop.
   */
  function resetGame() {
    stopPacmanPretraining();
    score = 0;
    lives = 3;
    level = 1;
    frightenedTicks = 0;
    readyTicks = READY_TICKS;
    state = "playing";
    tickCounter = 0;
    survivalTicks = 0;
    if (aiEnabled("persistentMemory")) {
      if (learnedTurns && behaviorClusters) saveLearning();
      loadLearning();
    } else {
      resetLearning();
    }
    parseMaze();
    maybeWarmStartPacmanPolicy();
    updateHud();
    draw();
    startLoop();
  }

  /**
   * Writes the current score, lives, and level values into the HUD.
   */
  function updateHud() {
    scoreEl.textContent = score;
    livesEl.textContent = lives;
    levelEl.textContent = level;
    updateAiStats();
  }

  /**
   * Checks whether a tile coordinate is blocked by a wall.
   *
   * @param {number} c Column index to test.
   * @param {number} r Row index to test.
   * @returns {boolean} True when the coordinate cannot be entered.
   */
  function isWall(c, r) {
    if (r < 0 || r >= ROWS) return true;
    if (c < 0 || c >= COLS) return false;
    return grid[r][c] === "#";
  }

  /**
   * Wraps a column around the horizontal tunnel edges.
   *
   * @param {number} c Column index before wrapping.
   * @returns {number} Column index inside the maze bounds.
   */
  function wrappedC(c) {
    if (c < 0) return COLS - 1;
    if (c >= COLS) return 0;
    return c;
  }

  /**
   * Determines whether an entity can move one tile in a direction.
   *
   * @param {{c: number, r: number}} entity Entity with grid coordinates.
   * @param {string} dirName Direction key from DIRS.
   * @returns {boolean} True when the target tile is open.
   */
  function canMove(entity, dirName) {
    const d = DIRS[dirName];
    if (!d) return false;
    const nr = entity.r + d.dr;
    const nc = entity.c + d.dc;
    return !isWall(nc, nr);
  }

  /**
   * Moves an entity by one tile if the requested direction is open.
   *
   * @param {{c: number, r: number, dir: string}} entity Entity to move.
   * @param {string} dirName Direction key from DIRS.
   * @returns {boolean} True when the entity moved.
   */
  function moveOneTile(entity, dirName) {
    if (!canMove(entity, dirName)) return false;
    const d = DIRS[dirName];
    entity.c = wrappedC(entity.c + d.dc);
    entity.r += d.dr;
    entity.dir = dirName;
    return true;
  }

  /**
   * Lists every direction an entity can currently move.
   *
   * @param {{c: number, r: number}} entity Entity with grid coordinates.
   * @returns {string[]} Open direction keys.
   */
  function validDirs(entity) {
    return DIR_NAMES.filter(d => canMove(entity, d));
  }

  /**
   * Clears the ghosts' learned player movement data.
   */
  function resetLearning() {
    playerTrail = [];
    learnedTurns = new Map();
    behaviorClusters = [];
    neuralNetwork = createNeuralNetwork();
    pacmanNetwork = createPacmanNetwork();
    neuralLogWindow = { loss: 0, correct: 0, confidence: 0, count: 0 };
    pacmanLogWindow = { reward: 0, loss: 0, count: 0 };
    learnedMovesSinceSave = 0;
  }

  /**
   * Loads persistent ghost learning from browser storage.
   */
  function loadLearning() {
    resetLearning();

    try {
      const rawLearning = localStorage.getItem(LEARNING_STORAGE_KEY);
      if (!rawLearning) return;

      const stored = JSON.parse(rawLearning);
      learnedTurns = new Map((stored.learnedTurns || []).map(([key, counts]) => ([
        key,
        {
          left: Number(counts.left) || 0,
          right: Number(counts.right) || 0,
          up: Number(counts.up) || 0,
          down: Number(counts.down) || 0
        }
      ])));
      behaviorClusters = (stored.behaviorClusters || [])
        .filter(cluster => Array.isArray(cluster.center) && cluster.center.length)
        .slice(0, BEHAVIOR_CLUSTER_LIMIT)
        .map(cluster => ({
          center: cluster.center.map(Number),
          outcomes: {
            left: Number(cluster.outcomes?.left) || 0,
            right: Number(cluster.outcomes?.right) || 0,
            up: Number(cluster.outcomes?.up) || 0,
            down: Number(cluster.outcomes?.down) || 0
          },
          weight: Number(cluster.weight) || 1,
          samples: Number(cluster.samples) || 1
        }));
      neuralNetwork = restoreNeuralNetwork(stored.neuralNetwork);
      pacmanNetwork = restorePacmanNetwork(stored.pacmanNetwork);
    } catch {
      resetLearning();
    }
  }

  /**
   * Saves persistent ghost learning into browser storage.
   */
  function saveLearning() {
    if (!learnedTurns || !behaviorClusters || !aiEnabled("persistentMemory")) return;

    try {
      localStorage.setItem(LEARNING_STORAGE_KEY, JSON.stringify({
        learnedTurns: [...learnedTurns.entries()],
        behaviorClusters,
        neuralNetwork,
        pacmanNetwork
      }));
      learnedMovesSinceSave = 0;
    } catch {
      // Storage can fail in private browsing or full-quota situations.
    }
  }

  /**
   * Marks learning state as changed and periodically saves it.
   */
  function noteLearningChanged() {
    learnedMovesSinceSave++;
    if (learnedMovesSinceSave >= LEARNING_SAVE_INTERVAL) saveLearning();
  }

  /**
   * Builds a stable key for a maze tile.
   *
   * @param {{c: number, r: number}} tile Tile to identify.
   * @returns {string} Unique tile key.
   */
  function tileKey(tile) {
    return `${tile.c},${tile.r}`;
  }

  /**
   * Builds a stable key for a cached route distance.
   *
   * @param {{c: number, r: number}} start Starting tile.
   * @param {{c: number, r: number}} target Target tile.
   * @returns {string} Cache key for the tile pair.
   */
  function routeKey(start, target) {
    return `${tileKey(start)}>${tileKey(target)}`;
  }

  /**
   * Stores a route distance while keeping the cache bounded.
   *
   * @param {{c: number, r: number}} start Starting tile.
   * @param {{c: number, r: number}} target Target tile.
   * @param {number} distance Distance to cache.
   */
  function rememberRouteDistance(start, target, distance) {
    while (routeDistanceCache.size > ROUTE_DISTANCE_CACHE_LIMIT - 2) {
      routeDistanceCache.delete(routeDistanceCache.keys().next().value);
    }
    routeDistanceCache.set(routeKey(start, target), distance);
    routeDistanceCache.set(routeKey(target, start), distance);
  }

  /**
   * Builds a learning key for a tile entered from a specific direction.
   *
   * @param {{c: number, r: number}} tile Tile where a decision happened.
   * @param {string} incomingDir Direction the player was traveling.
   * @returns {string} Unique learned-turn key.
   */
  function turnKey(tile, incomingDir) {
    return `${tileKey(tile)},${incomingDir}`;
  }

  /**
   * Updates the player movement model after a successful move.
   *
   * @param {{c: number, r: number, dir: string}} previousPlayer Player state before moving.
   */
  function learnPlayerMove(previousPlayer) {
    const behaviorVector = playerContextVector(previousPlayer, player.nextDir);
    let trained = false;

    playerTrail.push({ c: player.c, r: player.r, dir: player.dir });
    if (playerTrail.length > PLAYER_MEMORY_LIMIT) playerTrail.shift();

    if (aiEnabled("patternMemory")) {
      decayLearnedTurns();
      const key = turnKey(previousPlayer, previousPlayer.dir);
      const counts = learnedTurns.get(key) || { left: 0, right: 0, up: 0, down: 0 };
      counts[player.dir]++;
      learnedTurns.set(key, counts);
      trained = true;
    }

    if (aiEnabled("behaviorClusters")) {
      decayBehaviorClusters();
      learnBehaviorCluster(behaviorVector, player.dir);
      trained = true;
    }

    if (aiEnabled("neuralNetwork")) {
      trainNeuralNetwork(behaviorVector, player.dir);
      trained = true;
    }

    if (trained) {
      noteLearningChanged();
    }
  }

  /**
   * Fades old player habits so new patterns become more important over time.
   */
  function decayLearnedTurns() {
    learnedTurns.forEach((counts, key) => {
      let total = 0;
      DIR_NAMES.forEach(dirName => {
        counts[dirName] *= LEARNING_DECAY;
        total += counts[dirName];
      });
      if (total < MIN_LEARNED_WEIGHT) learnedTurns.delete(key);
    });
  }

  /**
   * Fades old movement clusters so recent behavior can reshape predictions.
   */
  function decayBehaviorClusters() {
    behaviorClusters = behaviorClusters.filter(cluster => {
      let total = 0;
      cluster.weight *= BEHAVIOR_CLUSTER_DECAY;
      DIR_NAMES.forEach(dirName => {
        cluster.outcomes[dirName] *= BEHAVIOR_CLUSTER_DECAY;
        total += cluster.outcomes[dirName];
      });
      return cluster.weight >= MIN_LEARNED_WEIGHT || total >= MIN_LEARNED_WEIGHT;
    });
  }

  /**
   * Converts a direction into a one-hot feature vector.
   *
   * @param {string} dirName Direction key from DIRS.
   * @returns {number[]} One-hot encoded direction.
   */
  function directionVector(dirName) {
    return DIR_NAMES.map(d => d === dirName ? 1 : 0);
  }

  /**
   * Encodes which directions are available from a tile.
   *
   * @param {{c: number, r: number}} tile Tile to inspect.
   * @returns {number[]} Open-direction feature vector.
   */
  function openDirectionVector(tile) {
    return DIR_NAMES.map(dirName => canMove(tile, dirName) ? 1 : 0);
  }

  /**
   * Reads a recent movement direction as a feature vector.
   *
   * @param {number} offset Number of trail entries back from the latest move.
   * @returns {number[]} One-hot encoded recent direction.
   */
  function recentDirectionVector(offset) {
    const sample = playerTrail[playerTrail.length - offset];
    return directionVector(sample?.dir || player.dir);
  }

  /**
   * Builds an unsupervised learning vector for Pac-Man's current context.
   *
   * @param {{c: number, r: number, dir: string}} tile Tile and travel direction.
   * @param {string} intendedDir Direction Pac-Man is currently trying to take.
   * @returns {number[]} Numeric behavior features for clustering.
   */
  function playerContextVector(tile, intendedDir) {
    return [
      tile.c / (COLS - 1),
      tile.r / (ROWS - 1),
      ...directionVector(tile.dir),
      ...directionVector(intendedDir),
      ...openDirectionVector(tile),
      ...recentDirectionVector(1),
      ...recentDirectionVector(2)
    ];
  }

  /**
   * Creates a small feed-forward neural network for next-direction prediction.
   *
   * @returns {{inputWeights: number[][], hiddenBiases: number[], outputWeights: number[][], outputBiases: number[], samples: number}} Neural network state.
   */
  function createNeuralNetwork() {
    return {
      inputWeights: Array.from({ length: NN_HIDDEN }, () => (
        Array.from({ length: NN_INPUTS }, () => (Math.random() * 2 - 1) * 0.35)
      )),
      hiddenBiases: Array.from({ length: NN_HIDDEN }, () => (Math.random() * 2 - 1) * 0.1),
      outputWeights: Array.from({ length: NN_OUTPUTS }, () => (
        Array.from({ length: NN_HIDDEN }, () => (Math.random() * 2 - 1) * 0.35)
      )),
      outputBiases: Array.from({ length: NN_OUTPUTS }, () => 0),
      samples: 0,
      rollingAccuracy: 0,
      rollingConfidence: 0,
      rollingLoss: null
    };
  }

  /**
   * Reads a finite numeric value with a fallback.
   *
   * @param {unknown} value Value to read.
   * @param {number} fallback Fallback when the value is not finite.
   * @returns {number} Finite number.
   */
  function finiteNumber(value, fallback) {
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  }

  /**
   * Restores a neural network from persisted data, filling bad values safely.
   *
   * @param {object} stored Persisted neural-network state.
   * @returns {{inputWeights: number[][], hiddenBiases: number[], outputWeights: number[][], outputBiases: number[], samples: number}} Restored network.
   */
  function restoreNeuralNetwork(stored) {
    const network = createNeuralNetwork();
    if (!stored) return network;

    network.inputWeights = network.inputWeights.map((row, h) => (
      row.map((value, i) => finiteNumber(stored.inputWeights?.[h]?.[i], value))
    ));
    network.hiddenBiases = network.hiddenBiases.map((value, h) => finiteNumber(stored.hiddenBiases?.[h], value));
    network.outputWeights = network.outputWeights.map((row, o) => (
      row.map((value, h) => finiteNumber(stored.outputWeights?.[o]?.[h], value))
    ));
    network.outputBiases = network.outputBiases.map((value, o) => finiteNumber(stored.outputBiases?.[o], value));
    network.samples = Math.max(0, Number(stored.samples) || 0);
    network.rollingAccuracy = finiteNumber(stored.rollingAccuracy, 0);
    network.rollingConfidence = finiteNumber(stored.rollingConfidence, 0);
    network.rollingLoss = stored.rollingLoss === null ? null : finiteNumber(stored.rollingLoss, null);
    return network;
  }

  /**
   * Pads or trims a feature vector for the neural network.
   *
   * @param {number[]} vector Source feature vector.
   * @returns {number[]} Fixed-size neural-network input.
   */
  function neuralInputVector(vector) {
    return Array.from({ length: NN_INPUTS }, (_, i) => Number(vector[i]) || 0);
  }

  /**
   * Converts raw neural output scores to probabilities.
   *
   * @param {number[]} scores Raw output scores.
   * @returns {number[]} Probability distribution.
   */
  function softmax(scores) {
    const maxScore = Math.max(...scores);
    const exps = scores.map(score => Math.exp(score - maxScore));
    const total = exps.reduce((sum, value) => sum + value, 0);
    return exps.map(value => value / total);
  }

  /**
   * Runs a forward pass through the neural network.
   *
   * @param {number[]} vector Movement-context features.
   * @returns {{input: number[], hidden: number[], probabilities: number[]} | null} Forward pass output.
   */
  function neuralForward(vector) {
    if (!neuralNetwork) return null;

    const input = neuralInputVector(vector);
    const hidden = neuralNetwork.inputWeights.map((weights, h) => {
      const sum = weights.reduce((total, weight, i) => total + weight * input[i], neuralNetwork.hiddenBiases[h]);
      return Math.tanh(sum);
    });
    const scores = neuralNetwork.outputWeights.map((weights, o) => (
      weights.reduce((total, weight, h) => total + weight * hidden[h], neuralNetwork.outputBiases[o])
    ));

    return { input, hidden, probabilities: softmax(scores) };
  }

  /**
   * Finds the most likely output index from a probability distribution.
   *
   * @param {number[]} probabilities Direction probabilities.
   * @returns {number} Index of the highest probability.
   */
  function predictedDirectionIndex(probabilities) {
    return probabilities.reduce((best, probability, i) => (
      probability > probabilities[best] ? i : best
    ), 0);
  }

  /**
   * Converts probabilities into a compact direction-keyed log object.
   *
   * @param {number[]} probabilities Direction probabilities.
   * @returns {{left: number, right: number, up: number, down: number}} Rounded probabilities.
   */
  function probabilityLog(probabilities) {
    return Object.fromEntries(DIR_NAMES.map((dirName, i) => [
      dirName,
      Number((probabilities[i] || 0).toFixed(3))
    ]));
  }

  /**
   * Records neural-network training metrics and emits periodic console logs.
   *
   * @param {object} metrics Latest training metrics.
   */
  function recordNeuralTrainingMetrics(metrics) {
    const { beforeProbabilities, afterProbabilities, targetIndex, rate, loss } = metrics;
    const predictedIndex = predictedDirectionIndex(beforeProbabilities);
    const confidence = beforeProbabilities[predictedIndex] || 0;
    const correct = predictedIndex === targetIndex ? 1 : 0;
    const decay = neuralNetwork.samples <= 1 ? 0 : NN_METRIC_DECAY;

    neuralNetwork.rollingLoss = neuralNetwork.rollingLoss === null
      ? loss
      : neuralNetwork.rollingLoss * decay + loss * (1 - decay);
    neuralNetwork.rollingAccuracy = neuralNetwork.rollingAccuracy * decay + correct * (1 - decay);
    neuralNetwork.rollingConfidence = neuralNetwork.rollingConfidence * decay + confidence * (1 - decay);

    neuralLogWindow.loss += loss;
    neuralLogWindow.correct += correct;
    neuralLogWindow.confidence += confidence;
    neuralLogWindow.count++;

    if (!aiEnabled("nnLogs") || neuralLogWindow.count < NN_LOG_INTERVAL) return;

    const averageLoss = neuralLogWindow.loss / neuralLogWindow.count;
    const averageAccuracy = neuralLogWindow.correct / neuralLogWindow.count;
    const averageConfidence = neuralLogWindow.confidence / neuralLogWindow.count;
    console.info("[Pac-Man NN]", {
      samples: neuralNetwork.samples,
      window: neuralLogWindow.count,
      averageLoss: Number(averageLoss.toFixed(4)),
      rollingLoss: Number(neuralNetwork.rollingLoss.toFixed(4)),
      averageAccuracy: Number(averageAccuracy.toFixed(3)),
      rollingAccuracy: Number(neuralNetwork.rollingAccuracy.toFixed(3)),
      averageConfidence: Number(averageConfidence.toFixed(3)),
      rollingConfidence: Number(neuralNetwork.rollingConfidence.toFixed(3)),
      learningRate: Number(rate.toFixed(5)),
      target: DIR_NAMES[targetIndex],
      predicted: DIR_NAMES[predictedIndex],
      before: probabilityLog(beforeProbabilities),
      after: probabilityLog(afterProbabilities),
      pressure: difficultyPressure(),
      horizon: predictionHorizon()
    });
    neuralLogWindow = { loss: 0, correct: 0, confidence: 0, count: 0 };
  }

  /**
   * Trains the neural network from one observed player movement.
   *
   * @param {number[]} vector Movement-context features before the move.
   * @param {string} outcomeDir Direction Pac-Man actually chose.
   */
  function trainNeuralNetwork(vector, outcomeDir) {
    const targetIndex = DIR_NAMES.indexOf(outcomeDir);
    const pass = neuralForward(vector);
    if (!pass || targetIndex < 0) return;

    const rate = NN_LEARNING_RATE / Math.sqrt(1 + neuralNetwork.samples / 120);
    const beforeProbabilities = [...pass.probabilities];
    const loss = -Math.log(Math.max(1e-6, beforeProbabilities[targetIndex] || 0));
    const outputErrors = pass.probabilities.map((probability, i) => probability - (i === targetIndex ? 1 : 0));
    const previousOutputWeights = neuralNetwork.outputWeights.map(row => [...row]);

    neuralNetwork.outputWeights = neuralNetwork.outputWeights.map((weights, o) => (
      weights.map((weight, h) => weight - rate * outputErrors[o] * pass.hidden[h])
    ));
    neuralNetwork.outputBiases = neuralNetwork.outputBiases.map((bias, o) => bias - rate * outputErrors[o]);

    neuralNetwork.inputWeights = neuralNetwork.inputWeights.map((weights, h) => {
      const hiddenError = (1 - pass.hidden[h] ** 2)
        * outputErrors.reduce((sum, error, o) => sum + previousOutputWeights[o][h] * error, 0);
      neuralNetwork.hiddenBiases[h] -= rate * hiddenError;
      return weights.map((weight, i) => weight - rate * hiddenError * pass.input[i]);
    });
    neuralNetwork.samples++;
    recordNeuralTrainingMetrics({
      beforeProbabilities,
      afterProbabilities: neuralForward(vector)?.probabilities || beforeProbabilities,
      targetIndex,
      rate,
      loss
    });
  }

  /**
   * Creates Pac-Man's policy network for automatic gameplay.
   *
   * @returns {{inputWeights: number[][], hiddenBiases: number[], outputWeights: number[][], outputBiases: number[], samples: number, onlineSamples: number, pretrainSamples: number, rollingReward: number | null, rollingLoss: number | null}} Pac-Man policy network.
   */
  function createPacmanNetwork() {
    return {
      inputWeights: Array.from({ length: PAC_NN_HIDDEN }, () => (
        Array.from({ length: PAC_NN_INPUTS }, () => (Math.random() * 2 - 1) * 0.28)
      )),
      hiddenBiases: Array.from({ length: PAC_NN_HIDDEN }, () => (Math.random() * 2 - 1) * 0.08),
      outputWeights: Array.from({ length: PAC_NN_OUTPUTS }, () => (
        Array.from({ length: PAC_NN_HIDDEN }, () => (Math.random() * 2 - 1) * 0.28)
      )),
      outputBiases: Array.from({ length: PAC_NN_OUTPUTS }, () => 0),
      samples: 0,
      onlineSamples: 0,
      pretrainSamples: 0,
      rollingReward: null,
      rollingLoss: null,
      rollingEpsilon: 1
    };
  }

  /**
   * Restores Pac-Man's policy network from persisted data.
   *
   * @param {object} stored Persisted policy network.
   * @returns {{inputWeights: number[][], hiddenBiases: number[], outputWeights: number[][], outputBiases: number[], samples: number, onlineSamples: number, pretrainSamples: number, rollingReward: number | null, rollingLoss: number | null}} Restored policy network.
   */
  function restorePacmanNetwork(stored) {
    const network = createPacmanNetwork();
    if (!stored) return network;

    network.inputWeights = network.inputWeights.map((row, h) => (
      row.map((value, i) => finiteNumber(stored.inputWeights?.[h]?.[i], value))
    ));
    network.hiddenBiases = network.hiddenBiases.map((value, h) => finiteNumber(stored.hiddenBiases?.[h], value));
    network.outputWeights = network.outputWeights.map((row, o) => (
      row.map((value, h) => finiteNumber(stored.outputWeights?.[o]?.[h], value))
    ));
    network.outputBiases = network.outputBiases.map((value, o) => finiteNumber(stored.outputBiases?.[o], value));
    network.samples = Math.max(0, Number(stored.samples) || 0);
    network.pretrainSamples = Math.max(0, Number(stored.pretrainSamples) || 0);
    const storedOnlineSamples = Number(stored.onlineSamples);
    network.onlineSamples = Math.max(
      0,
      Number.isFinite(storedOnlineSamples) ? storedOnlineSamples : network.samples - network.pretrainSamples
    );
    network.rollingReward = stored.rollingReward === null ? null : finiteNumber(stored.rollingReward, null);
    network.rollingLoss = stored.rollingLoss === null ? null : finiteNumber(stored.rollingLoss, null);
    network.rollingEpsilon = finiteNumber(stored.rollingEpsilon, 1);
    return network;
  }

  /**
   * Pads or trims a feature vector for Pac-Man's policy network.
   *
   * @param {number[]} vector Source feature vector.
   * @returns {number[]} Fixed-size policy-network input.
   */
  function pacmanInputVector(vector) {
    return Array.from({ length: PAC_NN_INPUTS }, (_, i) => Number(vector[i]) || 0);
  }

  /**
   * Runs Pac-Man's policy network and returns Q-values for directions.
   *
   * @param {number[]} vector Gameplay-state features.
   * @returns {{input: number[], hidden: number[], values: number[]} | null} Policy forward pass.
   */
  function pacmanForward(vector) {
    if (!pacmanNetwork) return null;

    const input = pacmanInputVector(vector);
    const hidden = pacmanNetwork.inputWeights.map((weights, h) => {
      const sum = weights.reduce((total, weight, i) => total + weight * input[i], pacmanNetwork.hiddenBiases[h]);
      return Math.tanh(sum);
    });
    const values = pacmanNetwork.outputWeights.map((weights, o) => (
      weights.reduce((total, weight, h) => total + weight * hidden[h], pacmanNetwork.outputBiases[o])
    ));

    return { input, hidden, values };
  }

  /**
   * Chooses Pac-Man's exploration rate for automatic play.
   *
   * @returns {number} Epsilon exploration rate.
   */
  function pacmanExplorationRate() {
    const onlineSamples = pacmanNetwork?.onlineSamples || 0;
    const pretrainReadiness = Math.min(1, (pacmanNetwork?.pretrainSamples || 0) / PAC_NN_PRETRAIN_TARGET);
    const initialExploration = 0.45 - pretrainReadiness * 0.28;
    return Math.max(0.04, initialExploration * Math.exp(-onlineSamples / 260));
  }

  /**
   * Limits a number to a range.
   *
   * @param {number} value Number to clamp.
   * @param {number} min Minimum value.
   * @param {number} max Maximum value.
   * @returns {number} Clamped number.
   */
  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  /**
   * Converts Q-values into a compact direction-keyed log object.
   *
   * @param {number[]} values Direction Q-values.
   * @returns {{left: number, right: number, up: number, down: number}} Rounded Q-values.
   */
  function qValueLog(values) {
    return Object.fromEntries(DIR_NAMES.map((dirName, i) => [
      dirName,
      Number((values[i] || 0).toFixed(3))
    ]));
  }

  /**
   * Records Pac-Man policy training metrics and emits periodic console logs.
   *
   * @param {object} metrics Latest training metrics.
   */
  function recordPacmanTrainingMetrics(metrics) {
    const { actionIndex, reward, loss, epsilon, beforeValues, afterValues, target } = metrics;
    const decay = pacmanNetwork.onlineSamples <= 1 ? 0 : PAC_NN_METRIC_DECAY;

    pacmanNetwork.rollingReward = pacmanNetwork.rollingReward === null
      ? reward
      : pacmanNetwork.rollingReward * decay + reward * (1 - decay);
    pacmanNetwork.rollingLoss = pacmanNetwork.rollingLoss === null
      ? loss
      : pacmanNetwork.rollingLoss * decay + loss * (1 - decay);
    pacmanNetwork.rollingEpsilon = epsilon;

    pacmanLogWindow.reward += reward;
    pacmanLogWindow.loss += loss;
    pacmanLogWindow.count++;

    if (!aiEnabled("pacmanLogs") || pacmanLogWindow.count < PAC_NN_LOG_INTERVAL) return;

    const averageReward = pacmanLogWindow.reward / pacmanLogWindow.count;
    const averageLoss = pacmanLogWindow.loss / pacmanLogWindow.count;
    console.info("[Pac-Man Policy NN]", {
      samples: pacmanNetwork.samples,
      onlineSamples: pacmanNetwork.onlineSamples,
      pretrainSamples: pacmanNetwork.pretrainSamples,
      window: pacmanLogWindow.count,
      averageReward: Number(averageReward.toFixed(4)),
      rollingReward: Number(pacmanNetwork.rollingReward.toFixed(4)),
      averageLoss: Number(averageLoss.toFixed(4)),
      rollingLoss: Number(pacmanNetwork.rollingLoss.toFixed(4)),
      epsilon: Number(epsilon.toFixed(3)),
      action: DIR_NAMES[actionIndex],
      target: Number(target.toFixed(3)),
      beforeQ: qValueLog(beforeValues),
      afterQ: qValueLog(afterValues),
      pressure: difficultyPressure(),
      nearestGhost: Number(nearestActiveGhostDistance(player).toFixed(2))
    });
    pacmanLogWindow = { reward: 0, loss: 0, count: 0 };
  }

  /**
   * Trains Pac-Man's policy network with one Q-learning update.
   *
   * @param {number[]} vector State before the action.
   * @param {string} actionDir Direction Pac-Man chose.
   * @param {number} reward Reward received after the action.
   * @param {number[]} nextVector State after the action.
   * @param {boolean} done True if the action ended the run.
   */
  function trainPacmanNetwork(vector, actionDir, reward, nextVector, done) {
    const actionIndex = DIR_NAMES.indexOf(actionDir);
    const pass = pacmanForward(vector);
    const nextPass = pacmanForward(nextVector);
    if (!pass || !nextPass || actionIndex < 0) return;

    const validNextIndexes = validDirs(player).map(dirName => DIR_NAMES.indexOf(dirName));
    const nextBestValue = validNextIndexes.length
      ? Math.max(...validNextIndexes.map(i => nextPass.values[i]))
      : Math.max(...nextPass.values);
    const target = reward + (done ? 0 : PAC_NN_GAMMA * nextBestValue);
    const error = pass.values[actionIndex] - target;
    const loss = error ** 2;
    const rate = PAC_NN_LEARNING_RATE / Math.sqrt(1 + (pacmanNetwork.onlineSamples || 0) / 180);
    const previousOutputWeights = pacmanNetwork.outputWeights.map(row => [...row]);

    pacmanNetwork.outputWeights[actionIndex] = pacmanNetwork.outputWeights[actionIndex].map((weight, h) => (
      weight - rate * error * pass.hidden[h]
    ));
    pacmanNetwork.outputBiases[actionIndex] -= rate * error;
    pacmanNetwork.inputWeights = pacmanNetwork.inputWeights.map((weights, h) => {
      const hiddenError = (1 - pass.hidden[h] ** 2) * previousOutputWeights[actionIndex][h] * error;
      pacmanNetwork.hiddenBiases[h] -= rate * hiddenError;
      return weights.map((weight, i) => weight - rate * hiddenError * pass.input[i]);
    });

    pacmanNetwork.samples++;
    pacmanNetwork.onlineSamples = (pacmanNetwork.onlineSamples || 0) + 1;
    noteLearningChanged();
    recordPacmanTrainingMetrics({
      actionIndex,
      reward,
      loss,
      epsilon: pacmanExplorationRate(),
      beforeValues: pass.values,
      afterValues: pacmanForward(vector)?.values || pass.values,
      target
    });
  }

  /**
   * Lists every non-wall tile in the maze.
   *
   * @returns {{c: number, r: number}[]} Walkable maze tiles.
   */
  function walkableTiles() {
    const tiles = [];
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        if (!isWall(c, r)) tiles.push({ c, r });
      }
    }
    return tiles;
  }

  /**
   * Chooses a random item from an array.
   *
   * @template T
   * @param {T[]} items Items to sample from.
   * @returns {T} Random item.
   */
  function randomItem(items) {
    return items[Math.floor(Math.random() * items.length)];
  }

  /**
   * Measures wrapped Manhattan distance between two tiles.
   *
   * @param {{c: number, r: number}} a First tile.
   * @param {{c: number, r: number}} b Second tile.
   * @returns {number} Approximate tile distance.
   */
  function manhattanDistance(a, b) {
    const dc = Math.min(Math.abs(a.c - b.c), COLS - Math.abs(a.c - b.c));
    return dc + Math.abs(a.r - b.r);
  }

  /**
   * Finds the nearest grid symbol using fast approximate distance.
   *
   * @param {{c: number, r: number}} origin Starting tile.
   * @param {string[]} symbols Grid symbols to search for.
   * @returns {{tile: {c: number, r: number} | null, distance: number}} Nearest matching tile.
   */
  function nearestApproxGridTarget(origin, symbols) {
    const targets = [];
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        if (!symbols.includes(grid[r][c])) continue;
        targets.push({ c, r });
      }
    }
    return nearestApproxTargetFromList(origin, targets);
  }

  /**
   * Finds the nearest target from a prebuilt target list.
   *
   * @param {{c: number, r: number}} origin Starting tile.
   * @param {{c: number, r: number}[]} targets Candidate target tiles.
   * @returns {{tile: {c: number, r: number} | null, distance: number}} Nearest tile.
   */
  function nearestApproxTargetFromList(origin, targets) {
    let best = { tile: null, distance: Infinity };
    for (const target of targets) {
      const distance = manhattanDistance(origin, target);
      if (distance < best.distance) best = { tile: target, distance };
    }
    return best;
  }

  /**
   * Builds cached maze facts used by high-volume Pac-Man pretraining.
   *
   * @returns {{tiles: object[], pellets: object[], powers: object[], dataByKey: Map<string, object>}} Pretraining cache.
   */
  function buildPacmanPretrainCache() {
    const tiles = walkableTiles();
    const pellets = [];
    const powers = [];

    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const ch = grid[r][c];
        if (ch === "." || ch === "o") pellets.push({ c, r });
        if (ch === "o") powers.push({ c, r });
      }
    }

    const dataByKey = new Map();
    tiles.forEach(tile => {
      const valid = validDirs(tile);
      dataByKey.set(tileKey(tile), {
        tile,
        ch: grid[tile.r]?.[tile.c] || " ",
        valid,
        open: DIR_NAMES.map(dirName => valid.includes(dirName) ? 1 : 0),
        nearestPellet: nearestApproxTargetFromList(tile, pellets),
        nearestPowerPellet: nearestApproxTargetFromList(tile, powers),
        nextByDir: Object.fromEntries(valid.map(dirName => [dirName, adjacentTile(tile, dirName)]))
      });
    });

    return { tiles, pellets, powers, dataByKey };
  }

  /**
   * Reads cached pretraining data for a tile.
   *
   * @param {object} cache Pretraining cache.
   * @param {{c: number, r: number}} tile Tile to read.
   * @returns {object | null} Cached tile data.
   */
  function pacmanPretrainData(cache, tile) {
    return cache?.dataByKey?.get(tileKey(tile)) || null;
  }

  /**
   * Samples a tile near a training origin.
   *
   * @param {{c: number, r: number}} origin Origin tile.
   * @param {object} cache Pretraining cache.
   * @param {number} minDistance Minimum approximate distance.
   * @param {number} maxDistance Maximum approximate distance.
   * @returns {{c: number, r: number}} Sampled tile.
   */
  function randomNearbyPretrainTile(origin, cache, minDistance, maxDistance) {
    for (let attempt = 0; attempt < 18; attempt++) {
      const tile = randomItem(cache.tiles);
      const distance = manhattanDistance(origin, tile);
      if (distance >= minDistance && distance <= maxDistance) return tile;
    }
    return randomItem(cache.tiles);
  }

  /**
   * Encodes a sampled Pac-Man state for policy pretraining.
   *
   * @param {{c: number, r: number, dir: string}} samplePlayer Sampled Pac-Man.
   * @param {{c: number, r: number}[]} sampleGhosts Sampled ghosts.
   * @param {number} frightened Sampled frightened flag.
   * @param {number} pressure Sampled difficulty pressure.
   * @param {object | null} data Cached tile data.
   * @returns {number[]} Policy feature vector.
   */
  function pacmanPretrainFeatureVector(samplePlayer, sampleGhosts, frightened, pressure, data) {
    const nearestPellet = data?.nearestPellet || nearestApproxGridTarget(samplePlayer, [".", "o"]);
    const nearestPowerPellet = data?.nearestPowerPellet || nearestApproxGridTarget(samplePlayer, ["o"]);
    const pelletVector = nearestPellet.tile
      ? [(nearestPellet.tile.c - samplePlayer.c) / COLS, (nearestPellet.tile.r - samplePlayer.r) / ROWS]
      : [0, 0];
    const powerVector = nearestPowerPellet.tile
      ? [(nearestPowerPellet.tile.c - samplePlayer.c) / COLS, (nearestPowerPellet.tile.r - samplePlayer.r) / ROWS]
      : [0, 0];

    return [
      samplePlayer.c / (COLS - 1),
      samplePlayer.r / (ROWS - 1),
      ...directionVector(samplePlayer.dir),
      ...(data?.open || openDirectionVector(samplePlayer)),
      frightened,
      pressure,
      ...sampleGhosts.flatMap(ghost => pretrainGhostVector(samplePlayer, ghost, frightened)),
      ...pelletVector,
      ...powerVector
    ];
  }

  /**
   * Encodes a sampled ghost for Pac-Man policy pretraining.
   *
   * @param {{c: number, r: number}} samplePlayer Sampled Pac-Man tile.
   * @param {{c: number, r: number} | null} ghost Sampled ghost tile.
   * @param {number} frightened Sampled frightened state.
   * @returns {number[]} Fixed-size ghost feature vector.
   */
  function pretrainGhostVector(samplePlayer, ghost, frightened) {
    if (!ghost) return [0, 0, 1, 0, 0];

    const distance = manhattanDistance(samplePlayer, ghost);
    return [
      clamp((ghost.c - samplePlayer.c) / COLS, -1, 1),
      clamp((ghost.r - samplePlayer.r) / ROWS, -1, 1),
      clamp(distance / (ROWS + COLS), 0, 1),
      frightened ? 0 : 1,
      frightened ? 1 : 0
    ];
  }

  /**
   * Builds one randomized Pac-Man policy pretraining state.
   *
   * @param {object} cache Pretraining cache.
   * @returns {{player: {c: number, r: number, dir: string}, data: object | null, ghosts: {c: number, r: number}[], frightened: number, pressure: number, vector: number[]}} Sampled state.
   */
  function samplePacmanPretrainState(cache) {
    const samplePlayer = { ...randomItem(cache.tiles), dir: randomItem(DIR_NAMES) };
    const data = pacmanPretrainData(cache, samplePlayer);
    const frightened = Math.random() < 0.18 ? 1 : 0;
    const pressure = Math.random();
    const sampleGhosts = Array.from({ length: 4 }, (_, i) => {
      const nearChance = frightened ? 0.5 - i * 0.08 : 0.78 - i * 0.12;
      if (Math.random() < nearChance) {
        return randomNearbyPretrainTile(samplePlayer, cache, i === 0 ? 1 : 3, i === 0 ? 8 : 16);
      }
      return randomItem(cache.tiles);
    })
      .sort((a, b) => manhattanDistance(samplePlayer, a) - manhattanDistance(samplePlayer, b));

    return {
      player: samplePlayer,
      data,
      ghosts: sampleGhosts,
      frightened,
      pressure,
      vector: pacmanPretrainFeatureVector(samplePlayer, sampleGhosts, frightened, pressure, data)
    };
  }

  /**
   * Scores a sampled Pac-Man action for supervised warm-start training.
   *
   * @param {object} sample Sampled pretraining state.
   * @param {string} dirName Direction to evaluate.
   * @param {object} cache Pretraining cache.
   * @returns {number} Expert heuristic score.
   */
  function pacmanPretrainScore(sample, dirName, cache) {
    const next = sample.data?.nextByDir?.[dirName] || adjacentTile(sample.player, dirName);
    if (!next) return -999;

    const nextData = pacmanPretrainData(cache, next);
    const ch = nextData?.ch ?? grid[next.r]?.[next.c];
    const ghostDistances = sample.ghosts
      .map(ghost => manhattanDistance(next, ghost))
      .sort((a, b) => a - b);
    const ghostDistance = ghostDistances[0] ?? ROWS + COLS;
    const secondGhostDistance = ghostDistances[1] ?? ROWS + COLS;
    const pelletDistance = nextData?.nearestPellet?.distance ?? nearestApproxGridTarget(next, [".", "o"]).distance;
    const powerDistance = nextData?.nearestPowerPellet?.distance ?? nearestApproxGridTarget(next, ["o"]).distance;
    const escapeRoutes = nextData?.valid?.length ?? validDirs(next).length;
    const dangerWeight = 1 + sample.pressure * 0.75;
    let score = 0;

    if (ch === ".") score += 2.7;
    if (ch === "o") score += sample.frightened ? 4.6 : 8;
    score += escapeRoutes * 0.18;
    if (sample.frightened) {
      score += clamp(11 - ghostDistance, -3, 11) * 0.32;
      if (Number.isFinite(pelletDistance)) score -= pelletDistance * 0.025;
    } else {
      score += clamp(ghostDistance, 0, 15) * 0.38 * dangerWeight;
      score += clamp(secondGhostDistance, 0, 12) * 0.08;
      if (ghostDistance <= 1) score -= 18;
      else if (ghostDistance <= 2) score -= 9;
      else if (ghostDistance <= 3) score -= 4.5;
      if (escapeRoutes <= 1 && ghostDistance <= 6) score -= 3.5;
      if (Number.isFinite(powerDistance) && ghostDistance <= 8) score -= powerDistance * 0.16;
      if (Number.isFinite(pelletDistance)) score -= pelletDistance * 0.05;
    }
    if (dirName === REVERSE[sample.player.dir]) score -= 0.25;
    if (dirName === sample.player.dir && ghostDistance > 4) score += 0.08;
    return score;
  }

  /**
   * Trains Pac-Man policy with one expert action.
   *
   * @param {number[]} vector Sampled gameplay state.
   * @param {string[]} validDirsForState Directions available in the sampled state.
   * @param {string} targetDir Expert direction.
   * @returns {{loss: number, correct: boolean} | null} Training metrics.
   */
  function trainPacmanWarmStart(vector, validDirsForState, targetDir) {
    const pass = pacmanForward(vector);
    const targetIndex = DIR_NAMES.indexOf(targetDir);
    if (!pass || targetIndex < 0) return null;

    const validIndexes = new Set(validDirsForState.map(dirName => DIR_NAMES.indexOf(dirName)));
    const targets = DIR_NAMES.map((_, i) => validIndexes.has(i) ? -0.2 : -0.9);
    targets[targetIndex] = 1.2;
    const prediction = validDirsForState
      .map(dirName => ({ dirName, value: pass.values[DIR_NAMES.indexOf(dirName)] }))
      .sort((a, b) => b.value - a.value)[0]?.dirName;
    const outputErrors = pass.values.map((value, i) => value - targets[i]);
    const loss = outputErrors.reduce((sum, error) => sum + error ** 2, 0) / outputErrors.length;
    const previousOutputWeights = pacmanNetwork.outputWeights.map(row => [...row]);

    pacmanNetwork.outputWeights = pacmanNetwork.outputWeights.map((weights, o) => (
      weights.map((weight, h) => weight - PAC_NN_PRETRAIN_RATE * outputErrors[o] * pass.hidden[h])
    ));
    pacmanNetwork.outputBiases = pacmanNetwork.outputBiases.map((bias, o) => bias - PAC_NN_PRETRAIN_RATE * outputErrors[o]);
    pacmanNetwork.inputWeights = pacmanNetwork.inputWeights.map((weights, h) => {
      const hiddenError = (1 - pass.hidden[h] ** 2)
        * outputErrors.reduce((sum, error, o) => sum + previousOutputWeights[o][h] * error, 0);
      pacmanNetwork.hiddenBiases[h] -= PAC_NN_PRETRAIN_RATE * hiddenError;
      return weights.map((weight, i) => weight - PAC_NN_PRETRAIN_RATE * hiddenError * pass.input[i]);
    });
    pacmanNetwork.samples++;
    pacmanNetwork.pretrainSamples++;
    return { loss, correct: prediction === targetDir };
  }

  /**
   * Trains a bounded number of Pac-Man pretraining samples.
   *
   * @param {object} cache Pretraining cache.
   * @param {number} iterations Maximum number of sampled states to train.
   * @param {number} budgetMs Maximum wall-clock time for this slice.
   * @returns {{trained: number, loss: number, correct: number}} Training metrics.
   */
  function trainPacmanPretrainSamples(cache, iterations, budgetMs = PAC_NN_PRETRAIN_BUDGET_MS) {
    let trained = 0;
    let loss = 0;
    let correct = 0;
    const startedAt = typeof performance !== "undefined" ? performance.now() : Date.now();

    for (let i = 0; i < iterations; i++) {
      const sample = samplePacmanPretrainState(cache);
      const valid = sample.data?.valid || validDirs(sample.player);
      if (!valid.length) continue;

      const best = valid
        .map(dirName => ({ dirName, score: pacmanPretrainScore(sample, dirName, cache) }))
        .sort((a, b) => b.score - a.score)[0].dirName;
      const metrics = trainPacmanWarmStart(sample.vector, valid, best);
      if (!metrics) continue;
      trained++;
      loss += metrics.loss;
      if (metrics.correct) correct++;

      if (trained >= PAC_NN_PRETRAIN_MIN_CHUNK) {
        const elapsed = (typeof performance !== "undefined" ? performance.now() : Date.now()) - startedAt;
        if (elapsed >= budgetMs) break;
      }
    }

    return { trained, loss, correct };
  }

  /**
   * Checks whether Pac-Man pretraining is currently running.
   *
   * @returns {boolean} True when a pretraining job is active.
   */
  function isPacmanPretraining() {
    return Boolean(pacmanPretrainJob?.running);
  }

  /**
   * Logs one window of Pac-Man pretraining convergence metrics.
   *
   * @param {object} job Active pretraining job.
   * @param {string} status Current job status.
   */
  function logPacmanPretrainProgress(job, status) {
    if (!aiEnabled("pacmanLogs") || !job.trainedSinceLog) return;

    const samples = pacmanNetwork?.pretrainSamples || 0;
    const elapsedMs = (typeof performance !== "undefined" ? performance.now() : Date.now()) - job.startedAt;
    console.info("[Pac-Man Pretrain]", {
      status,
      samples,
      target: job.target,
      progress: Number(((samples / job.target) * 100).toFixed(2)),
      window: job.trainedSinceLog,
      accuracy: Number((job.correctSinceLog / job.trainedSinceLog).toFixed(3)),
      averageLoss: Number((job.lossSinceLog / job.trainedSinceLog).toFixed(4)),
      totalPolicySamples: pacmanNetwork?.samples || 0,
      elapsedSeconds: Number((elapsedMs / 1000).toFixed(1))
    });

    job.trainedSinceLog = 0;
    job.lossSinceLog = 0;
    job.correctSinceLog = 0;
  }

  /**
   * Finishes the active Pac-Man pretraining job.
   *
   * @param {object} job Active pretraining job.
   * @param {string} status Final status label.
   */
  function finishPacmanPretraining(job, status) {
    if (!job || pacmanPretrainJob?.id !== job.id) return;

    job.running = false;
    logPacmanPretrainProgress(job, status);
    saveLearning();
    pacmanPretrainJob = null;
    updateAiStats();
  }

  /**
   * Schedules the next non-blocking pretraining slice.
   *
   * @param {object} job Active pretraining job.
   */
  function schedulePacmanPretrainChunk(job) {
    const runner = (deadline) => runPacmanPretrainChunk(job.id, deadline);
    if (typeof window !== "undefined" && typeof window.requestIdleCallback === "function") {
      window.requestIdleCallback(runner, { timeout: 90 });
    } else {
      setTimeout(runner, 0);
    }
  }

  /**
   * Runs one non-blocking slice of Pac-Man pretraining.
   *
   * @param {number} jobId Active job identifier.
   * @param {{timeRemaining?: function(): number} | undefined} deadline Browser idle callback deadline.
   */
  function runPacmanPretrainChunk(jobId, deadline) {
    const job = pacmanPretrainJob;
    if (!job || !job.running || job.id !== jobId) return;

    if (!aiEnabled("pacmanNn") || !aiEnabled("pacmanPretrain") || !pacmanNetwork || !grid) {
      stopPacmanPretraining();
      return;
    }

    const remaining = job.target - (pacmanNetwork.pretrainSamples || 0);
    if (remaining <= 0) {
      finishPacmanPretraining(job, "complete");
      return;
    }

    const idleBudget = typeof deadline?.timeRemaining === "function"
      ? Math.max(4, Math.min(PAC_NN_PRETRAIN_BUDGET_MS, deadline.timeRemaining()))
      : PAC_NN_PRETRAIN_BUDGET_MS;
    const metrics = trainPacmanPretrainSamples(job.cache, Math.min(PAC_NN_PRETRAIN_CHUNK, remaining), idleBudget);
    if (!metrics.trained) {
      finishPacmanPretraining(job, "stopped");
      return;
    }

    job.trainedSinceLog += metrics.trained;
    job.lossSinceLog += metrics.loss;
    job.correctSinceLog += metrics.correct;
    learnedMovesSinceSave += metrics.trained;

    if (learnedMovesSinceSave >= PAC_NN_PRETRAIN_BATCH) saveLearning();

    if ((pacmanNetwork.pretrainSamples || 0) >= job.target) {
      finishPacmanPretraining(job, "complete");
      return;
    }

    if (job.trainedSinceLog >= PAC_NN_PRETRAIN_BATCH) {
      logPacmanPretrainProgress(job, "training");
    }

    updateAiStats();
    schedulePacmanPretrainChunk(job);
  }

  /**
   * Starts or extends non-blocking Pac-Man pretraining.
   *
   * @param {number} targetSamples Desired cumulative pretraining sample count.
   */
  function startPacmanPretraining(targetSamples = PAC_NN_PRETRAIN_TARGET) {
    if (!aiEnabled("pacmanNn") || !aiEnabled("pacmanPretrain") || !pacmanNetwork || !grid) {
      updateAiStats();
      return;
    }

    const current = pacmanNetwork.pretrainSamples || 0;
    const target = Math.max(current, Math.floor(targetSamples));
    if (current >= target) {
      updateAiStats();
      return;
    }

    if (pacmanPretrainJob?.running) {
      pacmanPretrainJob.target = Math.max(pacmanPretrainJob.target, target);
      updateAiStats();
      return;
    }

    pacmanPretrainJob = {
      id: ++pacmanPretrainRunId,
      running: true,
      target,
      cache: buildPacmanPretrainCache(),
      startedAt: typeof performance !== "undefined" ? performance.now() : Date.now(),
      trainedSinceLog: 0,
      lossSinceLog: 0,
      correctSinceLog: 0
    };

    if (aiEnabled("pacmanLogs")) {
      console.info("[Pac-Man Pretrain]", {
        status: "started",
        samples: current,
        target,
        chunk: PAC_NN_PRETRAIN_CHUNK
      });
    }

    updateAiStats();
    schedulePacmanPretrainChunk(pacmanPretrainJob);
  }

  /**
   * Pauses active Pac-Man pretraining and saves current progress.
   */
  function stopPacmanPretraining() {
    const job = pacmanPretrainJob;
    if (!job) {
      updatePretrainButton();
      return;
    }

    job.running = false;
    logPacmanPretrainProgress(job, "paused");
    saveLearning();
    pacmanPretrainJob = null;
    updateAiStats();
  }

  /**
   * Queues Pac-Man policy pretraining without blocking gameplay.
   *
   * @param {number} iterations Minimum number of additional samples to request.
   */
  function pretrainPacmanPolicy(iterations = PAC_NN_PRETRAIN_BATCH) {
    const current = pacmanNetwork?.pretrainSamples || 0;
    const target = Math.max(PAC_NN_PRETRAIN_TARGET, current + Math.max(0, iterations));
    startPacmanPretraining(target);
  }

  /**
   * Runs automatic warm-start training up to the configured target.
   */
  function maybeWarmStartPacmanPolicy() {
    if (!aiEnabled("pacmanPretrain") || !aiEnabled("pacmanNn") || !grid) return;

    const remaining = PAC_NN_PRETRAIN_TARGET - (pacmanNetwork?.pretrainSamples || 0);
    if (remaining > 0) startPacmanPretraining(PAC_NN_PRETRAIN_TARGET);
  }

  /**
   * Measures squared distance between two feature vectors.
   *
   * @param {number[]} a First vector.
   * @param {number[]} b Second vector.
   * @returns {number} Squared Euclidean distance.
   */
  function squaredDistance(a, b) {
    return a.reduce((total, value, i) => total + (value - (Number(b[i]) || 0)) ** 2, 0);
  }

  /**
   * Finds the nearest learned movement cluster to a behavior vector.
   *
   * @param {number[]} vector Behavior vector to match.
   * @returns {{cluster: object, distance: number} | null} Nearest cluster match.
   */
  function nearestBehaviorCluster(vector) {
    if (!behaviorClusters.length) return null;
    return behaviorClusters
      .map(cluster => ({
        cluster,
        distance: Math.sqrt(squaredDistance(vector, cluster.center))
      }))
      .sort((a, b) => a.distance - b.distance)[0];
  }

  /**
   * Creates a new unsupervised behavior cluster.
   *
   * @param {number[]} vector Initial cluster center.
   * @param {string} outcomeDir Direction Pac-Man chose after this context.
   * @returns {{center: number[], outcomes: object, weight: number, samples: number}} Cluster record.
   */
  function createBehaviorCluster(vector, outcomeDir) {
    const outcomes = { left: 0, right: 0, up: 0, down: 0 };
    outcomes[outcomeDir] = 1;
    return {
      center: [...vector],
      outcomes,
      weight: 1,
      samples: 1
    };
  }

  /**
   * Trains the online movement clusters from one observed player context.
   *
   * @param {number[]} vector Movement-context features.
   * @param {string} outcomeDir Direction Pac-Man chose after this context.
   */
  function learnBehaviorCluster(vector, outcomeDir) {
    const match = nearestBehaviorCluster(vector);
    const shouldCreate = !match || match.distance > BEHAVIOR_CLUSTER_THRESHOLD;

    if (shouldCreate && behaviorClusters.length < BEHAVIOR_CLUSTER_LIMIT) {
      behaviorClusters.push(createBehaviorCluster(vector, outcomeDir));
      return;
    }

    if (shouldCreate) {
      const weakest = behaviorClusters.sort((a, b) => a.weight - b.weight)[0];
      Object.assign(weakest, createBehaviorCluster(vector, outcomeDir));
      return;
    }

    const cluster = match.cluster;
    const rate = Math.min(BEHAVIOR_CLUSTER_RATE, 1 / (cluster.samples + 1));
    cluster.center = cluster.center.map((value, i) => value + (vector[i] - value) * rate);
    cluster.outcomes[outcomeDir]++;
    cluster.weight++;
    cluster.samples++;
  }

  /**
   * Returns the open tile reached by moving once in a direction.
   *
   * @param {{c: number, r: number}} tile Starting tile.
   * @param {string} dirName Direction key from DIRS.
   * @returns {{c: number, r: number} | null} Adjacent tile or null when blocked.
   */
  function adjacentTile(tile, dirName) {
    const d = DIRS[dirName];
    if (!d) return null;
    const nc = tile.c + d.dc;
    const nr = tile.r + d.dr;
    if (isWall(nc, nr)) return null;
    return { c: wrappedC(nc), r: nr };
  }

  /**
   * Finds the nearest open tile to a requested target.
   *
   * @param {{c: number, r: number}} target Requested target tile.
   * @returns {{c: number, r: number}} Reachable open tile near the target.
   */
  function nearestOpenTile(target) {
    const start = {
      c: wrappedC(target.c),
      r: Math.max(0, Math.min(ROWS - 1, target.r))
    };
    if (!isWall(start.c, start.r)) return start;

    const queue = [start];
    const visited = new Set([tileKey(start)]);
    for (let i = 0; i < queue.length; i++) {
      const current = queue[i];
      for (const dirName of DIR_NAMES) {
        const d = DIRS[dirName];
        const next = {
          c: wrappedC(current.c + d.dc),
          r: current.r + d.dr
        };
        if (next.r < 0 || next.r >= ROWS) continue;
        const key = tileKey(next);
        if (visited.has(key)) continue;
        if (!isWall(next.c, next.r)) return next;
        visited.add(key);
        queue.push(next);
      }
    }

    return player ? { c: player.c, r: player.r } : start;
  }

  /**
   * Measures the shortest route length between two tiles.
   *
   * @param {{c: number, r: number}} start Starting tile.
   * @param {{c: number, r: number}} target Target tile.
   * @returns {number} Number of tile steps, or Infinity when unreachable.
   */
  function routeDistance(start, target) {
    const origin = nearestOpenTile(start);
    const destination = nearestOpenTile(target);
    if (tileKey(origin) === tileKey(destination)) return 0;

    const cached = routeDistanceCache.get(routeKey(origin, destination));
    if (cached != null) return cached;

    const queue = [{ ...origin, distance: 0 }];
    const visited = new Set([tileKey(origin)]);
    for (let i = 0; i < queue.length; i++) {
      const current = queue[i];
      for (const dirName of DIR_NAMES) {
        const next = adjacentTile(current, dirName);
        if (!next) continue;
        const key = tileKey(next);
        if (visited.has(key)) continue;
        if (key === tileKey(destination)) {
          const distance = current.distance + 1;
          rememberRouteDistance(origin, destination, distance);
          return distance;
        }
        visited.add(key);
        queue.push({ ...next, distance: current.distance + 1 });
      }
    }

    rememberRouteDistance(origin, destination, Infinity);
    return Infinity;
  }

  /**
   * Finds the nearest tile containing one of the requested maze symbols.
   *
   * @param {{c: number, r: number}} origin Starting tile.
   * @param {string[]} symbols Grid symbols to search for.
   * @returns {{tile: {c: number, r: number} | null, distance: number}} Nearest matching tile and route distance.
   */
  function nearestGridTarget(origin, symbols) {
    const wanted = new Set(symbols);
    const start = nearestOpenTile(origin);
    if (wanted.has(grid[start.r]?.[start.c])) return { tile: { ...start }, distance: 0 };

    const queue = [{ ...start, distance: 0 }];
    const visited = new Set([tileKey(start)]);
    for (let i = 0; i < queue.length; i++) {
      const current = queue[i];
      for (const dirName of DIR_NAMES) {
        const next = adjacentTile(current, dirName);
        if (!next) continue;
        const key = tileKey(next);
        if (visited.has(key)) continue;

        const distance = current.distance + 1;
        if (wanted.has(grid[next.r]?.[next.c])) {
          return { tile: next, distance };
        }

        visited.add(key);
        queue.push({ ...next, distance });
      }
    }

    return { tile: null, distance: Infinity };
  }

  /**
   * Measures distance to the nearest active ghost.
   *
   * @param {{c: number, r: number}} origin Starting tile.
   * @returns {number} Route distance to the nearest dangerous ghost.
   */
  function nearestActiveGhostDistance(origin) {
    const distances = ghosts
      .filter(g => !g.eaten)
      .map(g => routeDistance(origin, g));
    return distances.length ? Math.min(...distances) : ROWS + COLS;
  }

  /**
   * Encodes one ghost as Pac-Man policy features.
   *
   * @param {{c: number, r: number, eaten: boolean} | null} ghost Ghost to encode.
   * @param {number | null} knownDistance Previously measured route distance.
   * @returns {number[]} Fixed-size ghost feature vector.
   */
  function pacmanGhostVector(ghost, knownDistance = null) {
    if (!ghost) return [0, 0, 1, 0, 0];

    const distance = knownDistance ?? routeDistance(player, ghost);
    const edible = frightenedTicks > 0 && !ghost.eaten ? 1 : 0;
    const dangerous = frightenedTicks <= 0 && !ghost.eaten ? 1 : 0;
    return [
      clamp((ghost.c - player.c) / COLS, -1, 1),
      clamp((ghost.r - player.r) / ROWS, -1, 1),
      clamp(distance / (ROWS + COLS), 0, 1),
      dangerous,
      edible
    ];
  }

  /**
   * Builds Pac-Man's automatic-play state vector.
   *
   * @returns {number[]} Gameplay features for Pac-Man's policy network.
   */
  function pacmanStateVector() {
    const nearestGhosts = ghosts
      .map(ghost => ({ ghost, distance: routeDistance(player, ghost) }))
      .sort((a, b) => a.distance - b.distance)
      .slice(0, 4);
    while (nearestGhosts.length < 4) nearestGhosts.push({ ghost: null, distance: null });

    const nearestPellet = nearestGridTarget(player, [".", "o"]);
    const nearestPowerPellet = nearestGridTarget(player, ["o"]);
    const pelletVector = nearestPellet.tile
      ? [(nearestPellet.tile.c - player.c) / COLS, (nearestPellet.tile.r - player.r) / ROWS]
      : [0, 0];
    const powerVector = nearestPowerPellet.tile
      ? [(nearestPowerPellet.tile.c - player.c) / COLS, (nearestPowerPellet.tile.r - player.r) / ROWS]
      : [0, 0];

    return [
      player.c / (COLS - 1),
      player.r / (ROWS - 1),
      ...directionVector(player.dir),
      ...openDirectionVector(player),
      clamp(frightenedTicks / BASE_FRIGHTENED_TICKS, 0, 1),
      difficultyPressure() / 10,
      ...nearestGhosts.flatMap(({ ghost, distance }) => pacmanGhostVector(ghost, distance)),
      ...pelletVector,
      ...powerVector
    ];
  }

  /**
   * Scores one possible Pac-Man move with a simple safety/food heuristic.
   *
   * @param {string} dirName Direction to evaluate.
   * @returns {number} Heuristic action score.
   */
  function pacmanHeuristicScore(dirName) {
    const next = adjacentTile(player, dirName);
    if (!next) return -999;

    const ch = grid[next.r]?.[next.c];
    const ghostDistance = nearestActiveGhostDistance(next);
    const pelletDistance = nearestGridTarget(next, [".", "o"]).distance;
    const escapeRoutes = validDirs(next).length;
    let score = 0;
    if (ch === ".") score += 2;
    if (ch === "o") score += 5;
    score += escapeRoutes * 0.14;
    score += clamp(ghostDistance, 0, 12) * (frightenedTicks > 0 ? -0.06 : 0.18);
    score -= Number.isFinite(pelletDistance) ? pelletDistance * 0.035 : 0;
    if (frightenedTicks <= 0) {
      if (ghostDistance <= 1) score -= 14;
      else if (ghostDistance <= 2) score -= 7;
      else if (ghostDistance <= 3) score -= 3.5;
      if (escapeRoutes <= 1 && ghostDistance <= 6) score -= 3;
      if (ch === "o" && ghostDistance <= 8) score += 3.5;
    } else if (ghostDistance <= 8) {
      score += (8 - ghostDistance) * 0.28;
    }
    if (dirName === REVERSE[player.dir]) score -= 0.2;
    return score;
  }

  /**
   * Chooses Pac-Man's automatic action from the policy network and heuristic.
   *
   * @returns {{dir: string, vector: number[], values: number[], epsilon: number}} Chosen action details.
   */
  function choosePacmanAutoAction() {
    const vector = pacmanStateVector();
    const valid = validDirs(player);
    const fallback = valid[0] || player.dir;
    const pass = pacmanForward(vector);
    const values = pass?.values || [0, 0, 0, 0];
    const epsilon = aiEnabled("pacmanNn") ? pacmanExplorationRate() : 0;

    let dir = fallback;
    if (aiEnabled("pacmanNn") && Math.random() < epsilon && valid.length) {
      dir = valid[Math.floor(Math.random() * valid.length)];
    } else {
      const pretrainReadiness = Math.min(1, (pacmanNetwork?.pretrainSamples || 0) / PAC_NN_PRETRAIN_TARGET);
      const heuristicWeight = aiEnabled("pacmanNn")
        ? 0.35 + (1 - pretrainReadiness) * 0.85
        : 1.4;
      dir = valid
        .map(dirName => ({
          dirName,
          score: values[DIR_NAMES.indexOf(dirName)] + pacmanHeuristicScore(dirName) * heuristicWeight
        }))
        .sort((a, b) => b.score - a.score)[0]?.dirName || fallback;
    }

    return { dir, vector, values, epsilon };
  }

  /**
   * Captures gameplay state for Pac-Man reward calculation.
   *
   * @returns {object} Snapshot of key reward signals.
   */
  function pacmanRewardSnapshot() {
    return {
      score,
      lives,
      level,
      pelletsLeft,
      c: player.c,
      r: player.r,
      nearestGhostDistance: nearestActiveGhostDistance(player),
      nearestPelletDistance: nearestGridTarget(player, [".", "o"]).distance,
      frightened: frightenedTicks > 0
    };
  }

  /**
   * Computes reinforcement reward for Pac-Man's last automatic action.
   *
   * @param {object} before Snapshot before the action.
   * @param {boolean} moved Whether Pac-Man moved successfully.
   * @returns {number} Reward signal.
   */
  function pacmanReward(before, moved) {
    const after = pacmanRewardSnapshot();
    let reward = 0.02;
    reward += (after.score - before.score) * 0.04;
    if (after.level > before.level) reward += 6;
    if (after.lives < before.lives) reward -= state === "gameover" ? 14 : 9;
    if (!moved) reward -= 0.12;

    const ghostDelta = after.nearestGhostDistance - before.nearestGhostDistance;
    reward += before.frightened ? clamp(-ghostDelta, -6, 6) * 0.04 : clamp(ghostDelta, -6, 6) * 0.08;

    const pelletDelta = before.nearestPelletDistance - after.nearestPelletDistance;
    if (Number.isFinite(pelletDelta)) reward += clamp(pelletDelta, -8, 8) * 0.03;
    return clamp(reward, -16, 16);
  }

  /**
   * Returns the direction Pac-Man is likely trying to travel.
   *
   * @returns {string} Direction key from DIRS.
   */
  function playerTravelDirection() {
    if (canMove(player, player.nextDir)) return player.nextDir;
    return player.dir;
  }

  /**
   * Computes the current difficulty pressure from level, survival, and learning depth.
   *
   * @returns {number} Adaptive pressure score.
   */
  function difficultyPressure() {
    const levelPressure = aiEnabled("levelScaling") ? Math.max(0, level - 1) : 0;
    const survivalPressure = aiEnabled("adaptiveDifficulty") ? Math.floor(survivalTicks / 90) : 0;
    const learningPressure = aiEnabled("adaptiveDifficulty") ? Math.floor(playerTrail.length / 45) : 0;
    return Math.min(10, levelPressure + survivalPressure + learningPressure);
  }

  /**
   * Chooses how far ahead the ghosts should predict Pac-Man.
   *
   * @returns {number} Number of future tiles to simulate.
   */
  function predictionHorizon() {
    return Math.min(24, PATTERN_PREDICTION_STEPS + difficultyPressure() + Math.floor(playerTrail.length / 35));
  }

  /**
   * Computes how long power pellets frighten ghosts at this difficulty.
   *
   * @returns {number} Frightened mode duration in ticks.
   */
  function frightenedDuration() {
    if (!aiEnabled("powerPelletNerf")) return BASE_FRIGHTENED_TICKS;
    const levelNerf = aiEnabled("levelScaling") ? Math.max(0, level - 1) * 4 : 0;
    return Math.max(MIN_FRIGHTENED_TICKS, BASE_FRIGHTENED_TICKS - difficultyPressure() * 5 - levelNerf);
  }

  /**
   * Scales cluster prediction influence as ghosts build confidence.
   *
   * @returns {number} Cluster prediction weight.
   */
  function clusterPredictionWeight() {
    return 1.15 + difficultyPressure() * 0.18;
  }

  /**
   * Scales neural prediction influence as training examples accumulate.
   *
   * @returns {number} Neural prediction weight.
   */
  function neuralPredictionWeight() {
    const trainingConfidence = Math.min(1.4, (neuralNetwork?.samples || 0) / 80);
    return 0.75 + difficultyPressure() * 0.1 + trainingConfidence;
  }

  /**
   * Projects a tile forward through open corridor spaces.
   *
   * @param {{c: number, r: number}} origin Starting tile.
   * @param {string} dirName Direction to project.
   * @param {number} steps Maximum number of tiles to project.
   * @returns {{c: number, r: number}} Projected open target tile.
   */
  function projectTile(origin, dirName, steps) {
    let target = { c: origin.c, r: origin.r };
    for (let i = 0; i < steps; i++) {
      const next = adjacentTile(target, dirName);
      if (!next) break;
      target = next;
    }
    return nearestOpenTile(target);
  }

  /**
   * Uses the nearest unsupervised movement cluster to score likely directions.
   *
   * @param {{c: number, r: number}} tile Tile being predicted from.
   * @param {string} incomingDir Direction entering the tile.
   * @returns {{left: number, right: number, up: number, down: number} | null} Cluster-derived direction weights.
   */
  function clusteredTurnWeights(tile, incomingDir) {
    if (!aiEnabled("behaviorClusters")) return null;

    const vector = playerContextVector({ c: tile.c, r: tile.r, dir: incomingDir }, incomingDir);
    const match = nearestBehaviorCluster(vector);
    const maxDistance = BEHAVIOR_CLUSTER_THRESHOLD * 1.75;
    if (!match || match.distance > maxDistance) return null;

    const confidence = (maxDistance - match.distance) / maxDistance;
    const weights = { left: 0, right: 0, up: 0, down: 0 };
    DIR_NAMES.forEach(dirName => {
      weights[dirName] = match.cluster.outcomes[dirName] * (1 + confidence);
    });
    return weights;
  }

  /**
   * Uses the neural network to score likely player directions.
   *
   * @param {{c: number, r: number}} tile Tile being predicted from.
   * @param {string} incomingDir Direction entering the tile.
   * @returns {{left: number, right: number, up: number, down: number} | null} Neural-network direction weights.
   */
  function neuralTurnWeights(tile, incomingDir) {
    if (!aiEnabled("neuralNetwork") || !neuralNetwork || neuralNetwork.samples < 4) return null;

    const vector = playerContextVector({ c: tile.c, r: tile.r, dir: incomingDir }, incomingDir);
    const prediction = neuralForward(vector);
    if (!prediction) return null;

    const sampleConfidence = Math.min(1, (neuralNetwork.samples - 3) / 30);
    const weights = { left: 0, right: 0, up: 0, down: 0 };
    DIR_NAMES.forEach((dirName, i) => {
      weights[dirName] = prediction.probabilities[i] * sampleConfidence;
    });
    return weights;
  }

  /**
   * Combines exact memory, behavior clusters, and neural predictions into direction weights.
   *
   * @param {{c: number, r: number}} tile Tile being predicted from.
   * @param {string} incomingDir Direction entering the tile.
   * @returns {{left: number, right: number, up: number, down: number}} Learned direction weights.
   */
  function learnedTurnWeights(tile, incomingDir) {
    const weights = { left: 0, right: 0, up: 0, down: 0 };
    const counts = aiEnabled("patternMemory") ? learnedTurns.get(turnKey(tile, incomingDir)) : null;
    if (counts) {
      DIR_NAMES.forEach(dirName => {
        weights[dirName] += counts[dirName];
      });
    }

    const clusterWeights = clusteredTurnWeights(tile, incomingDir);
    if (clusterWeights) {
      DIR_NAMES.forEach(dirName => {
        weights[dirName] += clusterWeights[dirName] * clusterPredictionWeight();
      });
    }

    const neuralWeights = neuralTurnWeights(tile, incomingDir);
    if (neuralWeights) {
      DIR_NAMES.forEach(dirName => {
        weights[dirName] += neuralWeights[dirName] * neuralPredictionWeight();
      });
    }

    return weights;
  }

  /**
   * Measures how strongly the model prefers one likely player direction.
   *
   * @param {{c: number, r: number}} tile Tile being predicted from.
   * @param {string} incomingDir Direction entering the tile.
   * @returns {number} Confidence from 0 to 1.
   */
  function learnedTurnConfidence(tile, incomingDir) {
    const weights = learnedTurnWeights(tile, incomingDir);
    const openWeights = DIR_NAMES
      .filter(dirName => canMove(tile, dirName))
      .map(dirName => weights[dirName]);
    const total = openWeights.reduce((sum, weight) => sum + weight, 0);
    return total > 0 ? Math.max(...openWeights) / total : 0;
  }

  /**
   * Picks the player's most likely turn from the hybrid learned model.
   *
   * @param {{c: number, r: number}} tile Tile being predicted from.
   * @param {string} incomingDir Direction entering the tile.
   * @returns {string | null} Most likely learned direction, if known.
   */
  function learnedTurnDirection(tile, incomingDir) {
    const weights = learnedTurnWeights(tile, incomingDir);
    const choices = DIR_NAMES
      .filter(dirName => weights[dirName] > 0 && canMove(tile, dirName))
      .sort((a, b) => weights[b] - weights[a]);
    return choices[0] || null;
  }

  /**
   * Predicts where the player is likely to be after following learned habits.
   *
   * @param {number} steps Number of future tiles to simulate.
   * @returns {{c: number, r: number, dir: string}} Predicted player tile and direction.
   */
  function predictPlayerTarget(steps) {
    let prediction = {
      c: player.c,
      r: player.r,
      dir: playerTravelDirection()
    };

    for (let i = 0; i < steps; i++) {
      let dirName = learnedTurnDirection(prediction, prediction.dir);
      if (!dirName && canMove(prediction, prediction.dir)) {
        dirName = prediction.dir;
      }
      if (!dirName) {
        const options = validDirs(prediction).filter(d => d !== REVERSE[prediction.dir]);
        dirName = options[0] || validDirs(prediction)[0];
      }
      if (!dirName) break;

      const next = adjacentTile(prediction, dirName);
      if (!next) break;
      prediction = { ...next, dir: dirName };
    }

    return prediction;
  }

  /**
   * Lists uneaten power pellet tiles.
   *
   * @returns {{c: number, r: number}[]} Remaining power pellet tiles.
   */
  function remainingPowerPellets() {
    const pellets = [];
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        if (grid[r][c] === "o") pellets.push({ c, r });
      }
    }
    return pellets;
  }

  /**
   * Chooses a power pellet to deny from Pac-Man.
   *
   * @param {{c: number, r: number}} ghost Ghost choosing a guard point.
   * @returns {{c: number, r: number} | null} Guard target or null if no power pellet remains.
   */
  function powerPelletGuardTarget(ghost) {
    const pellets = remainingPowerPellets();
    if (!pellets.length) return null;

    return pellets
      .map(pellet => ({
        pellet,
        score: routeDistance(player, pellet) * 1.4 + routeDistance(ghost, pellet)
      }))
      .sort((a, b) => a.score - b.score)[0].pellet;
  }

  /**
   * Predicts an exit point when Pac-Man is repeating a route loop.
   *
   * @returns {{c: number, r: number} | null} Intercept tile for a repeated loop.
   */
  function loopTrapTarget() {
    if (playerTrail.length < 18) return null;

    const latest = playerTrail[playerTrail.length - 1];
    for (let i = playerTrail.length - 9; i >= 0; i--) {
      const sample = playerTrail[i];
      if (sample.c !== latest.c || sample.r !== latest.r || sample.dir !== latest.dir) continue;

      const exitIndex = Math.min(playerTrail.length - 1, i + 5 + Math.floor(difficultyPressure() / 2));
      const exit = playerTrail[exitIndex];
      return projectTile(exit, exit.dir, 2 + Math.floor(difficultyPressure() / 3));
    }

    return null;
  }

  /**
   * Chooses the opposite tunnel exit when Pac-Man is near or heading into a tunnel.
   *
   * @returns {{c: number, r: number} | null} Tunnel cutoff target.
   */
  function tunnelTrapTarget() {
    const leftExit = { c: 0, r: TUNNEL_ROW };
    const rightExit = { c: COLS - 1, r: TUNNEL_ROW };
    const travelDir = playerTravelDirection();
    const predicted = predictPlayerTarget(5);
    const playerNearTunnel = Math.min(routeDistance(player, leftExit), routeDistance(player, rightExit)) <= 8;
    const predictionNearTunnel = predicted.r === TUNNEL_ROW && (predicted.c <= 5 || predicted.c >= COLS - 6);
    if (!playerNearTunnel && !predictionNearTunnel) return null;

    if (player.c < COLS / 2 || travelDir === "left") return rightExit;
    return leftExit;
  }

  /**
   * Returns the two directions perpendicular to a travel direction.
   *
   * @param {string} dirName Direction key from DIRS.
   * @returns {string[]} Perpendicular direction keys.
   */
  function perpendicularDirs(dirName) {
    return dirName === "left" || dirName === "right" ? ["up", "down"] : ["left", "right"];
  }

  /**
   * Assigns a dynamic team tactic based on proximity to the player.
   *
   * @param {{c: number, r: number, eaten: boolean}} ghost Ghost to classify.
   * @returns {string} Tactical role name.
   */
  function ghostTactic(ghost) {
    if (!aiEnabled("groupTactics")) return "chaser";

    const hunters = ghosts
      .filter(g => !g.eaten)
      .map(g => ({ ghost: g, distance: routeDistance(g, player) }))
      .sort((a, b) => a.distance - b.distance);
    const slot = Math.max(0, hunters.findIndex(entry => entry.ghost === ghost));
    return GHOST_TACTICS[slot % GHOST_TACTICS.length];
  }

  /**
   * Selects a coordinated target tile for a hunting ghost.
   *
   * @param {{c: number, r: number, id: number}} ghost Ghost choosing a target.
   * @returns {{c: number, r: number}} Tactical target tile.
   */
  function coordinatedGhostTarget(ghost) {
    const travelDir = playerTravelDirection();
    const tactic = ghostTactic(ghost);
    const learnedSteps = predictionHorizon();
    const nearPrediction = predictPlayerTarget(Math.max(3, Math.floor(learnedSteps / 2)));
    const farPrediction = predictPlayerTarget(learnedSteps);

    if (tactic === "chaser") {
      return { c: player.c, r: player.r };
    }

    if (tactic === "ambusher") {
      return nearestOpenTile(farPrediction);
    }

    if (tactic === "guardian") {
      const guardTarget = powerPelletGuardTarget(ghost);
      if (guardTarget && routeDistance(player, guardTarget) <= 12 + difficultyPressure()) {
        return guardTarget;
      }

      const sideDirs = perpendicularDirs(travelDir);
      const sideDir = sideDirs[ghost.id % sideDirs.length];
      return projectTile(nearPrediction, sideDir, 5 + Math.floor(difficultyPressure() / 3));
    }

    const loopTarget = aiEnabled("antiLoopTraps") ? loopTrapTarget() : null;
    if (loopTarget) return loopTarget;

    const tunnelTarget = aiEnabled("tunnelTraps") ? tunnelTrapTarget() : null;
    if (tunnelTarget) return tunnelTarget;

    return projectTile(farPrediction, REVERSE[farPrediction.dir || travelDir], 4 + Math.floor(difficultyPressure() / 2));
  }

  /**
   * Checks whether another active ghost already occupies a tile.
   *
   * @param {{c: number, r: number}} tile Tile to inspect.
   * @param {{c: number, r: number}} ghost Ghost that is allowed to occupy it.
   * @returns {boolean} True when another ghost is already there.
   */
  function occupiedByOtherGhost(tile, ghost) {
    return ghosts.some(g => g !== ghost && !g.eaten && g.c === tile.c && g.r === tile.r);
  }

  /**
   * Chooses the shortest path direction toward a target, with team spacing.
   *
   * @param {{c: number, r: number, dir: string}} ghost Ghost to steer.
   * @param {{c: number, r: number}} target Target tile.
   * @returns {string} Best direction key from DIRS.
   */
  function shortestPathDirection(ghost, target) {
    let options = validDirs(ghost);
    if (!options.length) return ghost.dir;

    const nonReverse = options.filter(d => d !== REVERSE[ghost.dir]);
    if (nonReverse.length) options = nonReverse;

    return options
      .map(dirName => {
        const next = adjacentTile(ghost, dirName);
        let score = next ? routeDistance(next, target) : Infinity;
        if (!Number.isFinite(score) && next) {
          score = 1000 + Math.hypot(next.c - target.c, next.r - target.r);
        }
        if (next && occupiedByOtherGhost(next, ghost)) score += 5;
        if (dirName === ghost.dir) score -= 0.25;
        if (dirName === REVERSE[ghost.dir]) score += 1.5;
        return { dirName, score };
      })
      .sort((a, b) => a.score - b.score)[0].dirName;
  }

  /**
   * Chooses a direction that carries a frightened ghost away from the player.
   *
   * @param {{c: number, r: number, dir: string}} ghost Ghost to steer.
   * @returns {string} Best escape direction key from DIRS.
   */
  function fleePlayerDirection(ghost) {
    let options = validDirs(ghost);
    if (!options.length) return ghost.dir;

    const nonReverse = options.filter(d => d !== REVERSE[ghost.dir]);
    if (nonReverse.length) options = nonReverse;

    return options
      .map(dirName => {
        const next = adjacentTile(ghost, dirName);
        let score = next ? routeDistance(next, player) : -Infinity;
        if (!Number.isFinite(score) && next) {
          score = Math.hypot(next.c - player.c, next.r - player.r);
        }
        if (dirName === ghost.dir) score += 0.25;
        return { dirName, score };
      })
      .sort((a, b) => b.score - a.score)[0].dirName;
  }

  /**
   * Chooses the player's active direction from queued input.
   *
   * @returns {string} Direction the player should try this tick.
   */
  function choosePlayerDirection() {
    if (canMove(player, player.nextDir)) return player.nextDir;
    return player.dir;
  }

  /**
   * Chooses a ghost direction using pathfinding and group tactics.
   *
   * @param {{c: number, r: number, dir: string, startC: number, startR: number, eaten: boolean, id: number}} ghost Ghost to steer.
   * @returns {string} Direction the ghost should try this tick.
   */
  function chooseGhostDirection(ghost) {
    if (frightenedTicks > 0 && !ghost.eaten) {
      return fleePlayerDirection(ghost);
    }

    const target = ghost.eaten
      ? { c: ghost.startC, r: ghost.startR }
      : (aiEnabled("groupTactics") ? coordinatedGhostTarget(ghost) : player);
    return shortestPathDirection(ghost, target);
  }

  /**
   * Decides whether ghosts move on this tick at the current pressure.
   *
   * @returns {boolean} True when ghosts should advance.
   */
  function shouldMoveGhostsThisTick() {
    const pressure = difficultyPressure();
    const skipEvery = pressure < 2 ? 3 : pressure < 5 ? 4 : 0;
    return skipEvery === 0 || tickCounter % skipEvery !== 0;
  }

  /**
   * Chooses how many ghost move passes to run this tick.
   *
   * @returns {number} Ghost move passes for this tick.
   */
  function ghostMovePasses() {
    if (!shouldMoveGhostsThisTick()) return 0;

    const pressure = difficultyPressure();
    const confidence = learnedTurnConfidence(player, playerTravelDirection());
    let passes = 1;
    if (!aiEnabled("speedBursts")) return passes;

    if (pressure >= 2 && confidence >= 0.55 && tickCounter % Math.max(3, 9 - pressure) === 0) passes++;
    if (pressure >= 7 && tickCounter % 5 === 0) passes++;
    return Math.min(3, passes);
  }

  /**
   * Moves ghosts using adaptive speed and burst passes.
   */
  function moveGhosts() {
    const passes = ghostMovePasses();
    for (let pass = 0; pass < passes; pass++) {
      ghosts.forEach(g => {
        g.dir = chooseGhostDirection(g);
        moveOneTile(g, g.dir);
        if (g.eaten && g.c === g.startC && g.r === g.startR) g.eaten = false;
      });
      handleCollisions();
      if (state !== "playing" || readyTicks > 0) return;
    }
  }

  /**
   * Consumes the pellet under the player and advances level state if needed.
   */
  function eatPellet() {
    const ch = grid[player.r]?.[player.c];
    if (ch === ".") {
      grid[player.r][player.c] = " ";
      score += 10;
      pelletsLeft--;
    } else if (ch === "o") {
      grid[player.r][player.c] = " ";
      score += 50;
      pelletsLeft--;
      frightenedTicks = frightenedDuration();
    }
    updateHud();
    if (pelletsLeft <= 0) {
      level++;
      readyTicks = READY_TICKS;
      parseMaze();
      updateHud();
    }
  }

  /**
   * Returns the player and ghosts to their starting tiles after a lost life.
   */
  function resetPositionsAfterHit() {
    player.c = 13;
    player.r = 23;
    player.dir = "left";
    player.nextDir = "left";
    ghosts.forEach(g => {
      g.c = g.startC;
      g.r = g.startR;
      g.dir = "left";
      g.eaten = false;
    });
    readyTicks = READY_TICKS;
    frightenedTicks = 0;
    survivalTicks = 0;
  }

  /**
   * Resolves player and ghost collisions, including frightened ghost scoring.
   */
  function handleCollisions() {
    for (const ghost of ghosts) {
      if (ghost.c === player.c && ghost.r === player.r && !ghost.eaten) {
        if (frightenedTicks > 0) {
          ghost.eaten = true;
          score += 200;
          updateHud();
        } else {
          lives--;
          updateHud();
          if (lives <= 0) {
            state = "gameover";
            saveLearning();
          } else {
            resetPositionsAfterHit();
          }
        }
      }
    }
  }

  /**
   * Advances the game by one timed step and redraws the board.
   */
  function stepGame() {
    if (state !== "playing") return;
    tickCounter++;

    if (readyTicks > 0) {
      draw();
      readyTicks--;
      return;
    }

    if (frightenedTicks > 0) frightenedTicks--;
    survivalTicks++;

    const autoAction = aiEnabled("autoPacman") ? choosePacmanAutoAction() : null;
    const autoRewardSnapshot = autoAction ? pacmanRewardSnapshot() : null;
    if (autoAction) player.nextDir = autoAction.dir;

    const previousPlayer = { c: player.c, r: player.r, dir: player.dir };
    player.dir = choosePlayerDirection();
    const playerMoved = moveOneTile(player, player.dir);
    if (playerMoved) learnPlayerMove(previousPlayer);
    player.mouthOpen = !player.mouthOpen;
    eatPellet();
    handleCollisions();

    if (state === "playing" && readyTicks <= 0) moveGhosts();
    if (autoAction && aiEnabled("pacmanNn")) {
      const lifeLost = autoRewardSnapshot.lives > lives;
      trainPacmanNetwork(
        autoAction.vector,
        autoAction.dir,
        pacmanReward(autoRewardSnapshot, playerMoved),
        pacmanStateVector(),
        lifeLost || state !== "playing"
      );
    }
    draw();
  }

  /**
   * Converts grid coordinates into canvas center-point coordinates.
   *
   * @param {number} c Column index.
   * @param {number} r Row index.
   * @returns {{x: number, y: number}} Canvas pixel coordinates.
   */
  function xy(c, r) {
    return { x: c * TILE + TILE / 2, y: r * TILE + TILE / 2 };
  }

  /**
   * Converts the remaining ready ticks into a visible countdown number.
   *
   * @returns {number} Seconds left in the ready countdown.
   */
  function readyCountdownSeconds() {
    if (readyTicks <= 0) return 0;
    return Math.max(1, Math.ceil((readyTicks / READY_TICKS) * READY_SECONDS));
  }

  /**
   * Draws walls, pellets, and power pellets on the canvas.
   */
  function drawMaze() {
    ctx.fillStyle = "#02030a";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const ch = grid[r][c];
        const x = c * TILE;
        const y = r * TILE;
        if (ch === "#") {
          ctx.fillStyle = "#151be0";
          ctx.fillRect(x + 1, y + 1, TILE - 2, TILE - 2);
          ctx.strokeStyle = "#4f62ff";
          ctx.strokeRect(x + 4, y + 4, TILE - 8, TILE - 8);
        } else if (ch === ".") {
          ctx.fillStyle = "#ffd9b3";
          ctx.beginPath();
          ctx.arc(x + TILE / 2, y + TILE / 2, 3, 0, Math.PI * 2);
          ctx.fill();
        } else if (ch === "o") {
          ctx.fillStyle = "#fff2a6";
          ctx.beginPath();
          ctx.arc(x + TILE / 2, y + TILE / 2, 7, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
  }

  /**
   * Draws the player sprite using the current direction and mouth state.
   */
  function drawPlayer() {
    const p = xy(player.c, player.r);
    const angle = { right: 0, down: Math.PI / 2, left: Math.PI, up: Math.PI * 1.5 }[player.dir] || 0;
    const mouth = player.mouthOpen ? 0.65 : 0.15;
    ctx.fillStyle = "#ffe74a";
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
    ctx.arc(p.x, p.y, TILE * 0.43, angle + mouth, angle + Math.PI * 2 - mouth);
    ctx.closePath();
    ctx.fill();
  }

  /**
   * Draws one ghost with normal, frightened, or eaten styling.
   *
   * @param {{c: number, r: number, color: string, eaten: boolean}} ghost Ghost to draw.
   */
  function drawGhost(ghost) {
    const p = xy(ghost.c, ghost.r);
    const frightened = frightenedTicks > 0 && !ghost.eaten;
    ctx.fillStyle = ghost.eaten ? "rgba(255,255,255,0.35)" : frightened ? "#2f58ff" : ghost.color;
    ctx.beginPath();
    ctx.arc(p.x, p.y - 2, TILE * 0.42, Math.PI, 0);
    ctx.lineTo(p.x + TILE * 0.42, p.y + TILE * 0.35);
    for (let i = 2; i >= -2; i--) {
      ctx.lineTo(p.x + i * TILE * 0.17, p.y + (i % 2 ? TILE * 0.22 : TILE * 0.35));
    }
    ctx.closePath();
    ctx.fill();

    ctx.fillStyle = "white";
    ctx.beginPath();
    ctx.arc(p.x - 5, p.y - 4, 4, 0, Math.PI * 2);
    ctx.arc(p.x + 5, p.y - 4, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#111";
    ctx.beginPath();
    ctx.arc(p.x - 4, p.y - 4, 2, 0, Math.PI * 2);
    ctx.arc(p.x + 6, p.y - 4, 2, 0, Math.PI * 2);
    ctx.fill();
  }

  /**
   * Draws a centered status overlay above the game board.
   *
   * @param {string} text Main overlay message.
   * @param {string} [subtext=""] Optional secondary message.
   */
  function drawOverlay(text, subtext = "") {
    ctx.save();
    ctx.fillStyle = "rgba(0, 0, 0, 0.68)";
    ctx.fillRect(0, canvas.height / 2 - 62, canvas.width, 124);
    ctx.fillStyle = "#ffe74a";
    ctx.textAlign = "center";
    ctx.font = "bold 36px system-ui, sans-serif";
    ctx.fillText(text, canvas.width / 2, canvas.height / 2 - 6);
    if (subtext) {
      ctx.fillStyle = "#fff";
      ctx.font = "18px system-ui, sans-serif";
      ctx.fillText(subtext, canvas.width / 2, canvas.height / 2 + 30);
    }
    ctx.restore();
  }

  /**
   * Redraws the complete current game frame.
   */
  function draw() {
    updateAiStats();
    drawMaze();
    drawPlayer();
    ghosts.forEach(drawGhost);
    if (state === "gameover") drawOverlay("Game over", "Press Restart or Space");
    else if (readyTicks > 0) {
      const seconds = readyCountdownSeconds();
      const label = seconds === 1 ? "1 second" : `${seconds} seconds`;
      drawOverlay("Ready!", `Starting in ${label}`);
    }
  }

  /**
   * Starts or restarts the fixed-interval game loop.
   */
  function startLoop() {
    if (loopId) clearInterval(loopId);
    loopId = setInterval(stepGame, STEP_MS);
  }

  document.addEventListener("keydown", (event) => {
    const dir = KEY_TO_DIR[event.code];
    if (dir) {
      event.preventDefault();
      player.nextDir = dir;
      // Also advance one frame immediately on key press. This makes controls work
      // even in browsers/extensions that throttle timers until a repaint event.
      if (readyTicks <= 0 && state === "playing") stepGame();
      else draw();
    }
    if (event.code === "Space" && state === "gameover") resetGame();
  }, { passive: false });

  let touchStart = null;
  canvas.addEventListener("pointerdown", (event) => {
    touchStart = { x: event.clientX, y: event.clientY };
    canvas.setPointerCapture?.(event.pointerId);
  });
  canvas.addEventListener("pointerup", (event) => {
    if (!touchStart) return;
    const dx = event.clientX - touchStart.x;
    const dy = event.clientY - touchStart.y;
    if (Math.hypot(dx, dy) > 18) {
      player.nextDir = Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? "right" : "left") : (dy > 0 ? "down" : "up");
      if (readyTicks <= 0 && state === "playing") stepGame();
    }
    touchStart = null;
  });

  aiToggleBtns.forEach(button => {
    button.addEventListener("click", () => {
      const name = button.dataset.aiToggle;
      setAiSetting(name, !aiEnabled(name));
    });
  });
  classicAiBtn?.addEventListener("click", () => setAllAiSettings(false));
  allAiBtn?.addEventListener("click", () => setAllAiSettings(true));
  pretrainPacmanBtn?.addEventListener("click", () => {
    if (isPacmanPretraining()) {
      stopPacmanPretraining();
    } else {
      pretrainPacmanPolicy(PAC_NN_PRETRAIN_BATCH);
    }
    draw();
  });
  resetAiMemoryBtn?.addEventListener("click", () => {
    clearAiMemory();
    draw();
  });
  restartBtn.addEventListener("click", resetGame);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      saveLearning();
    } else {
      startLoop();
      draw();
    }
  });
  window.addEventListener("beforeunload", saveLearning);

  loadAiSettings();
  updateAiControls();
  resetGame();
})();
