(() => {
  "use strict";

  const canvas = document.getElementById("game");
  const ctx = canvas.getContext("2d", { alpha: false });
  const scoreEl = document.getElementById("score");
  const livesEl = document.getElementById("lives");
  const levelEl = document.getElementById("level");
  const restartBtn = document.getElementById("restart");

  const TILE = 24;
  const ROWS = 31;
  const COLS = 28;
  const STEP_MS = 115;
  const READY_SECONDS = 5;
  const READY_TICKS = Math.ceil((READY_SECONDS * 1000) / STEP_MS);
  const GHOST_TACTICS = ["chaser", "ambusher", "flanker", "trapper"];
  const PLAYER_MEMORY_LIMIT = 140;
  const PATTERN_PREDICTION_STEPS = 8;
  const LEARNING_DECAY = 0.985;
  const MIN_LEARNED_WEIGHT = 0.05;
  const BEHAVIOR_CLUSTER_LIMIT = 10;
  const BEHAVIOR_CLUSTER_RATE = 0.18;
  const BEHAVIOR_CLUSTER_DECAY = 0.992;
  const BEHAVIOR_CLUSTER_THRESHOLD = 1.1;

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
    "     #.## ###GG### ##.#     ",
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

  let grid, score, lives, level, pelletsLeft, player, ghosts, frightenedTicks, state, readyTicks;
  let playerTrail, learnedTurns, behaviorClusters;
  let loopId = null;
  let tickCounter = 0;

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
  }

  /**
   * Resets score, lives, level state, actors, and starts a fresh game loop.
   */
  function resetGame() {
    score = 0;
    lives = 3;
    level = 1;
    frightenedTicks = 0;
    readyTicks = READY_TICKS;
    state = "playing";
    tickCounter = 0;
    resetLearning();
    parseMaze();
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
    decayLearnedTurns();
    decayBehaviorClusters();
    const behaviorVector = playerContextVector(previousPlayer, player.nextDir);

    playerTrail.push({ c: player.c, r: player.r, dir: player.dir });
    if (playerTrail.length > PLAYER_MEMORY_LIMIT) playerTrail.shift();

    const key = turnKey(previousPlayer, previousPlayer.dir);
    const counts = learnedTurns.get(key) || { left: 0, right: 0, up: 0, down: 0 };
    counts[player.dir]++;
    learnedTurns.set(key, counts);
    learnBehaviorCluster(behaviorVector, player.dir);
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
   * Measures squared distance between two feature vectors.
   *
   * @param {number[]} a First vector.
   * @param {number[]} b Second vector.
   * @returns {number} Squared Euclidean distance.
   */
  function squaredDistance(a, b) {
    return a.reduce((total, value, i) => total + (value - b[i]) ** 2, 0);
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
    while (queue.length) {
      const current = queue.shift();
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

    return { c: player.c, r: player.r };
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

    const queue = [{ ...origin, distance: 0 }];
    const visited = new Set([tileKey(origin)]);
    while (queue.length) {
      const current = queue.shift();
      for (const dirName of DIR_NAMES) {
        const next = adjacentTile(current, dirName);
        if (!next) continue;
        const key = tileKey(next);
        if (visited.has(key)) continue;
        if (key === tileKey(destination)) return current.distance + 1;
        visited.add(key);
        queue.push({ ...next, distance: current.distance + 1 });
      }
    }

    return Infinity;
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
   * Picks the player's most likely turn from exact memory and learned clusters.
   *
   * @param {{c: number, r: number}} tile Tile being predicted from.
   * @param {string} incomingDir Direction entering the tile.
   * @returns {string | null} Most likely learned direction, if known.
   */
  function learnedTurnDirection(tile, incomingDir) {
    const weights = { left: 0, right: 0, up: 0, down: 0 };
    const counts = learnedTurns.get(turnKey(tile, incomingDir));
    if (counts) {
      DIR_NAMES.forEach(dirName => {
        weights[dirName] += counts[dirName];
      });
    }

    const clusterWeights = clusteredTurnWeights(tile, incomingDir);
    if (clusterWeights) {
      DIR_NAMES.forEach(dirName => {
        weights[dirName] += clusterWeights[dirName] * 1.25;
      });
    }

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
    const hunters = ghosts
      .filter(g => !g.eaten)
      .sort((a, b) => routeDistance(a, player) - routeDistance(b, player));
    const slot = Math.max(0, hunters.indexOf(ghost));
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
    const learnedSteps = Math.min(14, PATTERN_PREDICTION_STEPS + Math.floor(playerTrail.length / 35));
    const nearPrediction = predictPlayerTarget(Math.max(3, Math.floor(learnedSteps / 2)));
    const farPrediction = predictPlayerTarget(learnedSteps);

    if (tactic === "chaser") {
      return { c: player.c, r: player.r };
    }

    if (tactic === "ambusher") {
      return nearestOpenTile(farPrediction);
    }

    if (tactic === "flanker") {
      const sideDirs = perpendicularDirs(travelDir);
      const sideDir = sideDirs[ghost.id % sideDirs.length];
      return projectTile(nearPrediction, sideDir, 5);
    }

    return projectTile(farPrediction, REVERSE[farPrediction.dir || travelDir], 4);
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
      : coordinatedGhostTarget(ghost);
    return shortestPathDirection(ghost, target);
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
      frightenedTicks = 70;
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

    const previousPlayer = { c: player.c, r: player.r, dir: player.dir };
    player.dir = choosePlayerDirection();
    const playerMoved = moveOneTile(player, player.dir);
    if (playerMoved) learnPlayerMove(previousPlayer);
    player.mouthOpen = !player.mouthOpen;
    eatPellet();

    // Move ghosts a little slower than the player on early levels.
    if (tickCounter % Math.max(2, 4 - Math.min(level, 2)) !== 0) {
      ghosts.forEach(g => {
        g.dir = chooseGhostDirection(g);
        moveOneTile(g, g.dir);
        if (g.eaten && g.c === g.startC && g.r === g.startR) g.eaten = false;
      });
    }

    handleCollisions();
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

  restartBtn.addEventListener("click", resetGame);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) {
      startLoop();
      draw();
    }
  });

  resetGame();
})();
