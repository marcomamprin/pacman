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

  const REVERSE = { left: "right", right: "left", up: "down", down: "up" };

  const KEY_TO_DIR = {
    ArrowLeft: "left", KeyA: "left",
    ArrowRight: "right", KeyD: "right",
    ArrowUp: "up", KeyW: "up",
    ArrowDown: "down", KeyS: "down"
  };

  let grid, score, lives, level, pelletsLeft, player, ghosts, frightenedTicks, state, readyTicks;
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
    readyTicks = 10;
    state = "playing";
    tickCounter = 0;
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
    return Object.keys(DIRS).filter(d => canMove(entity, d));
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
   * Chooses a ghost direction, preferring pursuit unless frightened or wandering.
   *
   * @param {{c: number, r: number, dir: string, startC: number, startR: number, eaten: boolean}} ghost Ghost to steer.
   * @returns {string} Direction the ghost should try this tick.
   */
  function chooseGhostDirection(ghost) {
    let options = validDirs(ghost).filter(d => d !== REVERSE[ghost.dir]);
    if (!options.length) options = validDirs(ghost);
    if (!options.length) return ghost.dir;

    if (frightenedTicks > 0 && !ghost.eaten) {
      return options[Math.floor(Math.random() * options.length)];
    }

    const target = ghost.eaten ? { c: ghost.startC, r: ghost.startR } : player;
    if (!ghost.eaten && Math.random() < 0.2) return options[Math.floor(Math.random() * options.length)];

    return options.sort((a, b) => {
      const da = DIRS[a], db = DIRS[b];
      const ad = Math.hypot(ghost.c + da.dc - target.c, ghost.r + da.dr - target.r);
      const bd = Math.hypot(ghost.c + db.dc - target.c, ghost.r + db.dr - target.r);
      return ad - bd;
    })[0];
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
      readyTicks = 10;
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
    readyTicks = 10;
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
      readyTicks--;
      draw();
      return;
    }

    if (frightenedTicks > 0) frightenedTicks--;

    player.dir = choosePlayerDirection();
    moveOneTile(player, player.dir);
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
    else if (readyTicks > 0) drawOverlay("Ready!", "Starts automatically");
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
