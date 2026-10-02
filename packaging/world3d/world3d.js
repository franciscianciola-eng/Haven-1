// Haven's valley in 3D, for the people keeping it company.
//
// Built from what the app says about the world (/api/world): the ground and its heights,
// and everything in the valley. Between moments, things glide to where they are now, so
// Haven walks, the ball rolls and the butterflies drift. This is only a picture of the
// world for people: Haven itself never sees it. What it senses is the world's own rays.
//
// Bundled with three.js into haven/static/world3d.js (see build.py next to this file).

import {
  ACESFilmicToneMapping,
  BoxGeometry,
  BufferGeometry,
  CanvasTexture,
  Color,
  ConeGeometry,
  CylinderGeometry,
  DirectionalLight,
  DodecahedronGeometry,
  DoubleSide,
  Float32BufferAttribute,
  Fog,
  Group,
  HemisphereLight,
  IcosahedronGeometry,
  LatheGeometry,
  LineBasicMaterial,
  LineSegments,
  Mesh,
  MeshLambertMaterial,
  MeshStandardMaterial,
  PCFShadowMap,
  PerspectiveCamera,
  PlaneGeometry,
  PointLight,
  Points,
  PointsMaterial,
  Raycaster,
  Scene,
  SphereGeometry,
  Sprite,
  SpriteMaterial,
  SRGBColorSpace,
  TorusGeometry,
  Vector2,
  Vector3,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const LEVEL = 0.6; // the height of one step of the ground (haven/world.py)
const WALL = 1.0; // walls are drawn low, so they don't hide the valley (to Haven they're tall)
const BASE = -1.4; // the bottom of the valley's block of earth
const DIRS = [[0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]];
const SKY_DAY = new Color("#9fd4f5");
const SKY_DUSK = new Color("#f2b38a");
const SKY_NIGHT = new Color("#0c1230");
// The weather the person can bring (see World.bring): the sky it gives, and how much; and how near the fog comes.
const WEATHER_SKY = { storm: ["#454c57", 0.7], snow: ["#c9d2dc", 0.5], heat: ["#ffa25c", 0.35], sun: ["#ffe2a0", 0.18] };
const WEATHER_FOG = { storm: [10, 42], snow: [7, 34], heat: [18, 58] };
const BUTTERFLY_COLORS = ["#ffffff", "#ffd54a", "#ff9b54", "#b9a7ff"];
const DROPS = 1400;  // raindrops or snowflakes, falling over the valley
// How the valley looks in each season (Haven feels the seasons, and sees the leaves turn and the flowers go).
const SEASONS = {
  spring: { leaves: ["#5aae52", "#6fbd5f"], ground: "#ffffff", frost: "#000000", water: "#3d8fd1", sky: "#9fd4f5", bush: "#2f7b3c" },
  summer: { leaves: ["#3f8f45", "#4a9d4e"], ground: "#f6f4df", frost: "#000000", water: "#3d8fd1", sky: "#93d0f7", bush: "#2f7b3c" },
  autumn: { leaves: ["#d9792b", "#c4532a"], ground: "#efdcb2", frost: "#000000", water: "#4a88bd", sky: "#b7cfe0", bush: "#5b7a3a" },
  winter: { leaves: null, ground: "#dde4ec", frost: "#2c3139", water: "#a7c5e2", sky: "#c3d3e2", bush: "#5a6e52" },
};

const mats = {};
function mat(color, options = {}) {
  const key = color + JSON.stringify(options);
  if (!mats[key]) mats[key] = new MeshLambertMaterial({ color, ...options });
  return mats[key];
}

function hash(x, y, salt = 0) {
  let h = (x * 374761393 + y * 668265263 + salt * 2147483647) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967295;
}

function mesh(geometry, material, { shadow = true, receive = false } = {}) {
  const m = new Mesh(geometry, material);
  m.castShadow = shadow;
  m.receiveShadow = receive;
  return m;
}

function label(object, text) {
  object.traverse((o) => { o.userData.label = text; });
  return object;
}

// --- the ground ---------------------------------------------------------------------------------

function groundHeight(world, x, y) {
  const rows = world.layout;
  if (y < 0 || y >= rows.length || x < 0 || x >= rows[0].length) return BASE;
  const c = rows[y][x];
  const h = Number(world.heights[y][x]) * LEVEL;
  if (c === "#") return h + (x === 0 || y === 0 || y === rows.length - 1 || x === rows[0].length - 1 ? WALL : 0.7);
  if (c === "W") return -0.32;
  return h;
}

function groundColor(world, x, y) {
  const c = world.layout[y][x];
  const v = hash(x, y) * 0.08 - 0.04;
  if (c === "#") return new Color("#6b6873").offsetHSL(0, 0, v);
  if (c === ":") return new Color("#dcc990").offsetHSL(0, 0, v * 0.6);
  if (c === "W") return new Color("#4f7f86");
  if (c === "F") return new Color("#6f6154");
  const level = Number(world.heights[y][x]);
  return new Color(level > 0 ? "#79b55f" : "#6aad57").offsetHSL(0.01 * level, 0, v - 0.015 * level);
}

function buildGround(world) {
  const rows = world.layout, n = rows.length, m = rows[0].length;
  const pos = [], col = [], nor = [];
  const quad = (a, b, c, d, color, normal) => {
    for (const p of [a, b, c, a, c, d]) {
      pos.push(...p);
      col.push(color.r, color.g, color.b);
      nor.push(...normal);
    }
  };
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < m; x++) {
      const h = groundHeight(world, x, y);
      const top = groundColor(world, x, y);
      quad([x, h, y], [x, h, y + 1], [x + 1, h, y + 1], [x + 1, h, y], top, [0, 1, 0]);
      const wall = rows[y][x] === "#";
      for (const [dx, dy, normal] of [[0, -1, [0, 0, -1]], [1, 0, [1, 0, 0]], [0, 1, [0, 0, 1]], [-1, 0, [-1, 0, 0]]]) {
        const low = groundHeight(world, x + dx, y + dy);
        if (low >= h) continue;
        const side = wall ? new Color("#56535d") : low <= BASE ? new Color("#6d5139") : new Color("#9a8a72");
        side.offsetHSL(0, 0, hash(x, y, dx + 2 * dy) * 0.06 - 0.03);
        if (dx !== 0) {
          const xs = dx === 1 ? x + 1 : x;
          const [ya, yb] = dx === 1 ? [y + 1, y] : [y, y + 1];
          quad([xs, low, ya], [xs, low, yb], [xs, h, yb], [xs, h, ya], side, normal);
        } else {
          const ys = dy === 1 ? y + 1 : y;
          const [xa, xb] = dy === 1 ? [x, x + 1] : [x + 1, x];
          quad([xa, low, ys], [xb, low, ys], [xb, h, ys], [xa, h, ys], side, normal);
        }
      }
    }
  }
  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new Float32BufferAttribute(pos, 3));
  geometry.setAttribute("color", new Float32BufferAttribute(col, 3));
  geometry.setAttribute("normal", new Float32BufferAttribute(nor, 3));
  const ground = new Mesh(geometry, new MeshLambertMaterial({ vertexColors: true }));  // (tinted by the season)
  ground.receiveShadow = true;
  ground.castShadow = true;
  return ground;
}

function buildWater(world) {
  let x0 = 99, y0 = 99, x1 = -1, y1 = -1;
  world.layout.forEach((row, y) => [...row].forEach((c, x) => {
    if (c === "W") { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
  }));
  if (x1 < 0) return null;
  const w = x1 - x0 + 1, d = y1 - y0 + 1;
  const geometry = new PlaneGeometry(w, d, w * 4, d * 4);
  geometry.rotateX(-Math.PI / 2);
  const water = new Mesh(
    geometry,
    new MeshStandardMaterial({ color: "#3d8fd1", transparent: true, opacity: 0.82, roughness: 0.18, metalness: 0.05 }),
  );
  water.position.set(x0 + w / 2, -0.07, y0 + d / 2);
  water.receiveShadow = true;
  water.userData.rest = Float32Array.from(geometry.attributes.position.array);
  return label(water, "the pond");
}

// --- things in the valley ---------------------------------------------------------------------

function tree(x, y) {
  const g = new Group();
  const trunk = mesh(new CylinderGeometry(0.1, 0.16, 1.5, 7), mat("#7a5230"));
  trunk.position.y = 0.75;
  const crown = new Group();
  const leaves = mat("#3f8f45", { flatShading: true });
  const big = mesh(new IcosahedronGeometry(0.85, 1), leaves);
  big.position.y = 1.95;
  const side = mesh(new IcosahedronGeometry(0.55, 1), mat("#4a9d4e", { flatShading: true }));
  side.position.set(0.45 * (hash(x, y) - 0.5) * 2, 1.7, 0.4);
  crown.add(big, side);
  const branches = new Group(); // what's left of it in winter
  for (let i = 0; i < 6; i++) {
    const branch = mesh(new CylinderGeometry(0.018, 0.045, 0.95, 5), mat("#6e4b2c"));
    const a = (i / 6) * Math.PI * 2 + hash(x, y, i + 30);
    branch.position.set(Math.cos(a) * 0.28, 1.72 + 0.2 * hash(x, y, i + 40), Math.sin(a) * 0.28);
    branch.rotation.set(Math.sin(a) * 0.75, 0, -Math.cos(a) * 0.75);
    branches.add(branch);
  }
  branches.visible = false;
  g.add(trunk, crown, branches);
  Object.assign(g.userData, { crown, branches, big, side });
  g.userData.apples = [0, 1, 2].map((i) => {
    const apple = mesh(new SphereGeometry(0.1, 10, 8), mat("#d8261f"));
    const a = (i / 3) * Math.PI * 2 + hash(x, y, i) * 0.8;
    apple.position.set(Math.cos(a) * 0.72, 1.62 + 0.28 * hash(x, y, i + 5), Math.sin(a) * 0.72);
    g.add(apple);
    return apple;
  });
  g.rotation.y = hash(x, y, 9) * Math.PI * 2;
  return label(g, "an apple tree");
}

function bush(x, y) {
  const g = new Group();
  const body = mesh(new IcosahedronGeometry(0.46, 1), mat("#2f7b3c", { flatShading: true }));
  body.scale.set(1, 0.82, 1);
  body.position.y = 0.4;
  g.add(body);
  g.userData.leaves = body;
  g.userData.berries = [0, 1, 2].map((i) => {
    const berry = mesh(new SphereGeometry(0.075, 8, 6), mat("#e8243c"), { shadow: false });
    const a = (i / 3) * Math.PI * 2 + hash(x, y, i) * 1.2;
    berry.position.set(Math.cos(a) * 0.4, 0.42 + 0.14 * (hash(x, y, i + 3) - 0.3), Math.sin(a) * 0.4);
    g.add(berry);
    return berry;
  });
  return label(g, "a berry bush");
}

function stone(x, y) {
  const s = mesh(new DodecahedronGeometry(0.34, 0), mat("#a3a39c", { flatShading: true }));
  s.scale.set(1.1, 0.8, 0.95);
  s.position.y = 0.24;
  s.rotation.set(hash(x, y) * 3, hash(x, y, 1) * 3, 0);
  const g = new Group();
  g.add(s);
  return label(g, "a stone");
}

function thorns(x, y) {
  const g = new Group();
  for (let i = 0; i < 7; i++) {
    const spike = mesh(new ConeGeometry(0.05, 0.45 + 0.2 * hash(x, y, i), 5), mat("#8a3bb2"));
    const a = hash(x, y, i + 10) * Math.PI * 2, r = 0.28 * hash(x, y, i + 20);
    spike.position.set(Math.cos(a) * r, 0.22, Math.sin(a) * r);
    spike.rotation.set((hash(x, y, i) - 0.5) * 0.7, 0, (hash(x, y, i + 1) - 0.5) * 0.7);
    g.add(spike);
  }
  return label(g, "thorns");
}

function nest() {
  const g = new Group();
  const ring = mesh(new TorusGeometry(0.34, 0.13, 8, 20), mat("#c9a24a", { flatShading: true }));
  ring.rotation.x = Math.PI / 2;
  ring.position.y = 0.12;
  const bed = mesh(new CylinderGeometry(0.32, 0.3, 0.06, 18), mat("#a5823a"));
  bed.position.y = 0.05;
  g.add(ring, bed);
  return label(g, "Haven's nest");
}

function flower(x, y, color) {
  const g = new Group();
  for (let i = 0; i < 3; i++) {
    const stem = mesh(new CylinderGeometry(0.012, 0.012, 0.3, 4), mat("#3e8a3a"), { shadow: false });
    const head = mesh(new SphereGeometry(0.07, 8, 6), mat(color));
    const a = (i / 3) * Math.PI * 2 + hash(x, y, i), r = i ? 0.2 : 0.02;
    const h = 0.24 + 0.1 * hash(x, y, i + 4);
    stem.position.set(Math.cos(a) * r, h / 2, Math.sin(a) * r);
    head.position.set(Math.cos(a) * r, h, Math.sin(a) * r);
    head.scale.set(1, 0.6, 1);
    g.add(stem, head);
  }
  return label(g, "flowers");
}

function mushroom(x, y, cap, spots) {
  const g = new Group();
  const stalk = mesh(new CylinderGeometry(0.05, 0.065, 0.2, 8), mat("#efe6d2"));
  stalk.position.y = 0.1;
  const top = mesh(new SphereGeometry(0.16, 12, 8, 0, Math.PI * 2, 0, Math.PI / 2), mat(cap));
  top.position.y = 0.18;
  g.add(stalk, top);
  if (spots) {
    for (let i = 0; i < 5; i++) {
      const spot = mesh(new SphereGeometry(0.025, 6, 4), mat("#ffffff"), { shadow: false });
      const a = (i / 5) * Math.PI * 2;
      spot.position.set(Math.cos(a) * 0.1, 0.29, Math.sin(a) * 0.1);
      g.add(spot);
    }
  }
  g.rotation.y = hash(x, y) * 6;
  return g;
}

function bell() {
  const g = new Group();
  const wood = mat("#7b5332");
  for (const side of [-0.42, 0.42]) {
    const post = mesh(new BoxGeometry(0.1, 1.5, 0.1), wood);
    post.position.set(side, 0.75, 0);
    g.add(post);
  }
  const beam = mesh(new BoxGeometry(1.0, 0.1, 0.12), wood);
  beam.position.y = 1.5;
  const swing = new Group();
  swing.position.y = 1.45;
  const profile = [
    [0.05, 0], [0.09, -0.06], [0.12, -0.18], [0.16, -0.32], [0.24, -0.44], [0.26, -0.48], [0, -0.48],
  ].map(([r, h]) => new Vector2(r, h));
  const cup = mesh(new LatheGeometry(profile, 20), new MeshStandardMaterial({
    color: "#d6a531", metalness: 0.7, roughness: 0.3, side: DoubleSide,
  }));
  const clapper = mesh(new SphereGeometry(0.05, 8, 6), mat("#8a6a1e"));
  clapper.position.y = -0.46;
  swing.add(cup, clapper);
  g.add(beam, swing);
  g.userData.swing = swing;
  return label(g, "the bell");
}

function fire() {
  const g = new Group();
  const logs = mat("#6a4122");
  for (const turn of [0.4, -0.4, 1.9]) {
    const log = mesh(new CylinderGeometry(0.06, 0.07, 0.62, 6), logs);
    log.rotation.set(Math.PI / 2, 0, turn);
    log.position.y = 0.06;
    g.add(log);
  }
  const ring = mesh(new TorusGeometry(0.36, 0.06, 5, 12), mat("#8d8a84", { flatShading: true }));
  ring.rotation.x = Math.PI / 2;
  ring.position.y = 0.04;
  g.add(ring);
  g.userData.flames = [["#ffb02e", 0.16, 0.5], ["#ff6a1f", 0.12, 0.38], ["#ffe07a", 0.07, 0.3]].map(([c, r, h], i) => {
    const flame = new Mesh(new ConeGeometry(r, h, 8), new MeshStandardMaterial({
      color: c, emissive: c, emissiveIntensity: 1.4, transparent: true, opacity: 0.9,
    }));
    flame.position.set((i - 1) * 0.05, 0.08 + h / 2, (i % 2) * 0.04);
    g.add(flame);
    return flame;
  });
  const light = new PointLight("#ff8a3a", 1, 7, 1.6);
  light.position.y = 0.6;
  g.add(light);
  g.userData.light = light;
  return label(g, "the campfire");
}

function wildfire(x, y) {  // grass the person set alight: flames, no logs, no light of its own (it burns out)
  const g = new Group();
  g.userData.flames = [["#ff7a1f", 0.2, 0.62], ["#ffb02e", 0.14, 0.46], ["#ffe07a", 0.08, 0.32]].map(([c, r, h], i) => {
    const flame = new Mesh(new ConeGeometry(r, h, 7), new MeshStandardMaterial({
      color: c, emissive: c, emissiveIntensity: 1.5, transparent: true, opacity: 0.85,
    }));
    flame.position.set((hash(x, y, i) - 0.5) * 0.3, h / 2, (hash(x, y, i + 3) - 0.5) * 0.3);
    g.add(flame);
    return flame;
  });
  return label(g, "a fire, spreading");
}

function ball() {
  const g = new Group();
  const b = mesh(new SphereGeometry(0.22, 18, 14), mat("#1bc6c6"));
  const stripe = mesh(new TorusGeometry(0.221, 0.03, 6, 24), mat("#f7f3ea"), { shadow: false });
  b.add(stripe);
  b.position.y = 0.22;
  g.add(b);
  g.userData.spin = b;
  return label(g, "the ball");
}

function fallenApple() {
  const g = new Group();
  const a = mesh(new SphereGeometry(0.1, 10, 8), mat("#d8261f"));
  const stem = mesh(new CylinderGeometry(0.01, 0.01, 0.06, 4), mat("#5a3a1a"), { shadow: false });
  stem.position.y = 0.11;
  a.position.y = 0.1;
  g.add(a, stem);
  return label(g, "an apple");
}

function butterfly(color) {
  const g = new Group();
  const body = mesh(new CylinderGeometry(0.012, 0.012, 0.12, 4), mat("#2a2622"), { shadow: false });
  body.rotation.x = Math.PI / 2;
  const wing = new MeshLambertMaterial({ color, side: DoubleSide });
  const shape = new PlaneGeometry(0.13, 0.11);
  shape.translate(0.068, 0, 0);
  const left = new Mesh(shape, wing), right = new Mesh(shape, wing);
  right.rotation.y = Math.PI;
  left.castShadow = right.castShadow = true;
  g.add(body, left, right);
  g.userData.wings = [left, right];
  return label(g, "a butterfly");
}

function creature() {
  const g = new Group();
  const bodyGroup = new Group();
  const fur = new MeshStandardMaterial({ color: "#f3dcc0", roughness: 0.75 });
  const body = mesh(new SphereGeometry(0.3, 24, 18), fur);
  body.scale.set(1, 0.9, 1);
  body.position.y = 0.28;
  const belly = mesh(new SphereGeometry(0.2, 16, 12), mat("#fff9f0"), { shadow: false });
  belly.position.set(0, 0.22, 0.14);
  belly.scale.set(1, 0.9, 0.6);
  const eyes = [-0.1, 0.1].map((side) => {
    const eye = mesh(new SphereGeometry(0.045, 10, 8), mat("#1b1a18"), { shadow: false });
    eye.position.set(side, 0.35, 0.26);
    return eye;
  });
  const cheeks = [-0.17, 0.17].map((side) => {
    const cheek = mesh(new SphereGeometry(0.04, 8, 6), mat("#f2a6a0"), { shadow: false });
    cheek.position.set(side, 0.27, 0.24);
    cheek.scale.set(1, 0.6, 0.4);
    return cheek;
  });
  const sprout = new Group();
  const stalk = mesh(new CylinderGeometry(0.012, 0.015, 0.14, 5), mat("#4f9a3f"), { shadow: false });
  stalk.position.y = 0.07;
  const leaf = mesh(new SphereGeometry(0.06, 8, 6), mat("#63b34d"), { shadow: false });
  leaf.scale.set(1.4, 0.35, 0.8);
  leaf.position.set(0.05, 0.14, 0);
  leaf.rotation.z = 0.5;
  sprout.add(stalk, leaf);
  sprout.position.y = 0.54;
  const ears = [-1, 1].map((side) => {
    const ear = mesh(new ConeGeometry(0.07, 0.16, 8), fur);
    ear.position.set(side * 0.15, 0.52, -0.02);
    ear.rotation.z = -side * 0.35;
    const inner = mesh(new ConeGeometry(0.04, 0.1, 6), mat("#f2a6a0"), { shadow: false });
    inner.position.set(0, -0.01, 0.03);
    ear.add(inner);
    return ear;
  });
  const tail = mesh(new SphereGeometry(0.08, 10, 8), mat("#fff6ea"));
  tail.position.set(0, 0.2, -0.29);
  const feet = [-0.13, 0.13].map((side) => {
    const foot = mesh(new SphereGeometry(0.07, 10, 8), mat("#e3c4a0"));
    foot.scale.set(1, 0.55, 1.3);
    foot.position.set(side, 0.03, 0.06);
    return foot;
  });
  bodyGroup.add(body, belly, ...eyes, ...cheeks, sprout, ...ears, tail, ...feet);
  g.add(bodyGroup);
  g.userData = { body: bodyGroup, eyes, sprout, ears, feet };
  return label(g, "Haven");
}

// --- little pictures: speech, sleep and what happened ------------------------------------------------

function iconTexture(draw) {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 64;
  const ctx = canvas.getContext("2d");
  draw(ctx);
  const texture = new CanvasTexture(canvas);
  texture.colorSpace = SRGBColorSpace;
  return texture;
}

const ICONS = {
  heart: (c) => {
    c.fillStyle = "#ff5a7a";
    c.beginPath();
    c.moveTo(32, 54);
    c.bezierCurveTo(4, 34, 10, 8, 32, 22);
    c.bezierCurveTo(54, 8, 60, 34, 32, 54);
    c.fill();
  },
  sparkle: (c) => {
    c.fillStyle = "#ffe066";
    c.beginPath();
    for (let i = 0; i < 8; i++) {
      const r = i % 2 ? 9 : 28, a = (i / 8) * Math.PI * 2;
      c.lineTo(32 + Math.cos(a) * r, 32 + Math.sin(a) * r);
    }
    c.fill();
  },
  drop: (c) => {
    c.fillStyle = "#5cb6ff";
    c.beginPath();
    c.moveTo(32, 6);
    c.bezierCurveTo(52, 34, 50, 56, 32, 56);
    c.bezierCurveTo(14, 56, 12, 34, 32, 6);
    c.fill();
  },
  note: (c) => {
    c.fillStyle = "#f2c14e";
    c.beginPath();
    c.ellipse(22, 46, 11, 8, -0.4, 0, Math.PI * 2);
    c.fill();
    c.fillRect(30, 10, 5, 38);
    c.fillRect(30, 10, 20, 6);
  },
  ouch: (c) => {
    c.strokeStyle = "#ff4b3a";
    c.lineWidth = 7;
    c.lineCap = "round";
    for (const [a, b] of [[[14, 14], [50, 50]], [[50, 14], [14, 50]]]) {
      c.beginPath();
      c.moveTo(...a);
      c.lineTo(...b);
      c.stroke();
    }
  },
  sick: (c) => {
    c.fillStyle = "#8bc34a";
    for (const [x, y, r] of [[22, 38, 12], [40, 26, 9], [44, 46, 7]]) {
      c.beginPath();
      c.arc(x, y, r, 0, Math.PI * 2);
      c.fill();
    }
  },
  z: (c) => {
    c.fillStyle = "#e8ecff";
    c.font = "bold 44px system-ui, sans-serif";
    c.textAlign = "center";
    c.textBaseline = "middle";
    c.fillText("z", 32, 34);
  },
  warm: (c) => {
    c.fillStyle = "#ff9d3b";
    c.beginPath();
    c.moveTo(32, 6);
    c.bezierCurveTo(56, 30, 50, 58, 32, 58);
    c.bezierCurveTo(14, 58, 8, 30, 32, 6);
    c.fill();
  },
};
ICONS.pin = (c) => {  // where it's going, when it's doing something it was asked
  c.fillStyle = "#ffcf4a";
  c.strokeStyle = "#7a5200";
  c.lineWidth = 3;
  c.beginPath();
  c.moveTo(32, 60);
  c.bezierCurveTo(10, 34, 10, 6, 32, 6);
  c.bezierCurveTo(54, 6, 54, 34, 32, 60);
  c.fill();
  c.stroke();
  c.fillStyle = "#fffaf0";
  c.beginPath();
  c.arc(32, 25, 8, 0, Math.PI * 2);
  c.fill();
};
const EVENT_ICONS = {
  ate: "sparkle", drank: "drop", rang: "note", pushed: "sparkle", smelled: "heart", shook: "sparkle",
  warmed: "warm", sick: "sick", hurt: "ouch", touched: "heart", chased: "sparkle",
};

// --- the view --------------------------------------------------------------------------------------

export function create(container, { onTouch = null, onHover = null } = {}) {
  let renderer;
  try {
    renderer = new WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  } catch {
    return null; // no WebGL here: the page shows its flat map instead
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = PCFShadowMap;
  renderer.toneMapping = ACESFilmicToneMapping;
  renderer.outputColorSpace = SRGBColorSpace;
  renderer.domElement.className = "world3d";
  container.prepend(renderer.domElement);

  const scene = new Scene();
  scene.background = SKY_DAY.clone();
  scene.fog = new Fog(SKY_DAY.clone(), 30, 70);
  const camera = new PerspectiveCamera(42, 1, 0.1, 200);
  const FOLLOW = new Vector3(0, 7.5, 10.5); // where the camera sits, from Haven
  camera.position.set(12, 17, 33);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.maxPolarAngle = Math.PI * 0.46;
  controls.minDistance = 3;
  controls.maxDistance = 48;
  controls.target.set(12, 0, 12);

  const hemi = new HemisphereLight("#d9ecff", "#56713f", 1.0);
  const sun = new DirectionalLight("#fff1d6", 2.0);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -18, right: 18, top: 18, bottom: -18, near: 1, far: 80 });
  sun.shadow.bias = -0.0006;
  sun.shadow.normalBias = 0.02;
  sun.target.position.set(12, 0, 12);
  scene.add(hemi, sun, sun.target);

  const starGeometry = new BufferGeometry();
  const starPositions = [];
  for (let i = 0; i < 500; i++) {
    const a = Math.random() * Math.PI * 2, e = Math.random() * 0.45 + 0.08;
    starPositions.push(12 + Math.cos(a) * Math.cos(e) * 90, Math.sin(e) * 90, 12 + Math.sin(a) * Math.cos(e) * 90);
  }
  starGeometry.setAttribute("position", new Float32BufferAttribute(starPositions, 3));
  const stars = new Points(starGeometry, new PointsMaterial({ color: "#ffffff", size: 0.5, transparent: true, opacity: 0, fog: false }));
  scene.add(stars);

  // Rain in a storm (falling streaks), snow in a snowstorm (soft flakes).
  const rainGeometry = new BufferGeometry();
  const streaks = new Float32Array(DROPS * 6);
  for (let i = 0; i < DROPS; i++) {
    const [x, y, z] = [Math.random() * 24, Math.random() * 14, Math.random() * 24];
    streaks.set([x, y, z, x + 0.06, y - 0.38, z], i * 6);
  }
  rainGeometry.setAttribute("position", new Float32BufferAttribute(streaks, 3));
  const rain = new LineSegments(rainGeometry, new LineBasicMaterial({ color: "#c9d6ea", transparent: true, opacity: 0.55, depthWrite: false }));
  rain.visible = false;
  const flake = document.createElement("canvas");
  flake.width = flake.height = 32;
  const fc = flake.getContext("2d"), glow = fc.createRadialGradient(16, 16, 0, 16, 16, 16);
  glow.addColorStop(0, "rgba(255,255,255,1)");
  glow.addColorStop(0.5, "rgba(255,255,255,0.8)");
  glow.addColorStop(1, "rgba(255,255,255,0)");
  fc.fillStyle = glow;
  fc.fillRect(0, 0, 32, 32);
  const snowGeometry = new BufferGeometry();
  const flakes = new Float32Array(DROPS * 3);
  for (let i = 0; i < DROPS; i++) flakes.set([Math.random() * 24, Math.random() * 14, Math.random() * 24], i * 3);
  snowGeometry.setAttribute("position", new Float32BufferAttribute(flakes, 3));
  const snow = new Points(snowGeometry, new PointsMaterial({
    map: new CanvasTexture(flake), size: 0.22, transparent: true, depthWrite: false, color: "#ffffff",
  }));
  snow.visible = false;
  scene.add(rain, snow);

  const view = {
    built: false, world: null, cells: new Map(), apples: [], butterflies: [], ball: null, haven: null,
    water: null, bell: null, fire: null, light: 1, phase: 0.3, speed: 8, follow: true, asleep: false,
    target: new Vector3(12, 0, 12), heading: 0, shownHeading: 0, lastTick: -1, ringAt: -1e9, voice: null,
    effects: [], moving: 0, touchedAt: -1e9, rungAt: null,
    made: new Map(), weather: null, flashAt: -1e9, struck: null, shaking: false, fluttering: true,
  };
  const textures = Object.fromEntries(Object.entries(ICONS).map(([k, draw]) => [k, iconTexture(draw)]));
  view.marker = new Sprite(new SpriteMaterial({ map: textures.pin, transparent: true, depthWrite: false }));
  view.marker.scale.setScalar(0.5);
  view.marker.visible = false;
  view.markerAt = new Vector3();
  scene.add(view.marker);

  const bubble = document.createElement("div");
  bubble.className = "speech";
  bubble.hidden = true;
  container.append(bubble);
  const tip = document.createElement("div");
  tip.className = "tip";
  tip.hidden = true;
  container.append(tip);

  function spot(x, y, lift = 0, out = new Vector3()) {
    return out.set(x + 0.5, groundHeight(view.world, x, y) + lift, y + 0.5);
  }

  function place(object, x, y, lift = 0) {
    spot(x, y, lift, object.position);
    return object;
  }

  function build(world) {
    view.world = world;
    view.ground = buildGround(world);
    scene.add(view.ground);
    view.water = buildWater(world);
    if (view.water) scene.add(view.water);
    const flowerColors = new Map((world.flowers || []).map(([x, y, c]) => [`${x},${y}`, c]));
    world.layout.forEach((row, y) => [...row].forEach((c, x) => {
      let thing = null;
      if (c === "A") thing = tree(x, y);
      else if (c === "B") thing = bush(x, y);
      else if (c === "S") thing = stone(x, y);
      else if (c === "T") thing = thorns(x, y);
      else if (c === "N") thing = nest();
      else if (c === "f") thing = flower(x, y, flowerColors.get(`${x},${y}`) || "#f280bf");
      else if (c === "m") thing = label(mushroom(x, y, "#8e5b32", false), "a mushroom");
      else if (c === "t") thing = label(mushroom(x, y, "#d8344f", true), "a toadstool");
      else if (c === "b") thing = view.bell = bell();
      else if (c === "F") thing = view.fire = fire();
      if (thing) {
        place(thing, x, y);
        scene.add(thing);
        view.cells.set(`${x},${y}`, thing);
      }
    }));
    view.ball = ball();
    scene.add(view.ball);
    view.haven = creature();
    view.haven.scale.setScalar(1.3);
    scene.add(view.haven);
    view.butterflies = [];
    moreButterflies((world.butterflies || []).length);
    view.ballAt = null;
    view.built = true;
  }

  function moreButterflies(n) {  // (as many as there are: the person can bring more)
    while (view.butterflies.length < n) {
      const i = view.butterflies.length;
      const b = butterfly(BUTTERFLY_COLORS[i % BUTTERFLY_COLORS.length]);
      b.userData.phase = i * 1.7;
      scene.add(b);
      view.butterflies.push(b);
    }
  }

  function brought(world, now) {  // what the person made happen: the weather, fires and thorns, the ground shaking
    view.weather = world.weather || null;
    if (world.flash !== undefined && world.flash < 6 && view.struck !== world.tick - world.flash) {
      view.struck = world.tick - world.flash;  // lightning
      view.flashAt = now;
    }
    if (world.shaking && !view.shaking) for (const thing of view.cells.values()) thing.userData.wobbleAt = now;
    view.shaking = !!world.shaking;
    view.fluttering = world.fluttering !== false;
    const made = new Set();
    for (const [x, y, what] of world.changed || []) {
      const key = `${x},${y}`;
      let thing = view.made.get(key);
      if (thing && thing.userData.what !== what) { scene.remove(thing); thing = null; }
      if (!thing) {
        thing = what === "fire" ? wildfire(x, y) : what === "thorns" ? thorns(x, y) : null;
        if (!thing) continue;
        thing.userData.what = what;
        thing.userData.popAt = view.lastTick >= 0 ? now : -1e9;
        place(thing, x, y);
        scene.add(thing);
        view.made.set(key, thing);
      }
      made.add(key);
      const under = view.cells.get(key);
      if (under) under.visible = false;  // (a flower that caught fire)
    }
    for (const [key, thing] of view.made) {
      if (made.has(key)) continue;
      scene.remove(thing);  // it burned out, or withered
      view.made.delete(key);
    }
    for (const [key, thing] of view.cells) {  // flowers sleep in winter, unless the person brings them out
      if (thing.userData.label === "flowers" && !made.has(key)) thing.visible = view.season !== "winter" || view.fluttering;
    }
  }

  function effect(kind, at) {
    const sprite = new Sprite(new SpriteMaterial({ map: textures[EVENT_ICONS[kind] || kind], transparent: true, depthWrite: false }));
    sprite.scale.setScalar(0.32);
    sprite.position.copy(at);
    sprite.userData = { born: performance.now(), drift: new Vector3((Math.random() - 0.5) * 0.3, 0.9, (Math.random() - 0.5) * 0.3) };
    scene.add(sprite);
    view.effects.push(sprite);
  }

  function seasonal(season) {
    const look = SEASONS[season] || SEASONS.summer;
    view.look = look;
    for (const [key, thing] of view.cells) {
      const ud = thing.userData;
      if (ud.crown) {
        ud.crown.visible = !!look.leaves;
        ud.branches.visible = !look.leaves;
        if (look.leaves) {
          ud.big.material = mat(look.leaves[0], { flatShading: true });
          ud.side.material = mat(look.leaves[1], { flatShading: true });
        }
      }
      if (ud.label === "flowers") thing.visible = season !== "winter"; // they sleep until spring
      void key;
    }
  }

  function update(world, extra = {}) {
    if (!view.built) build(world);
    view.world = world;
    if (world.season && world.season !== view.season) {
      view.season = world.season;
      seasonal(world.season);
    }
    view.speed = extra.speed || view.speed;
    view.light = world.light;
    view.phase = world.phase ?? view.phase;
    for (const [x, y, count] of world.berries) {
      const b = view.cells.get(`${x},${y}`);
      if (!b) continue;
      b.userData.berries.forEach((berry, i) => { berry.visible = i < count; });
      b.userData.leaves.material = mat(count ? view.look?.bush || "#2f7b3c" : "#4f7d45", { flatShading: true });
      b.userData.label = count ? `a berry bush, with ${count} ${count === 1 ? "berry" : "berries"}` : "a bush with no berries yet";
      b.traverse((o) => { o.userData.label = b.userData.label; });
    }
    for (const [x, y, count] of world.fruit) {
      const t = view.cells.get(`${x},${y}`);
      if (t) t.userData.apples.forEach((apple, i) => { apple.visible = i < count; });
    }
    const now = performance.now();
    for (const [x, y, grown] of world.mushrooms) {
      const m = view.cells.get(`${x},${y}`);
      if (!m) continue;
      if (grown && !m.visible && view.lastTick >= 0) m.userData.popAt = now; // it grew back
      m.visible = !!grown;
    }
    // Apples on the ground: the ones still lying there stay put; new ones fall from the tree.
    const lying = new Map();
    for (const a of view.apples) if (a.visible) lying.set(a.userData.key, [...(lying.get(a.userData.key) || []), a]);
    const kept = new Set();
    for (const [x, y] of world.apples) {
      const key = `${x},${y}`;
      let a = lying.get(key)?.shift();
      if (!a) {
        a = view.apples.find((b) => !b.visible && !kept.has(b)) || fallenApple();
        if (!view.apples.includes(a)) { scene.add(a); view.apples.push(a); }
        place(a, x, y).position.x += 0.18 * (hash(x, y, view.apples.indexOf(a)) - 0.5);
        a.userData.key = key;
        a.userData.ground = a.position.y;
        a.userData.fallAt = view.lastTick >= 0 ? now : -1e9;
      }
      a.visible = true;
      kept.add(a);
    }
    for (const a of view.apples) if (!kept.has(a)) a.visible = false;
    view.ballAt = spot(...world.ball);
    brought(world, now);
    moreButterflies(world.butterflies.length);
    view.butterflies.forEach((b, i) => { b.visible = i < world.butterflies.length; }); // (in winter they're gone)
    world.butterflies.forEach(([x, y], i) => {
      const b = view.butterflies[i];
      if (b) b.userData.to = spot(x, y, 0.75 + 0.25 * hash(x, y, i));
    });
    const rungAt = world.tick - world.rang;
    if (world.rang < 16 && rungAt !== view.rungAt) {
      view.ringAt = performance.now();
      extra.onRing?.();
    }
    view.rungAt = rungAt;
    const moved = world.x !== view.cellX || world.y !== view.cellY;
    view.cellX = world.x;
    view.cellY = world.y;
    spot(world.x, world.y, 0, view.target);
    if (!view.placed) {  // the first time: straight to Haven
      view.placed = true;
      view.haven.position.copy(view.target);
      view.shownHeading = Math.atan2(DIRS[world.heading][0], DIRS[world.heading][1]);
      if (view.follow) {
        controls.target.copy(view.target);
        camera.position.copy(view.target).add(FOLLOW);
      }
    }
    view.heading = world.heading;
    view.asleep = world.asleep;
    view.doing = world.asleep ? null : world.doing || null;  // a pastime: watching, singing, dancing...
    if (world.tick !== view.lastTick) {
      if (view.lastTick >= 0) {
        for (const kind of world.did || []) if (kind !== "bumped") effect(kind, view.haven.position.clone().add(new Vector3(0, 0.75, 0)));
        if ((world.did || []).some((kind) => kind === "shook" || kind === "ate" || kind === "smelled")) {
          const [dx, dy] = DIRS[world.heading];
          const touched = view.cells.get(`${world.x + dx},${world.y + dy}`);
          if (touched) touched.userData.wobbleAt = now; // the tree it shook, the bush it ate from
        }
      }
      view.lastTick = world.tick;
    }
    view.voice = world.voice && world.voice.ago < 3 * view.speed ? world.voice.text : null;
    if (moved) view.moving = 1;
    const errand = world.errand;  // where it's going to do what it was asked
    view.marker.visible = !!(errand && errand.to);
    if (view.marker.visible) {
      spot(errand.to[0], errand.to[1], 1.2, view.markerAt);
      view.marker.userData.label = `where Haven is going: to ${errand.do}`;
    }
  }

  function touched() {
    view.touchedAt = performance.now();
    effect("touched", view.haven.position.clone().add(new Vector3(0, 0.8, 0)));
  }

  // --- looking around with the mouse ---------------------------------------------------------------
  const ray = new Raycaster(), pointer = new Vector2();
  let downAt = null;
  function pick(event) {
    const box = renderer.domElement.getBoundingClientRect();
    pointer.set(((event.clientX - box.left) / box.width) * 2 - 1, -((event.clientY - box.top) / box.height) * 2 + 1);
    ray.setFromCamera(pointer, camera);
    const visible = (o) => { for (; o; o = o.parent) if (!o.visible) return false; return true; };
    const hit = ray.intersectObjects(scene.children, true).find((h) => h.object.isMesh && visible(h.object));
    return hit?.object.userData.label || null; // the nearest thing (the ground has no name)
  }
  renderer.domElement.addEventListener("pointermove", (event) => {
    if (event.buttons) { tip.hidden = true; return; }
    const what = pick(event);
    tip.hidden = !what;
    if (what) {
      const box = container.getBoundingClientRect();
      tip.textContent = what === "Haven" ? "Haven (click to pet it)" : what;
      tip.style.left = `${event.clientX - box.left + 14}px`;
      tip.style.top = `${event.clientY - box.top + 10}px`;
    }
    onHover?.(what);
  });
  renderer.domElement.addEventListener("pointerleave", () => { tip.hidden = true; });
  renderer.domElement.addEventListener("pointerdown", (event) => { downAt = [event.clientX, event.clientY]; });
  let tipTimer = null;
  renderer.domElement.addEventListener("pointerup", (event) => {
    if (!downAt || Math.hypot(event.clientX - downAt[0], event.clientY - downAt[1]) > 5) return;
    const what = pick(event);
    if (what === "Haven" && onTouch) {
      onTouch();
      touched();
    }
    if (what && event.pointerType !== "mouse") {  // no hovering on a touch screen: a tap says what it is
      const box = container.getBoundingClientRect();
      tip.textContent = what;
      tip.style.left = `${event.clientX - box.left + 10}px`;
      tip.style.top = `${event.clientY - box.top - 30}px`;
      tip.hidden = false;
      clearTimeout(tipTimer);
      tipTimer = setTimeout(() => { tip.hidden = true; }, 2500);
    }
  });

  // --- every frame -----------------------------------------------------------------------------------
  let last = performance.now(), running = true, paused = false;
  const skyNow = new Color();
  const sunDir = new Vector3();
  const tint = new Color(), frost = new Color(), pond = new Color(), skyDay = SKY_DAY.clone();
  function frame(now) {
    if (!running || paused) return; // (while the valley isn't shown, nothing is drawn)
    requestAnimationFrame(frame);
    if (now - last < 1000 / 31) return; // 30 frames a second is plenty, and easier on a laptop's battery
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    const t = now / 1000;
    if (!view.built) { renderer.render(scene, camera); return; }

    // Day and night.
    const light = view.light;
    const dusk = Math.max(0, 1 - Math.abs(light - 0.35) / 0.2) * 0.6;
    if (view.look) {  // the season comes over the valley gradually
      const turn = 1 - Math.exp(-dt * 1.5);
      view.ground.material.color.lerp(tint.set(view.look.ground), turn);
      view.ground.material.emissive.lerp(frost.set(view.look.frost), turn);
      if (view.water) view.water.material.color.lerp(pond.set(view.look.water), turn);
      skyDay.lerp(tint.set(view.look.sky), turn);
    }
    skyNow.copy(SKY_NIGHT).lerp(skyDay, Math.min(1, light * 1.1)).lerp(SKY_DUSK, dusk * 0.5);
    const weatherSky = WEATHER_SKY[view.weather];
    if (weatherSky) skyNow.lerp(tint.set(weatherSky[0]), weatherSky[1] * Math.min(1, 0.4 + light));
    const flash = Math.max(0, 1 - (now - view.flashAt) / 220);  // lightning
    scene.background.lerp(skyNow, 0.08);
    if (flash) scene.background.lerp(tint.set("#f4f6ff"), 0.85 * flash);
    scene.fog.color.copy(scene.background);
    const [near, far] = WEATHER_FOG[view.weather] || [30, 70];
    scene.fog.near += (near - scene.fog.near) * (1 - Math.exp(-dt * 0.8));
    scene.fog.far += (far - scene.fog.far) * (1 - Math.exp(-dt * 0.8));
    hemi.intensity = 0.25 + 0.85 * light + 2.5 * flash;
    sun.intensity = 0.15 + 2.1 * light;
    sun.color.set(light < 0.35 ? "#9fb4ff" : "#fff1d6");
    const angle = (view.phase - 0.25) * Math.PI * 2;
    sunDir.set(Math.cos(angle) * 0.9, Math.max(0.25, Math.sin(angle)), 0.45).normalize();
    sun.position.set(12 + sunDir.x * 30, sunDir.y * 30, 12 + sunDir.z * 30);
    stars.material.opacity = Math.max(0, 1 - light * 2.2);
    renderer.toneMappingExposure = 0.75 + 0.35 * light;

    // Haven.
    const h = view.haven, ud = h.userData;
    const k = 1 - Math.exp(-dt * 7);
    const before = h.position.clone();
    h.position.lerp(view.target, k);
    const step = before.distanceTo(h.position);
    view.moving = Math.max(0, view.moving - dt * 2);
    let want = Math.atan2(DIRS[view.heading][0], DIRS[view.heading][1]);
    if (view.doing === "company") {  // keeping you company, it turns to look at you
      want = Math.atan2(camera.position.x - h.position.x, camera.position.z - h.position.z);
    }
    let diff = want - view.shownHeading;
    diff = Math.atan2(Math.sin(diff), Math.cos(diff));
    view.shownHeading += diff * (1 - Math.exp(-dt * 9));
    h.rotation.y = view.shownHeading;
    const walking = step > 0.002;
    const dancing = view.doing === "dance", watching = view.doing === "watch" && !walking;
    const bob = walking ? Math.abs(Math.sin(t * 13)) * 0.07 : dancing ? Math.abs(Math.sin(t * 7)) * 0.16 : 0;
    const breathe = view.asleep ? 1 + Math.sin(t * 1.6) * 0.035 : 1 + Math.sin(t * 2.4) * 0.012;
    const petted = Math.max(0, 1 - (now - view.touchedAt) / 700);
    ud.body.position.y = bob + petted * Math.abs(Math.sin((now - view.touchedAt) / 90)) * 0.06;
    ud.body.scale.set(breathe + petted * 0.06, (view.asleep ? 0.86 : watching ? 0.92 : 1) / breathe, breathe + petted * 0.06);
    ud.body.rotation.z = dancing ? Math.sin(t * 7) * 0.18 : 0;  // swaying to its own tune
    const blink = !view.asleep && Math.sin(t * 0.9) > 0.985;
    ud.eyes.forEach((eye) => { eye.scale.y = view.asleep || blink ? 0.15 : 1; });
    ud.sprout.rotation.z = Math.sin(t * (walking ? 9 : 2)) * (walking ? 0.25 : 0.08);
    ud.ears.forEach((ear, i) => { ear.rotation.x = view.asleep ? 0.5 : Math.sin(t * 3 + i) * 0.06; });
    ud.feet.forEach((foot, i) => { foot.position.z = 0.06 + (walking ? Math.sin(t * 13 + i * Math.PI) * 0.07 : 0); });

    // Sleeping, it breathes out little z's; singing, little notes; keeping you company, now and then a heart.
    if (view.asleep && (!view.lastZ || now - view.lastZ > 1400)) {
      view.lastZ = now;
      effect("z", h.position.clone().add(new Vector3(0.15, 0.7, 0)));
    }
    const every = { sing: 700, dance: 1100, company: 2600 }[view.doing];
    if (every && !view.asleep && (!view.lastNote || now - view.lastNote > every)) {
      view.lastNote = now;
      effect(view.doing === "company" ? "heart" : "note", h.position.clone().add(new Vector3(0.18, 0.75, 0)));
    }

    // The ball rolls to where it is now.
    if (view.ballAt) {
      const b = view.ball, from = b.position.clone();
      if (!view.ballPlaced) { b.position.copy(view.ballAt); view.ballPlaced = true; }
      b.position.lerp(view.ballAt, 1 - Math.exp(-dt * 5));
      const moved = b.position.clone().sub(from);
      if (moved.lengthSq() > 1e-8) {
        const axis = new Vector3(moved.z, 0, -moved.x).normalize();
        b.userData.spin.rotateOnWorldAxis(axis, moved.length() / 0.22);
      }
    }

    // Butterflies drift about, flapping.
    view.butterflies.forEach((b, i) => {
      if (!b.userData.to) return;
      if (!b.userData.placed) { b.position.copy(b.userData.to); b.userData.placed = true; }
      const wobble = new Vector3(Math.sin(t * 1.3 + i) * 0.25, Math.sin(t * 2.1 + i * 2) * 0.12, Math.cos(t * 1.1 + i) * 0.25);
      const goal = b.userData.to.clone().add(wobble);
      const from = b.position.clone();
      b.position.lerp(goal, 1 - Math.exp(-dt * 1.5));
      const v = b.position.clone().sub(from);
      if (v.lengthSq() > 1e-7) b.rotation.y = Math.atan2(v.x, v.z);
      const flap = Math.sin(t * 16 + b.userData.phase) * 0.9;
      b.userData.wings[0].rotation.z = flap;
      b.userData.wings[1].rotation.z = -flap;
    });

    // A tree it shook, or a bush it ate from, sways; mushrooms that grew back pop up; new apples fall.
    for (const thing of view.cells.values()) {
      const wobble = (now - (thing.userData.wobbleAt ?? -1e9)) / 1000;
      if (wobble < 1.2) {
        thing.rotation.z = Math.sin(wobble * 22) * 0.07 * (1 - wobble / 1.2);
        thing.rotation.x = Math.cos(wobble * 19) * 0.04 * (1 - wobble / 1.2);
      } else if (thing.userData.wobbleAt) {
        thing.rotation.z = thing.rotation.x = 0;
        thing.userData.wobbleAt = undefined;
      }
      const pop = (now - (thing.userData.popAt ?? -1e9)) / 400;
      if (pop < 1) thing.scale.setScalar(Math.max(0.05, Math.sin(pop * Math.PI * 0.5) * 1.1));
      else if (thing.userData.popAt) { thing.scale.setScalar(1); thing.userData.popAt = undefined; }
    }
    for (const a of view.apples) {
      const fall = (now - (a.userData.fallAt ?? -1e9)) / 500;
      if (fall < 1) a.position.y = a.userData.ground + 1.7 * (1 - fall * fall);
      else if (a.userData.fallAt > 0) { a.position.y = a.userData.ground; a.userData.fallAt = -1e9; }
    }

    // The bell swings when it has just been rung.
    if (view.bell) {
      const since = (now - view.ringAt) / 1000;
      view.bell.userData.swing.rotation.z = since < 3 ? Math.sin(since * 11) * 0.5 * Math.exp(-since * 1.3) : 0;
    }

    // The fire flickers, and glows more by night.
    if (view.fire) {
      view.fire.userData.flames.forEach((f, i) => {
        const s = 0.85 + 0.25 * Math.sin(t * (9 + i * 3) + i) * Math.sin(t * 5.3 + i * 2);
        f.scale.set(1, s, 1);
        f.rotation.y = t * (0.6 + i * 0.3);
      });
      view.fire.userData.light.intensity = (0.5 + 2.4 * (1 - light)) * (0.9 + 0.12 * Math.sin(t * 13) * Math.sin(t * 7.7));
    }

    // Fires the person started flicker; rain or snow falls.
    for (const thing of view.made.values()) {
      for (const [i, f] of (thing.userData.flames || []).entries()) {
        f.scale.set(1, 0.8 + 0.3 * Math.sin(t * (10 + i * 3) + i + thing.position.x) * Math.sin(t * 6.1 + i), 1);
        f.rotation.y = t * (0.7 + i * 0.3);
      }
      const pop = (now - (thing.userData.popAt ?? -1e9)) / 400;
      thing.scale.setScalar(pop < 1 ? Math.max(0.05, Math.sin(pop * Math.PI * 0.5)) : 1);
    }
    rain.visible = view.weather === "storm";
    snow.visible = view.weather === "snow";
    if (rain.visible) {
      const arr = rain.geometry.attributes.position.array;
      for (let i = 0; i < DROPS; i++) {
        const fall = 13 * dt * (0.8 + 0.4 * ((i * 7919) % 13) / 13);
        arr[i * 6 + 1] -= fall;
        arr[i * 6 + 4] -= fall;
        if (arr[i * 6 + 4] < 0) {
          const [x, y, z] = [Math.random() * 24, 13 + Math.random(), Math.random() * 24];
          arr.set([x, y, z, x + 0.06, y - 0.38, z], i * 6);
        }
      }
      rain.geometry.attributes.position.needsUpdate = true;
    }
    if (snow.visible) {
      const arr = snow.geometry.attributes.position.array;
      for (let i = 0; i < DROPS; i++) {
        arr[i * 3 + 1] -= 1.4 * dt * (0.8 + 0.4 * ((i * 7919) % 13) / 13);
        arr[i * 3] += Math.sin(t * 0.8 + i) * 0.004;  // (snow drifts)
        if (arr[i * 3 + 1] < 0) arr.set([Math.random() * 24, 13 + Math.random(), Math.random() * 24], i * 3);
      }
      snow.geometry.attributes.position.needsUpdate = true;
    }

    // The pond ripples.
    if (view.water) {
      const p = view.water.geometry.attributes.position, rest = view.water.userData.rest;
      for (let i = 0; i < p.count; i++) {
        const x = rest[i * 3], z = rest[i * 3 + 2];
        p.array[i * 3 + 1] = Math.sin(x * 2.1 + t * 1.4) * 0.018 + Math.cos(z * 2.7 + t * 1.1) * 0.014;
      }
      p.needsUpdate = true;
      view.water.geometry.computeVertexNormals();
    }

    if (view.marker.visible) {
      view.marker.position.copy(view.markerAt);
      view.marker.position.y += 0.08 * Math.sin(now / 300);
    }

    // Little pictures of what happened rise and fade.
    view.effects = view.effects.filter((s) => {
      const age = (now - s.userData.born) / 1000;
      if (age > 1.6) { scene.remove(s); s.material.dispose(); return false; }
      s.position.addScaledVector(s.userData.drift, dt * 0.6);
      s.material.opacity = 1 - age / 1.6;
      return true;
    });

    // The camera keeps Haven in view, unless the person is looking around.
    if (view.follow) {
      const goal = h.position.clone();
      const shift = goal.sub(controls.target).multiplyScalar(1 - Math.exp(-dt * 2.5));
      controls.target.add(shift);
      camera.position.add(shift);
    }
    controls.update();
    const shake = view.shaking ? new Vector3((Math.random() - 0.5) * 0.14, (Math.random() - 0.5) * 0.1, 0) : null;
    if (shake) camera.position.add(shake);  // (the ground shaking)
    renderer.render(scene, camera);
    if (shake) camera.position.sub(shake);

    // What it says appears over its head.
    if (view.voice) {
      const head = h.position.clone().add(new Vector3(0, 0.9, 0)).project(camera);
      const box = container.getBoundingClientRect();
      bubble.hidden = head.z > 1;
      bubble.textContent = view.voice;
      bubble.style.left = `${((head.x + 1) / 2) * box.width}px`;
      bubble.style.top = `${((1 - head.y) / 2) * box.height}px`;
    } else bubble.hidden = true;
  }
  requestAnimationFrame(frame);

  const resize = () => {
    const { clientWidth: w, clientHeight: h } = container;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(container);
  resize();

  return {
    update,
    touched,
    follow(on) {
      view.follow = on;
      if (on) {
        if (view.haven) {
          controls.target.copy(view.haven.position);
          camera.position.copy(view.haven.position).add(FOLLOW);
        }
      } else {
        controls.target.set(12, 0, 12);
        camera.position.set(12, 21, 35);
      }
    },
    get following() { return view.follow; },
    inspect() {
      const v = (p) => p && [p.x, p.y, p.z].map((n) => Math.round(n * 100) / 100);
      return { camera: v(camera.position), target: v(controls.target), haven: v(view.haven?.position), cell: [view.cellX, view.cellY] };
    },
    stop() { running = false; renderer.dispose(); },
    pause(on) {
      if (on === paused) return;
      paused = on;
      if (!on && running) { last = performance.now(); resize(); requestAnimationFrame(frame); }
    },
    get paused() { return paused; },
  };
}
