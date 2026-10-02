// Haven in the browser: its life, run here, off the page's main thread. Pyodide runs Haven's own conversation code
// (py/haven.zip: haven/cortex, with haven/browser.py), and ONNX Runtime Web its cortex (cortex.js). Its clock ticks as
// the desktop's does (8 moments a second, while the page is open), and what it lives, says and reads is kept in the
// browser's storage (IndexedDB), so it's all still there next time. The page (app.js) and it talk in messages:
//
//   page → here:  {type: "say", id, text} · {type: "touch", what: "stroke" | "feed"} · {type: "keep"}
//   here → page:  {type: "loading", text, got, total} · {type: "ready", status, conversation, device}
//                 {type: "thought", id, kind, text} · {type: "answer", id, answer, confidence, status}
//                 {type: "status", status} · {type: "failed", text}
import { Cortex, fetchBytes, gpuReady } from "./cortex.js";

const SITE = new URL("../", import.meta.url);
const at = (path) => new URL(path, SITE).href;
const HOME = "/home/pyodide/haven"; // (its home, kept in the browser's storage, as the desktop's is a folder)
const MOMENT = 125; // ms: 8 moments a second, as the desktop's life
const KEEP = 30000; // ms between keeping how it is in storage, when nothing else has

let py = null;
let haven = null; // (its Python: say, touch, status, tick, keep, history)
const said = []; // what's been said to it, answered one at a time
let answering = false;

self.onmessage = (e) => {
  const m = e.data;
  if (m.type === "say") {
    said.push(m);
    answer();
  } else if (m.type === "touch" && haven) {
    post({ type: "status", status: JSON.parse(haven.touch(m.what)) });
    persist();
  } else if (m.type === "keep" && haven) {
    haven.keep();
    persist();
  }
};

const post = (message) => postMessage(message);
const began = performance.now();
const note = (text) => console.info(`Haven: ${text} (${((performance.now() - began) / 1000).toFixed(1)} s)`);

// --- waking up -------------------------------------------------------------------------------------------------

const downloads = new Map(); // what it's getting: [got, total] bytes
function downloading(name) {
  downloads.set(name, [0, 0]);
  return (got, total) => {
    downloads.set(name, [got, Math.max(total, got)]);
    let all = 0;
    let sum = 0;
    for (const [g, t] of downloads.values()) {
      sum += Math.min(g, t);
      all += t;
    }
    post({ type: "loading", text: "Getting Haven", got: sum, total: all });
  };
}

async function cortexReady() {
  const [config, gpu] = await Promise.all([json("cortex/cortex.json"), gpuReady()]);
  const ort = await import(gpu ? "../vendor/ort/ort.webgpu.min.mjs" : "../vendor/ort/ort.wasm.min.mjs");
  ort.env.logLevel = "error";
  ort.env.wasm.wasmPaths = at("vendor/ort/");
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  const runtime = gpu ? "ort-wasm-simd-threaded.asyncify.wasm" : "ort-wasm-simd-threaded.wasm";
  const [binary, model] = await Promise.all([
    fetchBytes(at(`vendor/ort/${runtime}`), downloading("runtime")),
    fetchBytes(at("cortex/cortex.onnx"), downloading("cortex")),
  ]);
  ort.env.wasm.wasmBinary = binary;
  note("its cortex is here");
  post({ type: "loading", text: "Waking its cortex" });
  const cortex = await Cortex.load(ort, model, config.config, { webgpu: gpu });
  note(`its cortex is awake, on the ${cortex.device === "webgpu" ? "graphics card" : "processor"}`);
  return { cortex, info: config };
}

async function pythonReady() {
  const { loadPyodide } = await import("../vendor/pyodide/pyodide.mjs");
  const [python, code, weights, tokenizer] = await Promise.all([
    loadPyodide({ indexURL: at("vendor/pyodide/"), stdout: () => {}, stderr: (text) => console.warn(text) }),
    fetchBytes(at("py/haven.zip"), downloading("python")),
    text("data/weights.json", downloading("weights")),
    text("cortex/tokenizer.json"),
  ]);
  python.unpackArchive(code, "zip", { extractDir: "/lib/haven-page" });
  python.FS.mkdirTree(HOME);
  python.FS.mount(python.FS.filesystems.IDBFS, {}, HOME);
  await new Promise((resolve, reject) => python.FS.syncfs(true, (error) => (error ? reject(error) : resolve())));
  note("its Python is ready, and what it kept");
  return { python, weights, tokenizer };
}

async function wake() {
  const [{ cortex, info }, { python, weights, tokenizer }] = await Promise.all([cortexReady(), pythonReady()]);
  post({ type: "loading", text: "Waking up" });
  py = python;
  py.globals.set("page_cortex", {
    parametersCount: () => cortex.parametersCount(),
    // (what Python hands over is turned into plain values before anything's awaited: it's only lent for the call)
    generate(prompt, state, options) {
      const o = options.toJs({ dict_converter: Object.fromEntries });
      return cortex.generate(Array.from(prompt.toJs()), state ? Float32Array.from(state.toJs()) : null, o).then((found) => [found.tokens, found.logprobs]);
    },
    meaning(ids, state) {
      return cortex.meaning(Array.from(ids.toJs()), state ? Float32Array.from(state.toJs()) : null).then((m) => Array.from(m));
    },
  });
  py.globals.set("page_fetch", fetchText);
  py.globals.set("page_heard", (kind, text) => post({ type: "thought", id: said[0]?.id, kind, text }));
  py.globals.set("config", JSON.stringify(info));
  py.globals.set("weights", weights);
  py.globals.set("tokenizer", tokenizer);
  py.globals.set("dictionary", at("dictionary"));
  await py.runPythonAsync(`
import json, sys
sys.path.insert(0, "/lib/haven-page")
from haven import browser

info = json.loads(config)
_haven = browser.Haven(
    ${JSON.stringify(HOME)},
    browser.Cortex(page_cortex, info["config"]),
    json.loads(tokenizer)["merges"],
    {**info.get("progress", {}), "described": info.get("described")},
    page_fetch,
    browser.Dictionary(page_fetch, dictionary),
    json.loads(weights),
)
del weights, tokenizer
_haven.on_thought = page_heard


async def say(text):
    return json.dumps(await _haven.say(text))


def touch(what):
    return json.dumps(_haven.touch(what))


def status():
    return json.dumps(_haven.status())


def history():
    return json.dumps(_haven.conversation())


def tick():
    _haven.step(1)


def keep():
    _haven.save()
`);
  haven = Object.fromEntries(["say", "touch", "status", "history", "tick", "keep"].map((name) => [name, py.globals.get(name)]));
  note("awake");
  post({ type: "ready", status: JSON.parse(haven.status()), conversation: JSON.parse(haven.history()), device: cortex.device });
  setInterval(tick, MOMENT);
  setInterval(() => post({ type: "status", status: JSON.parse(haven.status()) }), 1000);
  setInterval(() => {
    haven.keep();
    persist();
  }, KEEP);
  answer();
}

// --- living ----------------------------------------------------------------------------------------------------

function tick() {
  try {
    haven.tick();
  } catch (error) {
    console.error("Haven's moment went wrong:", error);
  }
}

async function answer() {
  if (!haven || answering || !said.length) return;
  answering = true;
  const { id, text } = said[0];
  try {
    const found = JSON.parse(await haven.say(text));
    post({ type: "answer", id, answer: found.answer, confidence: found.confidence, status: found.status });
  } catch (error) {
    console.error(error);
    post({ type: "answer", id, answer: "", confidence: 0, problem: String(error.message || error).split("\n").filter(Boolean).pop() });
  }
  said.shift();
  answering = false;
  persist();
  answer();
}

// Keeping what changed in the browser's storage (one keeping at a time: if more changes meanwhile, again after).
let keeping = false;
let again = false;
function persist() {
  if (!py) return;
  if (keeping) {
    again = true;
    return;
  }
  keeping = true;
  py.FS.syncfs(false, (error) => {
    if (error) console.warn("Haven couldn't keep what changed in this browser's storage:", error);
    keeping = false;
    if (again) {
      again = false;
      persist();
    }
  });
}

// What its shelf reads online (the Simple English Wikipedia), and its dictionary: in its own time, as it thinks.
function fetchText(url) {
  const request = new XMLHttpRequest();
  request.open("GET", url, false); // (synchronous: allowed in a worker, and what its Python expects)
  request.timeout = 20000;
  request.send();
  if (request.status !== 200) throw new Error(`${url.split("?")[0]} answered ${request.status}`);
  return request.responseText;
}

async function text(path, onProgress) {
  return new TextDecoder().decode(await fetchBytes(at(path), onProgress));
}

async function json(path) {
  return JSON.parse(await text(path));
}

wake().catch((error) => {
  console.error(error);
  post({ type: "failed", text: String(error.message || error).split("\n").filter(Boolean).pop() });
});
