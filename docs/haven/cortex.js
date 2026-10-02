// Haven's language cortex in the browser: the same network as the desktop's (docs/cortex/cortex.onnx, exported by
// packaging/web/export_cortex.py), run by ONNX Runtime Web, on the graphics card where the browser lets it, or else on
// the processor. generate() and meaning() are those of haven/cortex/model.py; drafts() writes several drafts of a
// reply side by side, from one reading of what came to mind.

export class Cortex {
  constructor(ort, session, config, device) {
    this.ort = ort;
    this.session = session;
    this.cfg = config; // {vocab, d, layers, heads, context, slots, core}
    this.device = device; // "webgpu" or "wasm"
    this.size = config.d / config.heads;
    this.busy = Promise.resolve();
    this.read = null; // (its last reading of a prompt: drafts of a reply share it)
  }

  static async load(ort, url, config, { webgpu = true, software = false, onProgress } = {}) {
    const model = await fetchBytes(url, onProgress);
    const options = { graphOptimizationLevel: "all" };
    if (webgpu && (await gpuReady({ software }))) {
      try {
        const outputs = {};
        for (let i = 0; i < config.layers; i++) {
          outputs[`now.${i}.keys`] = "gpu-buffer";
          outputs[`now.${i}.values`] = "gpu-buffer";
        }
        const session = await ort.InferenceSession.create(model, {
          ...options,
          executionProviders: ["webgpu"],
          preferredOutputLocation: outputs,
        });
        return new Cortex(ort, session, config, "webgpu");
      } catch (error) {
        console.warn("Haven's cortex can't use the graphics card here, so it uses the processor:", error);
      }
    }
    const session = await ort.InferenceSession.create(model, { ...options, executionProviders: ["wasm"] });
    return new Cortex(ort, session, config, "wasm");
  }

  parametersCount() {
    const { vocab, d, layers, context, slots, core } = this.cfg;
    return vocab * d + (slots + context) * d + core * d + d + slots * d + layers * (12 * d * d + 9 * d) + 2 * d + d * core + core;
  }

  // One run of the network: rows of new pieces (all the same length), the state rows (or none), and the cache.
  async run(rows, state, past) {
    const { ort, cfg } = this;
    const b = rows.length;
    const t = rows[0].length;
    const feeds = {
      tokens: new ort.Tensor("int64", BigInt64Array.from(rows.flat(), BigInt), [b, t]),
      state: state
        ? new ort.Tensor("float32", tile(state, b), [b, cfg.slots, cfg.core])
        : new ort.Tensor("float32", new Float32Array(0), [b, 0, cfg.core]),
    };
    for (let i = 0; i < cfg.layers; i++) {
      for (const kind of ["keys", "values"]) {
        feeds[`past.${i}.${kind}`] = past ? past[`now.${i}.${kind}`] : new ort.Tensor("float32", new Float32Array(0), [b, cfg.heads, 0, this.size]);
      }
    }
    return this.session.run(feeds);
  }

  // What a piece of text comes to, in the workspace's format (the meaning after its last piece).
  async meaning(ids, state) {
    return this.exclusive(async () => {
      const out = await this.run([ids.slice(-this.cfg.context)], state, null);
      const meaning = Float32Array.from(out.meaning.data);
      release(out);
      return meaning;
    });
  }

  async generate(prompt, state, options = {}) {
    return (await this.drafts(prompt, state, [options.temperature ?? 0.8], options))[0];
  }

  // Drafts of a continuation, one for each temperature, side by side; each is {tokens, logprobs} as generate() gives.
  async drafts(prompt, state, temperatures, { maxNew = 60, topK = 40, stop = [], noRepeat = 0 } = {}) {
    return this.exclusive(async () => {
      const cfg = this.cfg;
      const n = temperatures.length;
      const room = cfg.context - 1;
      const ids = prompt.slice(-Math.max(room - maxNew, 1));
      const given = new Map(); // (what the prompt has after each run of noRepeat - 1 pieces)
      for (let i = 0; noRepeat && i + noRepeat <= ids.length; i++) {
        const key = ids.slice(i, i + noRepeat - 1).join(",");
        if (!given.has(key)) given.set(key, new Set());
        given.get(key).add(ids[i + noRepeat - 1]);
      }
      const rows = Array.from({ length: n }, () => ({ out: [], logprobs: [], said: new Map(), done: false }));
      const rowsRead = this.device === "webgpu" ? n : 1;
      const key = `${rowsRead}|${ids.join(",")}|${state ? Array.from(state).join(",") : ""}`;
      if (!this.read || this.read.key !== key) {
        if (this.read) release(this.read.out);
        this.read = { key, out: await this.run(Array.from({ length: rowsRead }, () => ids), state, null) };
      }
      const first = this.read.out; // (one reading of what came to mind, for every draft of the reply)
      let out = first;
      if (rowsRead < n) {
        out = { logits: tileTensor(this.ort, first.logits, n) };
        for (const name of Object.keys(first)) if (name.startsWith("now.")) out[name] = tileTensor(this.ort, first[name], n);
      }
      let length = ids.length;
      for (let step = 0; step < maxNew; step++) {
        if (step) {
          if (length >= room) break; // (no room left to think in)
          const next = await this.run(rows.map((r) => [r.done ? 0 : r.out[r.out.length - 1]]), null, out);
          if (out !== first) release(out);
          out = next;
        }
        const logits = out.logits.data;
        for (let r = 0; r < n; r++) {
          const row = rows[r];
          if (row.done) continue;
          const scores = Float32Array.from(logits.subarray(r * cfg.vocab, (r + 1) * cfg.vocab));
          const run = noRepeat && row.out.length >= noRepeat - 1 ? row.out.slice(row.out.length - (noRepeat - 1)).join(",") : null;
          if (run !== null) {
            for (const token of row.said.get(run) || []) if (!(given.get(run) || new Set()).has(token)) scores[token] = -Infinity;
          }
          const token = pick(scores, temperatures[r], topK);
          row.logprobs.push(logSoftmaxAt(scores, token));
          if (stop.includes(token)) {
            row.done = true;
            continue;
          }
          if (run !== null) {
            if (!row.said.has(run)) row.said.set(run, new Set());
            row.said.get(run).add(token);
          }
          row.out.push(token);
        }
        length += 1;
        if (rows.every((r) => r.done)) break;
      }
      if (out !== first) release(out);
      return rows.map((r) => ({ tokens: r.out, logprobs: r.logprobs }));
    });
  }

  // (One thing at a time: a reply, or a meaning.)
  exclusive(work) {
    const result = this.busy.then(work);
    this.busy = result.catch(() => {});
    return result;
  }
}

function tile(state, b) {
  const out = new Float32Array(b * state.length);
  for (let i = 0; i < b; i++) out.set(state, i * state.length);
  return out;
}

function tileTensor(ort, tensor, n) {
  const [, ...rest] = tensor.dims;
  return new ort.Tensor("float32", tile(tensor.data, n), [n, ...rest]);
}

function release(out) {
  for (const tensor of Object.values(out)) if (tensor && tensor.location === "gpu-buffer") tensor.dispose();
}

// The next piece: the likeliest (temperature 0), or drawn from the top k, as model.py's generate draws it.
function pick(logits, temperature, topK) {
  const t = Math.max(temperature, 1e-4);
  let max = -Infinity;
  for (const v of logits) if (v > max) max = v;
  const probs = new Float64Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) sum += probs[i] = Math.exp((logits[i] - max) / t);
  for (let i = 0; i < probs.length; i++) probs[i] /= sum;
  if (topK && topK < probs.length) {
    const cutoff = Float64Array.from(probs).sort().reverse()[topK - 1];
    let kept = 0;
    for (let i = 0; i < probs.length; i++) {
      if (probs[i] < cutoff) probs[i] = 0;
      kept += probs[i];
    }
    for (let i = 0; i < probs.length; i++) probs[i] /= kept;
  }
  if (temperature <= 1e-4) {
    let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
    return best;
  }
  let r = Math.random();
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i];
    if (r <= 0 && probs[i] > 0) return i;
  }
  return probs.findLastIndex((p) => p > 0);
}

function logSoftmaxAt(logits, token) {
  let max = -Infinity;
  for (const v of logits) if (v > max) max = v;
  let sum = 0;
  for (const v of logits) if (v !== -Infinity) sum += Math.exp(v - max);
  return logits[token] - max - Math.log(sum);
}

// Whether there's a graphics card to think on: a real one (a browser may offer one done in software instead, which is
// far slower than the processor; `software`: take that too, as the tests do).
export async function gpuReady({ software = false } = {}) {
  try {
    if (!globalThis.navigator?.gpu) return false;
    const none = new Promise((resolve) => setTimeout(() => resolve(null), 5000)); // (a graphics card that doesn't answer)
    const adapter = await Promise.race([navigator.gpu.requestAdapter(), none]);
    if (!adapter) return false;
    const info = adapter.info || {};
    const emulated = adapter.isFallbackAdapter || info.isFallbackAdapter || /swiftshader|llvmpipe|software/i.test(`${info.vendor} ${info.architecture} ${info.description}`);
    return software || !emulated;
  } catch {
    return false;
  }
}

// The bytes of one of Haven's files, saying how far along the download is. With `keep`, they're kept in the browser's
// storage (its Cache Storage: a browser's own cache may not keep a file this big), and next time only checked against
// the site's copy (or, if the site can't be reached, taken as they were).
export async function fetchBytes(url, onProgress, { keep = false } = {}) {
  if (typeof url !== "string") return url; // (bytes already)
  let cache = null;
  let kept = null;
  if (keep) {
    try {
      cache = await caches.open("haven");
      kept = await cache.match(url);
    } catch {
      cache = null; // (no storage for it here)
    }
  }
  const headers = {};
  if (kept?.headers.get("etag")) headers["If-None-Match"] = kept.headers.get("etag");
  else if (kept?.headers.get("last-modified")) headers["If-Modified-Since"] = kept.headers.get("last-modified");
  let response;
  try {
    response = await fetch(url, { headers });
  } catch (error) {
    if (kept) return keptBytes(kept, onProgress);
    throw error;
  }
  if (response.status === 304 && kept) return keptBytes(kept, onProgress);
  if (!response.ok) throw new Error(`couldn't get ${url} (${response.status})`);
  const bytes = await read(response, onProgress);
  if (cache) {
    const saved = new Headers({ "content-type": response.headers.get("content-type") || "application/octet-stream" });
    for (const name of ["etag", "last-modified"]) if (response.headers.get(name)) saved.set(name, response.headers.get(name));
    try {
      await cache.put(url, new Response(bytes, { headers: saved }));
    } catch {
      /* not kept: next time, it's downloaded again */
    }
  }
  return bytes;
}

async function keptBytes(kept, onProgress) {
  const bytes = new Uint8Array(await kept.arrayBuffer());
  onProgress?.(bytes.length, bytes.length);
  return bytes;
}

async function read(response, onProgress) {
  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body || !onProgress) return new Uint8Array(await response.arrayBuffer());
  const reader = response.body.getReader();
  const parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    got += value.length;
    onProgress(got, total);
  }
  const bytes = new Uint8Array(got);
  let at = 0;
  for (const part of parts) {
    bytes.set(part, at);
    at += part.length;
  }
  return bytes;
}
