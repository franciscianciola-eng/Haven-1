// The page's Haven, run by Pyodide in Node, for tests: a conversation (JSON on stdin), what came to mind before each
// reply (JSON on stdout). Its cortex says the replies it's given (or, with "real": true, is its own: cortex.js with
// ONNX Runtime); Wikipedia is tests/web/fake_wiki.py, serving the articles it's given.
import { readFileSync } from "node:fs";
import { loadPyodide } from "./node_modules/pyodide/pyodide.mjs";
import { Tokenizer, END } from "../../docs/haven/tokenizer.js";

const here = new URL(".", import.meta.url);
const spec = JSON.parse(readFileSync(0, "utf8"));
const docs = new URL("../../docs/", here);
const tok = new Tokenizer(JSON.parse(readFileSync(new URL("cortex/tokenizer.json", docs))).merges);
const config = JSON.parse(readFileSync(new URL("cortex/cortex.json", docs))).config;

let cortex;
let reply = "";
if (spec.real) {
  const ort = await import("onnxruntime-web");
  ort.env.wasm.numThreads = 1;
  const { Cortex } = await import("../../docs/haven/cortex.js");
  cortex = await Cortex.load(ort, new Uint8Array(readFileSync(new URL("cortex/cortex.onnx", docs))), config, { webgpu: false });
} else {
  cortex = {
    parametersCount: () => 53001772,
    generate: async () => {
      const tokens = tok.encode(reply);
      return { tokens, logprobs: [...tokens, END].map(() => -0.05) };
    },
    meaning: async () => new Array(config.core).fill(0.01),
  };
}
const page = {
  parametersCount: () => cortex.parametersCount(),
  generate: async (prompt, state, options) => {
    const o = options.toJs ? options.toJs({ dict_converter: Object.fromEntries }) : options;
    const found = await cortex.generate(Array.from(prompt.toJs ? prompt.toJs() : prompt), state ? Float32Array.from(state.toJs ? state.toJs() : state) : null, o);
    return [found.tokens, found.logprobs];
  },
  meaning: async (ids, state) => Array.from(await cortex.meaning(Array.from(ids.toJs ? ids.toJs() : ids), state ? Float32Array.from(state.toJs ? state.toJs() : state) : null)),
};

spec.config = config;
spec.merges = JSON.parse(readFileSync(new URL("cortex/tokenizer.json", docs))).merges;
const py = await loadPyodide({ indexURL: new URL("node_modules/pyodide/", here).pathname });
py.unpackArchive(new Uint8Array(readFileSync(spec.zip)), "zip", { extractDir: "/lib/haven-page" });
py.FS.writeFile("/lib/haven-page/fake_wiki.py", readFileSync(new URL("fake_wiki.py", here), "utf8"));
py.globals.set("spec", py.toPy(spec));
py.globals.set("page", page);
py.globals.set("set_reply", (text) => {
  reply = text;
});
let out;
try {
  out = await py.runPythonAsync(`
import json, random, sys
sys.path.insert(0, "/lib/haven-page")
from haven import browser
from fake_wiki import FakeWiki

spec = spec.to_py() if hasattr(spec, "to_py") else spec
wiki = FakeWiki(spec.get("articles", []), spec.get("redirects", {}))
words = spec.get("dictionary") or {}
haven = browser.Haven(
    "/home/pyodide/haven",
    browser.Cortex(page, spec["config"]),
    spec["merges"],
    {"levels": {}},
    wiki,
    (lambda word: [tuple(e) for e in words.get(word, [])]) if words else None,
    spec.get("weights"),
)
prompts = []
said = haven.thinker._say
async def recording(prompt, state, drafts, most=100, heard=None):
    prompts.append(haven.thinker.tok.decode(prompt))
    return await said(prompt, state, drafts, most, heard)
haven.thinker._say = recording
turns = []
for text, answer in zip(spec["conversation"], spec.get("answers") or [""] * len(spec["conversation"])):
    set_reply(answer)
    prompts.clear()
    if spec.get("traits"):
        haven.life.mind.character.traits = dict(spec["traits"])
    if spec.get("calm"):  # (its chemistry as a mind without a brain keeps it: all usual)
        haven.life.mind.chemistry = dict.fromkeys(haven.life.mind.chemistry, 1.0)
    random.seed(spec.get("seed", 0))
    result = await haven.say(text)
    shelved = []
    for name, volume in haven.shelf.volumes.items():
        shelved += [f"{name}:{t}" for (t,) in volume.db.execute("SELECT title FROM articles")]
    turns.append({"you": text, "prompts": list(prompts), "answer": result["answer"], "confidence": result["confidence"], "thoughts": result["thoughts"], "shelf": shelved, "found": getattr(haven.shelf.find(text), "title", None)})
json.dumps({"turns": turns, "asked": len(wiki.asked), "problem": haven.shelf.problem})
`);
} catch (error) {
  process.stderr.write(String(error.message || error).slice(-4000));
  process.exit(1);
}
process.stdout.write(out);
