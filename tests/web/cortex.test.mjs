import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import * as ort from "onnxruntime-web";
import { Cortex } from "../../docs/haven/cortex.js";
import { END, YOU } from "../../docs/haven/tokenizer.js";

const here = new URL(".", import.meta.url);
const read = (path) => readFileSync(new URL(path, here));
ort.env.wasm.numThreads = 1;

test("its cortex in the browser says and means what the desktop's does", async () => {
  const config = JSON.parse(read("../../docs/cortex/cortex.json")).config;
  const cortex = await Cortex.load(ort, new Uint8Array(read("../../docs/cortex/cortex.onnx")), config, { webgpu: false });
  assert.equal(cortex.device, "wasm");
  assert.equal(cortex.parametersCount(), 53001772); // (as the desktop counts its connections)
  for (const c of JSON.parse(read("cortex-cases.json"))) {
    const state = Float32Array.from(c.state);
    const [greedy, story] = await Promise.all([
      cortex.generate(c.prompt, state, { maxNew: 40, temperature: 0, stop: [END, YOU] }),
      cortex.generate(c.prompt, state, { maxNew: 30, temperature: 0, stop: [END, YOU], noRepeat: 4 }),
    ]);
    assert.deepEqual(greedy.tokens, c.greedy, c.text);
    assert.deepEqual(story.tokens, c.noRepeat4);
    greedy.logprobs.forEach((p, i) => assert.ok(Math.abs(p - c.logprobs[i]) < 1e-3));
    const meaning = await cortex.meaning(c.prompt, state);
    c.meaning.forEach((m, i) => assert.ok(Math.abs(m - meaning[i]) < 1e-3, `meaning ${i}`));
    const drafts = await cortex.drafts(c.prompt, state, [0, 0.6, 0.6], { maxNew: 40, stop: [END, YOU] });
    assert.deepEqual(drafts[0].tokens, c.greedy); // (the careful draft, beside two freer ones)
    assert.equal(drafts.length, 3);
  }
});
