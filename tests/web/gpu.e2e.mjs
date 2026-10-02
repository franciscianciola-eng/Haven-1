// Haven's cortex on a graphics card (WebGPU, as most browsers will run it), in headless Chromium with its software
// adapter (SwiftShader: slow, but the same arithmetic): what it says and means, as the desktop's cortex does
// (cortex-cases.json). Prints what it found as JSON.
import { spawn } from "node:child_process";
import { once } from "node:events";
import { chromium } from "playwright";

const server = spawn("python3", [new URL("serve.py", import.meta.url).pathname], { stdio: ["ignore", "pipe", "inherit"] });
const [port] = (await once(server.stdout, "data")).map((b) => String(b).trim());
const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan"] });
try {
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${port}/tests/web/`); // (a page of the same site, without Haven on it)
  const found = await page.evaluate(async () => {
    const ort = await import("/docs/vendor/ort/ort.webgpu.min.mjs");
    ort.env.wasm.wasmPaths = "/docs/vendor/ort/";
    ort.env.logLevel = "error";
    const { Cortex } = await import("/docs/haven/cortex.js");
    const config = (await (await fetch("/docs/cortex/cortex.json")).json()).config;
    const cortex = await Cortex.load(ort, "/docs/cortex/cortex.onnx", config, { webgpu: true, software: true });
    const c = (await (await fetch("/tests/web/cortex-cases.json")).json())[0];
    const state = Float32Array.from(c.state);
    const greedy = await cortex.generate(c.prompt, state, { maxNew: 6, temperature: 0, stop: [256, 257] });
    const drafts = await cortex.drafts(c.prompt, state, [0, 0.6, 0.6], { maxNew: 6, stop: [256, 257] });
    const meaning = Array.from(await cortex.meaning(c.prompt, state));
    return { device: cortex.device, greedy: greedy.tokens, logprobs: greedy.logprobs, drafts: drafts.map((d) => d.tokens), meaning, want: c };
  });
  process.stdout.write(JSON.stringify(found));
} finally {
  await browser.close();
  server.kill();
}
