// Haven's page in a real browser (headless Chromium, with Playwright): it wakes up, talks, reads what it's asked
// about (in a stand-in for the Simple English Wikipedia: serve.py), and is stroked; opened in a second tab meanwhile, it
// waits there until the first is closed, and comes there with everything it had; offline, it answers from what it
// has; and started over, it's a newborn again. Prints what happened as JSON.
// Options, as environment variables: HAVEN_GPU=1 to let its cortex use a (software) graphics card, HAVEN_SHOT=path for
// pictures of the page, HAVEN_VERBOSE=1 to see the page's console as it goes.
import { spawn } from "node:child_process";
import { once } from "node:events";
import { chromium } from "playwright";

const here = new URL(".", import.meta.url);
const server = spawn("python3", [new URL("serve.py", here).pathname], { stdio: ["ignore", "pipe", "inherit"] });
const [port] = (await once(server.stdout, "data")).map((b) => String(b).trim());
const browser = await chromium.launch({
  args: process.env.HAVEN_GPU ? ["--enable-unsafe-webgpu", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan"] : [],
});
const out = { visits: [], wiki: [], console: [] };
let offline = false;

async function open(context) {
  const page = await context.newPage();
  const said = (line) => {
    out.console.push(line.slice(0, 400));
    if (process.env.HAVEN_VERBOSE) process.stderr.write(line + "\n");
  };
  page.on("console", (m) => m.type() !== "debug" && said(`${m.type()}: ${m.text()}`));
  page.on("pageerror", (e) => said(`pageerror: ${e.message}`));
  page.started = Date.now();
  await page.goto(`http://127.0.0.1:${port}/docs/`);
  return page;
}

async function awake(page) {
  await page.waitForSelector("#text:not([disabled]), .notice.error", { timeout: 300000 });
  if (await page.locator(".notice.error").isVisible()) throw new Error(`it didn't wake up: ${await page.textContent("#setup")}`);
  return {
    awake: (Date.now() - page.started) / 1000,
    where: await page.textContent("#cortex"),
    earlier: await page.locator(".event", { hasText: "Earlier" }).count(),
    turns: [],
  };
}

async function talk(page, seen, text) {
  const asking = Date.now();
  const n = await page.locator(".msg.haven").count();
  await page.fill("#text", text);
  await page.press("#text", "Enter");
  await page.waitForFunction(
    (i) => /% sure|couldn't find|went wrong/.test(document.querySelectorAll(".msg.haven .note")[i]?.textContent || ""),
    n,
    { timeout: 600000 },
  );
  const reply = page.locator(".msg.haven").nth(n);
  seen.turns.push({
    you: text,
    haven: await reply.locator(".bubble").textContent(),
    note: await reply.locator(".note").textContent(),
    thoughts: await reply.locator(".thoughts li").allTextContents(),
    seconds: (Date.now() - asking) / 1000,
  });
}

async function done(page, seen, n) {
  seen.state = await page.textContent("#state");
  if (process.env.HAVEN_SHOT) await page.screenshot({ path: process.env.HAVEN_SHOT.replace(/(\.png)?$/, `-${n}.png`) });
  out.visits.push(seen);
}

try {
  const context = await browser.newContext();
  await context.route("https://simple.wikipedia.org/**", async (route) => {
    if (offline) return route.abort("internetdisconnected");
    const url = new URL(route.request().url());
    out.wiki.push(Object.fromEntries(url.searchParams));
    const answer = await fetch(`http://127.0.0.1:${port}/w/api.php${url.search}`);
    await route.fulfill({ status: answer.status, headers: { "access-control-allow-origin": "*", "content-type": "application/json" }, body: await answer.text() });
  });
  const first = await open(context);
  const seen = await awake(first);
  for (const text of ["Hi, I'm Sam", "Tell me about volcanoes", "What do koalas eat?"]) await talk(first, seen, text);
  await first.click("#more");
  await first.click("[data-touch=stroke]");
  await first.waitForFunction(() => /cuddly/.test(document.querySelector("#state").textContent), null, { timeout: 10000 });
  await first.click("#more");
  await first.click("#aboutBtn");
  out.about = await first.textContent("#aboutCortex");
  await first.click("#aboutClose");
  await done(first, seen, 1);

  const second = await open(context); // (while the first is still open)
  await second.waitForSelector("#setupTitle:has-text('another tab')", { timeout: 20000 });
  out.waited = await second.textContent("#setupTitle");
  await first.waitForTimeout(1500); // (what changed, kept in the browser's storage)
  await first.close();
  const again = await awake(second);
  for (const text of ["Tell me about volcanoes", "What's my name?"]) await talk(second, again, text);
  offline = true;
  await talk(second, again, "Who was Cleopatra?");
  offline = false;
  await done(second, again, 2);

  second.on("dialog", (dialog) => dialog.accept()); // ("Start over with a newborn Haven?")
  second.started = Date.now();
  await second.click("#more");
  await second.click("#forget");
  const newborn = await awake(second);
  newborn.hello = await second.locator("#hello").count();
  await done(second, newborn, 3);
  out.asked = await (await fetch(`http://127.0.0.1:${port}/asked`)).json();
} finally {
  await browser.close();
  server.kill();
}
process.stdout.write(JSON.stringify(out, null, 1));
