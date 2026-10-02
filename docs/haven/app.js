// Haven in the browser: its window. Haven itself lives in worker.js (its Python, its cortex, its clock, its storage);
// this is the conversation with it, how it is, its voice, and being with it, as in the desktop's window
// (haven/app.html).

const $ = (id) => document.getElementById(id);
const HOME = "/home/pyodide/haven"; // (the worker keeps its life in the browser's storage under this name)
const store = {
  // little things the page remembers (its voice, whether it speaks), if the browser lets it
  get(key, fallback) {
    try {
      const v = localStorage.getItem("haven." + key);
      return v === null ? fallback : JSON.parse(v);
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem("haven." + key, JSON.stringify(value));
    } catch {
      /* not kept */
    }
  },
};
let name = "Haven";
let ready = false;
let worker = null;
let busy = null; // what it's answering: {id, view, resolve, shown, draft}
let turns = 0;

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function dots() {
  const d = el("span", "dots");
  d.append(el("i"), el("i"), el("i"));
  return d;
}

const plain = (t) => String(t).replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[ch]);
const mb = (bytes) => (bytes / 1e6).toFixed(0);

// --- waking it up ----------------------------------------------------------------------------------------------

// Haven lives in one place at a time: in one tab of this browser (its life is kept in the browser's storage). If
// it's open in another, it comes here once that one is closed (or, reloading, once the page before has let it go).
function start() {
  if (!navigator.locks) {
    wake();
    return;
  }
  let here = false;
  const waiting = setTimeout(() => here || elsewhere(), 2000);
  navigator.locks.request("haven", () => {
    here = true;
    clearTimeout(waiting);
    wake();
    return new Promise(() => {}); // (this tab has it, for as long as it's open)
  });
}

function wake() {
  notice("Haven is waking up…", "");
  worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  worker.onmessage = (e) => heard(e.data);
  worker.onerror = (e) => failed(e.message || "its page couldn't start here");
}

function heard(m) {
  if (m.type === "loading") loading(m);
  else if (m.type === "ready") awake(m);
  else if (m.type === "status") render(m.status);
  else if (m.type === "thought") thought(m);
  else if (m.type === "answer") answered(m);
  else if (m.type === "failed") failed(m.text);
}

function notice(title, text, { error = false, total = 0, got = 0 } = {}) {
  const box = $("setup");
  box.hidden = false;
  box.classList.toggle("error", error);
  $("setupTitle").textContent = title;
  $("setupText").textContent = text;
  const bar = $("setupBar");
  bar.hidden = error;
  bar.classList.toggle("unknown", !total);
  bar.firstElementChild.style.width = total ? `${Math.round((100 * got) / total)}%` : "";
  box.querySelector("button")?.remove();
}

function loading(m) {
  if (m.total) {
    notice(
      "Getting Haven…",
      `${mb(m.got)} of ${mb(m.total)} MB: its cortex, and what runs it. The first time takes a while; after that, your browser keeps them.`,
      m,
    );
  } else if (m.text === "Waking its cortex") {
    notice("Waking its cortex…", "Its language cortex is getting ready to think, here on your computer.");
  } else {
    notice("Waking up…", "Its life, its shelf and your conversation are kept in this browser.");
  }
}

function awake(m) {
  ready = true;
  $("setup").hidden = true;
  render(m.status);
  const where = m.device === "webgpu" ? "on your graphics card" : "on your processor";
  $("cortex").textContent = `Its language area runs here, ${where}`;
  $("cortex").title = `Its language area: ${m.status.cortex}.`;
  $("aboutCortex").textContent = `Its language area: ${m.status.cortex}. Here, it runs ${where}${m.device === "webgpu" ? "" : " (your browser doesn't let it use the graphics card, so it's slower: a reply can take half a minute)"}.`;
  showHistory(m.conversation);
  for (const b of document.querySelectorAll("[data-touch]")) b.disabled = false;
  setComposer();
  $("text").focus();
}

function failed(text) {
  ready = false;
  notice("Haven couldn't wake up here", `${text}. It needs a recent browser: Chrome, Edge, Firefox or Safari.`, { error: true });
  const again = el("button", "", "Try again");
  again.type = "button";
  again.onclick = () => location.reload();
  $("setup").append(again);
  setComposer();
}

function elsewhere() {
  notice("Haven is open in another tab", "It lives in one place at a time: close it there, and it will come here.");
  $("setupBar").hidden = true;
}

// --- how it is -------------------------------------------------------------------------------------------------

function needs(s) {
  // what it needs most, in a few words
  const d = s.needs;
  const found = [];
  const word = (v) => (v > 0.75 ? "very " : v > 0.5 ? "" : "a bit ");
  if (d.hunger > 0.3) found.push(word(d.hunger) + "hungry");
  if (d.tiredness > 0.3) found.push(word(d.tiredness) + "tired");
  if (d.warmth > 0.3) found.push(word(d.warmth) + (s.cold ? "cold" : "too warm"));
  if (d.hurt > 0.3) found.push(word(d.hurt) + "hurt");
  return found;
}

function render(s) {
  name = s.name;
  $("name").textContent = name;
  document.title = name;
  const mood = s.mood || (s.feeling > 0.25 ? "content" : s.feeling < -0.25 ? "unhappy" : "");
  const parts = [s.asleep ? "<b>asleep</b>" : mood ? `feeling <b>${plain(mood)}</b>` : "", ...needs(s), plain(s.age), plain(`${s.season} ${s.time}`)];
  $("state").innerHTML = parts.filter(Boolean).join(" · ");
}

// --- the conversation ------------------------------------------------------------------------------------------

function scrollDown(force) {
  const log = $("log");
  if (force || log.scrollHeight - log.scrollTop - log.clientHeight < 200) log.scrollTop = log.scrollHeight;
}

function addYou(text) {
  $("hello")?.remove();
  const m = el("div", "msg you");
  m.append(el("div", "bubble", text));
  $("thread").append(m);
  scrollDown(true);
}

function havenMessage(text) {
  $("hello")?.remove();
  const m = el("div", "msg haven");
  const who = el("div", "who", name);
  const hear = el("button", "", "🔈");
  hear.type = "button";
  hear.title = "Hear it say this";
  const view = { who, thoughts: el("div", "thoughts"), list: el("ol"), bubble: el("div", "bubble"), note: el("div", "note") };
  hear.onclick = () => {
    const said = view.bubble.textContent;
    if (said && !view.bubble.classList.contains("pending")) voice.say(said, { now: true });
  };
  who.append(hear);
  view.thoughts.hidden = true;
  view.thoughts.append(view.list);
  if (text !== undefined) view.bubble.textContent = text;
  m.append(view.who, view.thoughts, view.bubble, view.note);
  $("thread").append(m);
  scrollDown();
  return view;
}

function addEvent(text) {
  $("hello")?.remove();
  $("thread").append(el("div", "event", text));
  scrollDown();
}

function thoughtLine(t) {
  if (t.kind === "draft") return el("li", "", `Words that came to it: “${t.text}”`);
  if (t.kind === "read") return el("li", "read", `Read about ${t.text} in the Simple English Wikipedia, online.`);
  if (t.kind === "found") return el("li", "read", `Found it on its shelf: ${t.text}.`);
  return el("li", "", t.text);
}

function send(text) {
  // says something to Haven; resolves to its reply ("" if it couldn't answer)
  addYou(text);
  const view = havenMessage();
  view.bubble.classList.add("pending");
  view.bubble.append(dots());
  view.note.textContent = "listening…";
  const id = ++turns;
  return new Promise((resolve) => {
    busy = { id, view, resolve, shown: 0, draft: "" };
    setComposer();
    worker.postMessage({ type: "say", id, text });
  });
}

function addThought(turn, t) {
  turn.view.list.append(thoughtLine(t));
  turn.view.thoughts.hidden = false;
  turn.shown += 1;
}

function thought(m) {
  // a reply taking shape (as the desktop's window follows one: haven/app.py)
  const turn = busy;
  if (!turn || m.id !== turn.id) return;
  if (m.kind === "draft") {
    // a fresh try at the words: the last one becomes a passing thought
    if (turn.draft.trim()) addThought(turn, { kind: "draft", text: turn.draft.trim() });
    turn.draft = "";
    turn.view.note.textContent = "answering…";
  } else if (m.kind === "words") {
    turn.draft += m.text;
    turn.view.bubble.textContent = turn.draft;
  } else if (m.kind === "pondering") {
    turn.view.note.textContent = "thinking it over…";
  } else if (m.kind === "reading") {
    turn.view.note.textContent = `reading about ${m.text} in the Simple English Wikipedia…`;
  } else if (["thought", "read", "found"].includes(m.kind)) {
    addThought(turn, m);
  }
  scrollDown();
}

function answered(m) {
  const turn = busy;
  if (!turn || m.id !== turn.id) return;
  busy = null;
  const reply = m.answer || "";
  const view = turn.view;
  view.bubble.classList.remove("pending");
  view.bubble.textContent = reply || "…";
  view.note.textContent = reply
    ? `${Math.round(m.confidence * 100)}% sure`
    : m.problem
      ? `Something went wrong as it thought (${m.problem}).`
      : "It couldn't find the words this time.";
  if (turn.shown) {
    // its thinking, tucked away once it has answered
    const details = el("details");
    details.append(el("summary", "", turn.shown === 1 ? "How it got there (1 thought)" : `How it got there (${turn.shown} thoughts)`), view.list);
    view.thoughts.replaceChildren(details);
  }
  if (m.status) render(m.status);
  setComposer();
  scrollDown();
  turn.resolve(reply);
}

function showHistory(conversation) {
  if (!conversation.length) {
    const hello = el("div", "hello");
    hello.id = "hello";
    hello.innerHTML =
      `<h2>Talk with ${plain(name)}</h2>` +
      `<p>${plain(name)} answers in words of its own, from what it feels and what it reads. Every word comes from its own small ` +
      `language cortex, running here, on your computer. Ask it anything, and it reads about it in the Simple English ` +
      `Wikipedia, and tells you what it read.</p><p>Press 🎙 Talk to talk out loud.</p>`;
    const tries = el("div", "tries");
    for (const q of ["Hi! What's your name?", "How are you feeling?", "What is a black hole?", "Who was Cleopatra?", "Tell me about volcanoes.", "What's the capital of Japan?"]) {
      const b = el("button", "", q);
      b.type = "button";
      b.onclick = async () => {
        if (!ready || busy) return;
        const reply = await send(q);
        if (reply && voice.speaking) voice.say(reply);
      };
      tries.append(b);
    }
    hello.append(tries);
    $("thread").append(hello);
    return;
  }
  addEvent("Earlier");
  for (const c of conversation) {
    if (c.who === "you") addYou(c.text);
    else havenMessage(c.text);
  }
  addEvent("Now");
  scrollDown(true);
}

function setComposer() {
  const text = $("text");
  text.disabled = !ready;
  $("send").disabled = !ready || !!busy;
  $("mic").disabled = !ready || !!busy || !voice.canListen;
  $("talk").disabled = !ready;
  text.placeholder = !ready ? "Haven is waking up…" : busy ? `${name} is answering…` : `Talk to ${name}, or ask it anything…`;
}

function autosize() {
  const t = $("text");
  t.style.height = "auto";
  t.style.height = Math.min(t.scrollHeight + 2, 200) + "px";
}

$("text").addEventListener("input", autosize);
$("text").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    $("form").requestSubmit();
  }
});
$("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = $("text").value.trim();
  if (!text || !ready || busy) return;
  $("text").value = "";
  autosize();
  const reply = await send(text);
  if (reply && voice.speaking && !voice.mode) voice.say(reply);
});

// --- being with it ---------------------------------------------------------------------------------------------

function touch(what) {
  if (!ready) return;
  worker.postMessage({ type: "touch", what });
  addEvent(what === "stroke" ? `You stroked ${name}.` : `You gave ${name} a berry.`);
}

for (const b of document.querySelectorAll("[data-touch]")) {
  b.onclick = () => {
    closeMenus();
    touch(b.dataset.touch);
  };
}

$("forget").onclick = async () => {
  closeMenus();
  const sure = confirm(
    `Start over with a newborn Haven?\n\n${name} will forget everything: your conversation, what it has read, and how it has lived. This can't be undone.`,
  );
  if (!sure) return;
  voice.stop();
  worker?.terminate();
  await new Promise((resolve) => {
    const request = indexedDB.deleteDatabase(HOME);
    request.onsuccess = request.onerror = request.onblocked = resolve;
  });
  location.reload();
};

// (what changed since it last kept it, kept as the page goes out of sight)
document.addEventListener("visibilitychange", () => {
  if (document.hidden && ready) worker.postMessage({ type: "keep" });
});

// --- its voice: speaking, and listening ------------------------------------------------------------------------
// Haven's words are its own; the browser only gives them a sound (the computer's speech voices), and turns what
// you say into words for it (in Chrome and Edge, your voice is sent to their speech service to be written down).

const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const voice = {
  synth: window.speechSynthesis || null,
  canListen: !!Recognition,
  speaking: store.get("speak", false), // whether it reads its replies aloud
  mode: false, // talking out loud, hands-free
  rec: null,
  queue: 0,
  misses: 0,

  voices() {
    if (!this.synth) return [];
    const all = this.synth.getVoices();
    const lang = (navigator.language || "en").slice(0, 2);
    const mine = all.filter((v) => v.lang && v.lang.slice(0, 2) === lang);
    return mine.length ? mine : all;
  },
  chosen() {
    const all = this.voices();
    const wanted = store.get("voice", "");
    return (
      all.find((v) => v.name === wanted) ||
      all.find((v) => /natural|neural|premium|enhanced|siri/i.test(v.name) && v.localService) ||
      all.find((v) => /natural|neural|premium|enhanced/i.test(v.name)) ||
      all.find((v) => v.default) ||
      all[0] ||
      null
    );
  },
  setSpeaking(on) {
    this.speaking = on;
    store.set("speak", on);
    $("speakOn").checked = on;
    $("voiceBtn").textContent = on ? "🔊" : "🔈";
    $("voiceBtn").append(el("span", "wide", " Voice"));
    if (!on && !this.mode) this.synth?.cancel();
  },
  say(text, { now = false, done = null } = {}) {
    // it says it, sentence by sentence (long speech gets cut off)
    if (!this.synth) {
      done?.();
      return;
    }
    if (now) this.synth.cancel();
    const parts = (text.match(/[^.!?…]+[.!?…]*["”’)]*\s*/g) || [text]).map((p) => p.trim()).filter(Boolean);
    const chosen = this.chosen();
    this.queue += parts.length;
    parts.forEach((part, i) => {
      const u = new SpeechSynthesisUtterance(part);
      if (chosen) {
        try {
          u.voice = chosen;
          u.lang = chosen.lang;
        } catch {
          /* the browser's own choice, then */
        }
      }
      u.rate = Number(store.get("rate", 1));
      u.pitch = Number(store.get("pitch", 1.15));
      const last = i === parts.length - 1;
      u.onstart = () => this.show("speaking");
      u.onend = u.onerror = () => {
        this.queue = Math.max(0, this.queue - 1);
        if (last) done?.();
        if (!this.queue && !this.mode) this.show(null);
      };
      this.synth.speak(u);
    });
  },
  show(state, heard) {
    $("orb").className = "orb" + (state ? " " + state : "");
    $("voiceState").textContent =
      { listening: "Listening…", thinking: `${name} is thinking…`, speaking: `${name} is speaking…`, paused: "Tap the circle to talk" }[state] || "";
    if (heard !== undefined) $("voiceHeard").textContent = heard;
  },
  listen(onWords) {
    // listens once; calls onWords(text) with what it heard ("" for nothing)
    if (!Recognition) {
      onWords("");
      return;
    }
    const rec = new Recognition();
    this.rec = rec;
    rec.lang = navigator.language || "en-US";
    rec.interimResults = true;
    rec.maxAlternatives = 1;
    let final = "";
    let ended = false;
    rec.onresult = (e) => {
      let interim = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        if (e.results[i].isFinal) final += e.results[i][0].transcript;
        else interim += e.results[i][0].transcript;
      }
      this.show("listening", (final + interim).trim() || "…");
    };
    rec.onerror = (e) => {
      if (e.error === "not-allowed" || e.error === "service-not-allowed") {
        addEvent("The browser isn't allowed to use the microphone. Allow it (the icon by the address), then try again.");
        this.stop();
      } else if (e.error === "network") {
        addEvent("The browser couldn't reach its speech service to hear you (it needs the internet for that).");
        this.stop();
      }
    };
    rec.onend = () => {
      if (ended) return;
      ended = true;
      this.rec = null;
      onWords(final.trim());
    };
    try {
      rec.start();
      this.show("listening", "");
    } catch {
      ended = true;
      onWords("");
    }
  },
  start() {
    // talking out loud: it listens, answers, speaks, and listens again
    if (!Recognition) {
      addEvent("This browser can't listen. Use Chrome, Edge or Safari to talk out loud; here, Haven can still speak its replies (🔈 Voice).");
      this.setSpeaking(true);
      return;
    }
    if (this.mode) return;
    this.mode = true;
    this.misses = 0;
    $("voicebar").hidden = false;
    $("talk").setAttribute("aria-pressed", "true");
    this.next();
  },
  next() {
    if (!this.mode) return;
    if (busy || !ready) {
      this.show("thinking", "");
      setTimeout(() => this.next(), 500);
      return;
    }
    this.listen(async (text) => {
      if (!this.mode) return;
      if (!text) {
        if (++this.misses >= 3) {
          this.show("paused", "It didn't hear anything.");
          return;
        }
        this.next();
        return;
      }
      this.misses = 0;
      if (/^(stop|stop talking|stop listening|goodbye|bye bye)$/i.test(text)) {
        this.stop();
        addEvent("Voice mode stopped.");
        return;
      }
      this.show("thinking", text);
      const reply = await send(text);
      if (!this.mode) return;
      if (reply) this.say(reply, { done: () => this.next() });
      else this.next();
    });
  },
  stop() {
    this.mode = false;
    try {
      this.rec?.abort();
    } catch {
      /* already stopped */
    }
    this.rec = null;
    this.synth?.cancel();
    this.queue = 0;
    $("voicebar").hidden = true;
    $("talk").setAttribute("aria-pressed", "false");
    this.show(null, "");
  },
};

$("talk").onclick = () => (voice.mode ? voice.stop() : voice.start());
$("voiceStop").onclick = () => voice.stop();
$("orb").onclick = () => {
  // tap: stop it speaking and talk now; or start listening again
  if (!voice.mode) return;
  if (voice.rec) {
    try {
      voice.rec.stop();
    } catch {
      /* stopped */
    }
    return;
  }
  voice.synth?.cancel();
  voice.queue = 0;
  voice.misses = 0;
  voice.next();
};
$("mic").onclick = () => {
  // say one thing out loud instead of typing it
  if (voice.rec) {
    try {
      voice.rec.stop();
    } catch {
      /* stopped */
    }
    return;
  }
  if (!ready || busy) return;
  $("mic").classList.add("on");
  voice.listen(async (text) => {
    $("mic").classList.remove("on");
    if (!voice.mode) voice.show(null, "");
    if (!text) return;
    const reply = await send(text);
    if (reply) voice.say(reply); // (asked out loud: it answers out loud)
  });
};
if (!Recognition) $("mic").title = "This browser can't listen (Chrome, Edge and Safari can)";

function fillVoices() {
  const pick = $("voicePick");
  const all = voice.voices();
  const chosen = voice.chosen();
  pick.replaceChildren(
    ...all.map((v) => {
      const o = el("option", "", `${v.name}${v.localService ? "" : " (online)"}`);
      o.value = v.name;
      o.selected = chosen && v.name === chosen.name;
      return o;
    }),
  );
  $("voiceAbout").textContent = !voice.synth
    ? "This browser has no speech voices."
    : all.length
      ? "These are your computer's voices; the words are Haven's own."
      : "No voices yet…";
}
if (voice.synth) voice.synth.onvoiceschanged = fillVoices;
fillVoices();
$("voicePick").onchange = (e) => store.set("voice", e.target.value);
$("rate").value = store.get("rate", 1);
$("pitch").value = store.get("pitch", 1.15);
$("rate").oninput = (e) => store.set("rate", Number(e.target.value));
$("pitch").oninput = (e) => store.set("pitch", Number(e.target.value));
$("speakOn").onchange = (e) => voice.setSpeaking(e.target.checked);
$("voiceTry").onclick = () => voice.say(`Hello! I'm ${name}. This is how I sound.`, { now: true });
voice.setSpeaking(voice.speaking);

// --- menus, and about ------------------------------------------------------------------------------------------

const MENUS = [
  ["moreMenu", "more"],
  ["voiceMenu", "voiceBtn"],
];
function closeMenus(except) {
  for (const [menu, button] of MENUS) {
    if (menu === except) continue;
    $(menu).hidden = true;
    $(button).setAttribute("aria-expanded", "false");
  }
}
for (const [menu, button] of MENUS) {
  $(button).onclick = (e) => {
    e.stopPropagation();
    const open = $(menu).hidden;
    closeMenus(menu);
    $(menu).hidden = !open;
    $(button).setAttribute("aria-expanded", String(open));
    if (menu === "voiceMenu") fillVoices();
  };
  $(menu).addEventListener("click", (e) => e.stopPropagation());
}
document.addEventListener("click", () => closeMenus());
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeMenus();
});
$("aboutBtn").onclick = () => {
  closeMenus();
  $("about").showModal();
};
$("aboutClose").onclick = () => $("about").close();
$("about").addEventListener("click", (e) => {
  if (e.target === $("about")) $("about").close(); // (a click outside it)
});

setComposer();
start();
