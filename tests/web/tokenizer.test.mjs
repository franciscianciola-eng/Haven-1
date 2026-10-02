import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { Tokenizer } from "../../docs/haven/tokenizer.js";

const here = new URL(".", import.meta.url);
const tok = new Tokenizer(JSON.parse(readFileSync(new URL("../../docs/cortex/tokenizer.json", here))).merges);

test("it reads and writes words in the same pieces as the desktop's tokenizer", () => {
  const cases = JSON.parse(readFileSync(new URL("tokenizer-cases.json", here)));
  for (const { text, ids, decoded, specials } of cases) {
    assert.deepEqual(tok.encode(text), ids, text);
    assert.equal(tok.decode(ids), decoded);
    assert.equal(tok.decode(ids, true), specials);
  }
});
