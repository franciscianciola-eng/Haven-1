// Haven's own tokenizer, as haven/cortex/tokenizer.py: byte-level byte-pair encoding, with the merges it learned.

export const SPECIALS = ["<|end|>", "<|you|>", "<|haven|>", "<|think|>"];
export const [END, YOU, HAVEN, THINK] = [256, 257, 258, 259];
const BASE = 256 + SPECIALS.length;
const PIECES = /'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]{1,3}| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+/g;
const SPECIAL = new RegExp("(" + SPECIALS.map((s) => s.replace(/[|<>]/g, "\\$&")).join("|") + ")");
const utf8 = new TextEncoder();

export class Tokenizer {
  constructor(merges) {
    this.ranks = new Map();
    this.pieces = Array.from({ length: 256 }, (_, i) => Uint8Array.of(i)).concat(SPECIALS.map((s) => utf8.encode(s)));
    this.cache = new Map();
    merges.forEach(([a, b], rank) => {
      this.ranks.set(a * 65536 + b, rank);
      const joined = new Uint8Array(this.pieces[a].length + this.pieces[b].length);
      joined.set(this.pieces[a]);
      joined.set(this.pieces[b], this.pieces[a].length);
      this.pieces.push(joined);
    });
  }

  static async load(url) {
    const { merges } = await (await fetch(url)).json();
    return new Tokenizer(merges);
  }

  get length() {
    return this.pieces.length;
  }

  encode(text) {
    const ids = [];
    for (const part of text.split(SPECIAL)) {
      if (!part) continue;
      const special = SPECIALS.indexOf(part);
      if (special >= 0) {
        ids.push(256 + special);
        continue;
      }
      for (const word of part.match(PIECES) || []) {
        let found = this.cache.get(word);
        if (!found) {
          found = this.piece(word);
          if (this.cache.size > 200000) this.cache.clear();
          this.cache.set(word, found);
        }
        ids.push(...found);
      }
    }
    return ids;
  }

  piece(word) {
    let ids = Array.from(utf8.encode(word));
    while (ids.length > 1) {
      let best = -1;
      let rank = Infinity;
      for (let i = 0; i + 1 < ids.length; i++) {
        const r = this.ranks.get(ids[i] * 65536 + ids[i + 1]);
        if (r !== undefined && r < rank) {
          rank = r;
          best = i;
        }
      }
      if (best < 0) break;
      const [a, b] = [ids[best], ids[best + 1]];
      const merged = [];
      for (let j = 0; j < ids.length; j++) {
        if (j + 1 < ids.length && ids[j] === a && ids[j + 1] === b) {
          merged.push(BASE + rank);
          j++;
        } else merged.push(ids[j]);
      }
      ids = merged;
    }
    return ids;
  }

  decode(ids, specials = false) {
    const bytes = [];
    for (const i of ids) {
      if (i >= 256 && i < BASE && !specials) continue;
      if (i >= 0 && i < this.pieces.length) bytes.push(...this.pieces[i]);
    }
    return new TextDecoder("utf-8", { fatal: false }).decode(Uint8Array.from(bytes));
  }
}
