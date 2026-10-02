"""A stand-in for the Simple English Wikipedia's API (what the page's Haven reads online), serving given articles:
titles (and redirects to them), each article's plain text as Wikipedia gives it, and searching their words. It runs
inside Pyodide in the page's tests, and in plain Python."""

from __future__ import annotations

import json
import re
from urllib.parse import parse_qs, urlsplit


class FakeWiki:
    def __init__(self, articles: list, redirects: dict | None = None):
        # (Wikipedia gives an article's paragraphs and headings one to a line; the records the desktop reads, between
        # blank lines)
        self.articles = {title: re.sub(r"\n\n+", "\n", text) for title, text in articles}
        self.redirects = dict(redirects or {})
        self.asked: list[dict] = []

    def __call__(self, url: str) -> str:
        q = {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}
        self.asked.append(q)
        if q.get("list") == "search":
            return json.dumps(
                {"query": {"search": [{"title": t} for t in self.search(q["srsearch"], int(q.get("srlimit", 10)))]}}
            )
        if q.get("prop") == "extracts":
            title = self.final(q["titles"])
            return json.dumps({"query": {"pages": [{"title": title, "extract": self.articles.get(title, "")}]}})
        normalized, redirects, pages = [], [], []
        for asked in q["titles"].split("|"):
            title = asked[:1].upper() + asked[1:]
            if title != asked:
                normalized.append({"from": asked, "to": title})
            if title in self.redirects:
                redirects.append({"from": title, "to": self.redirects[title]})
                title = self.redirects[title]
            page = {"title": title} if title in self.articles else {"title": title, "missing": True}
            if page not in pages:
                pages.append(page)
        return json.dumps({"query": {"normalized": normalized, "redirects": redirects, "pages": pages}})

    def final(self, asked: str) -> str:
        title = asked[:1].upper() + asked[1:]
        return self.redirects.get(title, title)

    def search(self, text: str, limit: int) -> list[str]:
        words = set(re.findall(r"\w+", text.lower()))
        scored = []
        for title, body in self.articles.items():
            have = set(re.findall(r"\w+", (title + " " + body).lower()))
            n = len(words & have)
            if n:
                scored.append((-n, title))
        return [t for _, t in sorted(scored)[:limit]]
