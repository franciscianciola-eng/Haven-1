"""Haven's window on the internet: searching and reading pages. Read-only, and no API keys."""

from __future__ import annotations

import html
import http.client
import ipaddress
import json
import re
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import quote, urlencode, urljoin, urlsplit, urlunsplit

USER_AGENT = "Mozilla/5.0 (compatible; Haven/0.2; +https://github.com/franciscianciola-eng/Haven-1)"
MAX_BYTES = 3_000_000
PRIVATE = "that address is on a private or local network, so Haven won't open it"
_URL = re.compile(r"https?://[^\s<>\"'`\])}]+")


class WebError(Exception):
    pass


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


@dataclass(frozen=True)
class Page:
    url: str
    title: str
    text: str
    links: list[tuple[str, str]]  # (link text, address)


class Web:
    def __init__(self, safesearch: str = "off", timeout: float = 15):
        self.safesearch = safesearch
        self.timeout = timeout

    def search(self, query: str, limit: int = 6) -> list[SearchResult]:
        problems = []
        for name, engine in (("metasearch", self._metasearch), ("wikipedia", self._wikipedia)):
            try:
                results = engine(query, limit)
            except Exception as error:  # noqa: BLE001  one engine failing shouldn't end the search
                problems.append(f"{name}: {error}")
                continue
            if results:
                return results
        if problems:
            raise WebError("search failed (" + "; ".join(problems) + ")")
        return []

    def fetch(self, url: str, max_chars: int = 10000) -> Page:
        body, final_url, headers = self._get(url)
        kind = headers.get_content_type()
        try:
            text = body.decode(headers.get_content_charset() or "utf-8", "replace")
        except LookupError:
            text = body.decode("utf-8", "replace")
        looks_like_html = text.lstrip()[:15].lower().startswith(("<!doctype", "<html"))
        if kind in ("text/html", "application/xhtml+xml") or looks_like_html:
            title, content, links = extract(text, final_url)
        elif kind.startswith("text/") or kind in ("application/json", "application/xml"):
            title, content, links = final_url, text, []
        else:
            raise WebError(f"Haven can't read {kind} files, only web pages and text")
        content = _tidy(content)
        if len(content) > max_chars:
            content = content[:max_chars].rsplit(" ", 1)[0] + " […]"
        return Page(final_url, title.strip() or final_url, content, links)

    def _metasearch(self, query: str, limit: int) -> list[SearchResult]:
        try:
            from ddgs import DDGS
        except ImportError:
            return []
        rows = DDGS(timeout=int(self.timeout)).text(query, max_results=limit, safesearch=self.safesearch)
        return [
            SearchResult(row.get("title") or row["href"], row["href"], row.get("body") or "")
            for row in rows
            if row.get("href")
        ]

    def _wikipedia(self, query: str, limit: int) -> list[SearchResult]:
        params = {"action": "query", "list": "search", "srsearch": query, "srlimit": limit, "format": "json"}
        body, _, _ = self._get("https://en.wikipedia.org/w/api.php?" + urlencode(params), "application/json")
        hits = json.loads(body).get("query", {}).get("search", [])
        return [
            SearchResult(
                hit["title"],
                "https://en.wikipedia.org/wiki/" + quote(hit["title"].replace(" ", "_")),
                html.unescape(re.sub(r"<[^>]+>", "", hit.get("snippet", ""))),
            )
            for hit in hits
        ]

    def _get(self, url: str, accept: str = "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5"):
        check_address(url)
        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Accept": accept, "Accept-Language": "en;q=1.0, *;q=0.5"}
        )
        opener = urllib.request.build_opener(_CheckedRedirects)
        try:
            with opener.open(request, timeout=self.timeout) as response:
                return response.read(MAX_BYTES), response.geturl(), response.headers
        except urllib.error.HTTPError as error:
            raise WebError(f"the site answered with an error ({error.code} {error.reason})") from None
        except urllib.error.URLError as error:
            raise WebError(f"couldn't reach the site ({error.reason})") from None
        except (TimeoutError, ConnectionError, http.client.HTTPException) as error:
            raise WebError(f"couldn't read the site ({error})") from None


class _CheckedRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, ANN201
        check_address(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def check_address(url: str) -> None:
    """Refuse anything but public http(s) addresses, so a page can't steer Haven into your own network."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise WebError("only http and https addresses can be opened")
    host = (parts.hostname or "").rstrip(".").lower()
    if not host:
        raise WebError("that isn't a complete web address")
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal", ".lan", ".home.arpa")):
        raise WebError(PRIVATE)
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, parts.port or 443, proto=socket.IPPROTO_TCP)
        except (socket.gaierror, UnicodeError):
            if urllib.request.getproxies().get(parts.scheme):
                return  # names are resolved by the proxy, which can't reach this machine's network
            raise WebError(f"couldn't find {host}") from None
        addresses = [ipaddress.ip_address(info[4][0].split("%")[0]) for info in infos]
    for address in addresses:
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        if not address.is_global or address.is_multicast:
            raise WebError(PRIVATE)


def find_urls(text: str) -> set[str]:
    return {normalize(match.rstrip(".,;:!?")) for match in _URL.findall(text)}


def normalize(url: str) -> str:
    parts = urlsplit(url.strip())
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), parts.query, ""))


class _Extractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "form", "template", "iframe"}
    BREAK = {"p", "div", "section", "br", "tr", "table", "blockquote", "pre", "dd", "dt", "figcaption", "hr"}
    HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title = ""
        self.everything: list[str] = []
        self.main: list[str] = []
        self.links: list[tuple[str, str]] = []
        self._skipping = self._in_main = 0
        self._in_title = False
        self._href: str | None = None
        self._anchor: list[str] = []

    def _write(self, text: str) -> None:
        self.everything.append(text)
        if self._in_main:
            self.main.append(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP:
            self._skipping += 1
        elif tag == "title":
            self._in_title = True
        elif tag in ("main", "article"):
            self._in_main += 1
        if self._skipping:
            return
        if tag in self.BREAK:
            self._write("\n")
        elif tag in self.HEADINGS:
            self._write("\n\n## ")
        elif tag == "li":
            self._write("\n- ")
        elif tag == "a":
            self._href = dict(attrs).get("href")
            self._anchor = []

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self._skipping:
            self._skipping -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in ("main", "article") and self._in_main:
            self._in_main -= 1
        elif tag in self.HEADINGS or tag == "p":
            self._write("\n")
        elif tag == "a" and self._href is not None:
            self._add_link(self._href, " ".join("".join(self._anchor).split()))
            self._href = None

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skipping:
            self._write(data)
            if self._href is not None:
                self._anchor.append(data)

    def _add_link(self, href: str, text: str) -> None:
        if len(text) < 3 or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            return
        url = urljoin(self.base_url, href)
        if urlsplit(url).scheme in ("http", "https") and normalize(url) != normalize(self.base_url):
            if all(normalize(url) != normalize(known) for _, known in self.links):
                self.links.append((text[:80], url))


def extract(markup: str, base_url: str) -> tuple[str, str, list[tuple[str, str]]]:
    """A page's title, its readable text (the main article if it has one), and its links."""
    parser = _Extractor(base_url)
    parser.feed(markup)
    parser.close()
    main = "".join(parser.main)
    text = main if len(main.strip()) > 400 else "".join(parser.everything)
    return " ".join(parser.title.split()), text, parser.links[:20]


def _tidy(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    text = "\n".join(line for line in lines if line not in ("-", "##"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()
