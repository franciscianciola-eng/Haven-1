from __future__ import annotations

import email.message
import json

import pytest

from haven import web
from haven.web import SearchResult, Web, WebError, check_address, extract, find_urls, normalize

ARTICLE = (
    "<!doctype html><html><head><title> Octopus - Wikipedia </title><style>.x{}</style></head><body>"
    '<nav><a href="/wiki/Main_Page">Main page</a></nav><main><h1>Octopus</h1>'
    "<p>The octopus is a soft-bodied, eight-limbed mollusc.</p>"
    '<p>Octopuses <b>sleep</b> in two stages; see <a href="/wiki/Sleep">sleep in animals</a>.</p>'
    "<ul><li>Intelligent</li><li>Camouflage</li></ul><script>var tracking = 1;</script>"
    f"<p>{'More about octopuses. ' * 20}</p></main>"
    '<footer><a href="https://example.org/legal">Legal notice</a> Footer text</footer></body></html>'
)


def headers(content_type: str) -> email.message.Message:
    message = email.message.Message()
    message["Content-Type"] = content_type
    return message


def test_extract_keeps_the_article_and_drops_the_furniture():
    title, text, links = extract(ARTICLE, "https://en.wikipedia.org/wiki/Octopus")
    assert title == "Octopus - Wikipedia"
    assert "eight-limbed mollusc" in text and "\n- Intelligent" in text
    assert "Main page" not in text and "tracking" not in text and "Footer text" not in text
    assert links == [("sleep in animals", "https://en.wikipedia.org/wiki/Sleep")]


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:11434/api/tags",
        "http://localhost:8080/",
        "http://192.168.1.1/admin",
        "http://10.0.0.5/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "http://[::ffff:10.0.0.1]/",
        "http://100.64.0.1/",
        "http://printer.local/",
        "file:///etc/passwd",
        "ftp://example.com/file",
        "https:///nohost",
    ],
)
def test_private_local_and_odd_addresses_are_refused(url):
    with pytest.raises(WebError):
        check_address(url)


def test_public_addresses_are_allowed():
    check_address("https://8.8.8.8/")
    check_address("https://[2606:4700:4700::1111]/dns-query")


def test_redirects_into_the_local_network_are_refused():
    with pytest.raises(WebError):
        web._CheckedRedirects().redirect_request(None, None, 302, "Found", {}, "http://127.0.0.1/admin")


def test_addresses_are_found_and_normalized():
    found = find_urls("see https://en.wikipedia.org/wiki/Octopus. and (https://Example.com/a/) too")
    assert found == {"https://en.wikipedia.org/wiki/Octopus", "https://example.com/a"}
    assert normalize("HTTPS://Example.com/a/?q=1#top") == "https://example.com/a?q=1"


def test_fetch_reads_pages_and_plain_text(monkeypatch):
    responses = {
        "https://site.test/page": (ARTICLE.encode(), "https://site.test/page", headers("text/html; charset=utf-8")),
        "https://site.test/notes.txt": (b"plain words", "https://site.test/notes.txt", headers("text/plain")),
        "https://site.test/paper.pdf": (b"%PDF-1.7", "https://site.test/paper.pdf", headers("application/pdf")),
    }
    reader = Web()
    monkeypatch.setattr(reader, "_get", lambda url, accept=None: responses[url])

    page = reader.fetch("https://site.test/page")
    assert page.title == "Octopus - Wikipedia" and "eight-limbed mollusc" in page.text
    assert "\n\n\n" not in page.text
    assert reader.fetch("https://site.test/notes.txt").text == "plain words"
    with pytest.raises(WebError, match="can't read application/pdf"):
        reader.fetch("https://site.test/paper.pdf")
    short = reader.fetch("https://site.test/page", max_chars=100)
    assert len(short.text) <= 104 and short.text.endswith("[…]")


def test_search_falls_back_to_wikipedia(monkeypatch):
    reader = Web()

    def broken(query, limit):
        raise RuntimeError("rate limited")

    body = json.dumps(
        {
            "query": {
                "search": [{"title": "Octopus", "snippet": 'The <span class="searchmatch">octopus</span> &amp; kin'}]
            }
        }
    ).encode()
    monkeypatch.setattr(reader, "_metasearch", broken)
    monkeypatch.setattr(reader, "_get", lambda url, accept=None: (body, url, headers("application/json")))
    assert reader.search("octopus") == [
        SearchResult("Octopus", "https://en.wikipedia.org/wiki/Octopus", "The octopus & kin")
    ]


def test_search_says_so_when_every_engine_fails(monkeypatch):
    reader = Web()

    def broken(query, limit):
        raise RuntimeError("offline")

    monkeypatch.setattr(reader, "_metasearch", broken)
    monkeypatch.setattr(reader, "_wikipedia", broken)
    with pytest.raises(WebError, match="metasearch: offline; wikipedia: offline"):
        reader.search("anything")


def test_metasearch_results_are_used_first(monkeypatch):
    class FakeDDGS:
        def __init__(self, timeout):
            pass

        def text(self, query, max_results, safesearch):
            assert (query, max_results, safesearch) == ("octopus", 6, "off")
            return [
                {"title": "Octopus facts", "href": "https://a.test/octopus", "body": "Eight arms."},
                {"title": "no link"},
            ]

    monkeypatch.setattr("ddgs.DDGS", FakeDDGS)
    assert Web().search("octopus") == [SearchResult("Octopus facts", "https://a.test/octopus", "Eight arms.")]
