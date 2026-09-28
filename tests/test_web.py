import pytest

from haven.web import Web, WebError, check_address


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/x",
        "http://127.0.0.1/",
        "http://10.0.0.5/",
        "http://192.168.1.1/",
        "http://[::1]/",
        "file:///etc/passwd",
        "ftp://example.com/",
        "http://printer.local/",
    ],
)
def test_private_and_odd_addresses_are_refused(url):
    with pytest.raises(WebError):
        check_address(url)


def test_downloads_are_capped(internet):
    web = Web(delay=0, allow_private=True)
    assert len(web.get(internet["tinystories"], max_bytes=1000)) == 1000
    assert "Once upon a time" in web.text(internet["tinystories"], 5000)
    pages = web.json(internet["simplewiki"], {"action": "query", "generator": "random"})
    assert len(pages["query"]["pages"]) == 20


def test_a_normal_web_refuses_this_machine(internet):
    with pytest.raises(WebError):
        Web(delay=0).get(internet["tinystories"])
