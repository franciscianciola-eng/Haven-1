"""Haven's window on the internet: careful, read-only downloads for its reading.

Only public http(s) addresses are opened (never anything on your own network, even after
a redirect), downloads are capped in size, and each site is asked for at most one thing
per second. No API keys are needed for anything Haven reads.
"""

from __future__ import annotations

import contextlib
import functools
import http.client
import ipaddress
import json
import socket
import ssl
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlencode, urlsplit

from . import __version__

USER_AGENT = f"Haven/{__version__} (a learning program; https://github.com/franciscianciola-eng/Haven-1)"
PRIVATE = "that address is on a private or local network, so Haven won't open it"


class WebError(Exception):
    pass


class Web:
    def __init__(self, timeout: float = 30, delay: float = 1.0, allow_private: bool = False):
        self.timeout = timeout
        self.delay = delay
        self.allow_private = allow_private  # only for tests against a server on this machine
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def get(self, url: str, max_bytes: int = 5_000_000, accept: str = "*/*") -> bytes:
        """Download up to max_bytes (the rest is left unread)."""
        self._check(url)
        self._wait_turn(url)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
        handlers = [_CheckedRedirects(self), urllib.request.HTTPSHandler(context=tls())]
        if self.allow_private:
            handlers.append(urllib.request.ProxyHandler({}))
        opener = urllib.request.build_opener(*handlers)
        try:
            with opener.open(request, timeout=self.timeout) as response:
                chunks, total = [], 0
                while total < max_bytes:
                    chunk = response.read(min(1 << 16, max_bytes - total))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
                return b"".join(chunks)
        except urllib.error.HTTPError as error:
            raise WebError(f"{url} answered with an error ({error.code} {error.reason})") from None
        except urllib.error.URLError as error:
            raise WebError(f"couldn't reach {url} ({error.reason})") from None
        except (TimeoutError, ConnectionError, http.client.HTTPException) as error:
            raise WebError(f"couldn't read {url} ({error})") from None

    @contextlib.contextmanager
    def open(self, url: str, accept: str = "*/*"):
        """A download to read as it comes, for files too big to hold at once (an encyclopedia)."""
        self._check(url)
        self._wait_turn(url)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
        handlers = [_CheckedRedirects(self), urllib.request.HTTPSHandler(context=tls())]
        if self.allow_private:
            handlers.append(urllib.request.ProxyHandler({}))
        try:
            response = urllib.request.build_opener(*handlers).open(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            raise WebError(f"{url} answered with an error ({error.code} {error.reason})") from None
        except urllib.error.URLError as error:
            raise WebError(f"couldn't reach {url} ({error.reason})") from None
        except (TimeoutError, ConnectionError, http.client.HTTPException) as error:
            raise WebError(f"couldn't read {url} ({error})") from None
        try:
            yield response
        except (TimeoutError, ConnectionError, http.client.HTTPException) as error:
            raise WebError(f"the download from {url} broke off ({error})") from None
        finally:
            response.close()

    def text(self, url: str, max_bytes: int = 5_000_000) -> str:
        return self.get(url, max_bytes, "text/plain, */*;q=0.5").decode("utf-8", "replace")

    def json(self, url: str, params: dict | None = None, max_bytes: int = 50_000_000) -> object:
        if params:
            url += ("&" if "?" in url else "?") + urlencode(params)
        body = self.get(url, max_bytes, "application/json")
        try:
            return json.loads(body)
        except json.JSONDecodeError as error:
            raise WebError(f"{url} didn't send readable JSON ({error})") from None

    def _wait_turn(self, url: str) -> None:
        host = urlsplit(url).hostname or ""
        with self._lock:
            wait = self._last.get(host, 0.0) + self.delay - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last[host] = time.monotonic()

    def _check(self, url: str) -> None:
        if self.allow_private:
            if urlsplit(url).scheme not in ("http", "https"):
                raise WebError("only http and https addresses can be opened")
            return
        check_address(url)


@functools.cache
def tls() -> ssl.SSLContext:
    """How it checks that a website is the one it asked for: against this computer's certificates, and certifi's
    too where it's installed (some Pythons, like the ones uv installs on a Mac, can't find the computer's)."""
    context = ssl.create_default_context()
    try:
        import certifi

        context.load_verify_locations(certifi.where())
    except (ImportError, OSError):
        pass
    return context


class _CheckedRedirects(urllib.request.HTTPRedirectHandler):
    def __init__(self, web: Web):
        super().__init__()
        self.web = web

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.web._check(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def check_address(url: str) -> None:
    """Refuse anything but public http(s) addresses, so nothing can steer Haven into your own network."""
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
