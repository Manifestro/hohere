"""HTTP Range reader: exact reads, retries with backoff and a stall guard.

The Libri-Light archives are 9 GB / 321 GB / 3.3 TB, so nothing is ever downloaded
whole: every byte is fetched with a `Range` request. A single stream can stall for
minutes while the rest of the world is fine, so every request is retried and aborted
when it trickles in below `MIN_SPEED` for `STALL_WINDOW` seconds; the retry resumes
from the last byte actually received, never from the start of the read.

`opener(offset, length) -> Iterable[bytes]` is the transport and can be swapped in
tests, so the retry/stall logic is exercised without a socket.
"""

from __future__ import annotations

import http.client
import logging
import threading
import time
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from urllib.parse import urlsplit

log = logging.getLogger(__name__)

MIN_SPEED = 256 * 1024  # bytes/s; below this for STALL_WINDOW seconds a request is aborted
STALL_WINDOW = 20.0  # seconds
RETRIES = 5
BACKOFF = 0.5  # seconds, doubled after every failed attempt
CHUNK = 1 << 20  # bytes requested from the socket at a time
TIMEOUT = 30.0
USER_AGENT = "hohere/2"

Opener = Callable[[int, int], Iterable[bytes]]


class RangeError(RuntimeError):
    """A Range request could not be satisfied."""


class _KeepAlive:
    """One HTTP(S) connection per thread, reused across Range requests.

    A fresh TLS handshake per 4 KiB header costs more than the read; archive walking
    does tens of thousands of those, so the connection is kept and only dropped after
    an error or an aborted body (a body must be drained before the socket is reused).
    """

    def __init__(self, url: str, timeout: float) -> None:
        parts = urlsplit(url)
        self.secure = parts.scheme != "http"
        self.host = parts.hostname or ""
        self.port = parts.port
        self.path = parts.path + (f"?{parts.query}" if parts.query else "")
        self.timeout = timeout
        self._conn: http.client.HTTPConnection | None = None

    def connection(self) -> http.client.HTTPConnection:
        if self._conn is None:
            factory = (
                http.client.HTTPSConnection if self.secure else http.client.HTTPConnection
            )
            self._conn = factory(self.host, self.port, timeout=self.timeout)
        return self._conn

    def reset(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def get(self, offset: int, length: int) -> http.client.HTTPResponse:
        headers = {
            "Range": f"bytes={offset}-{offset + length - 1}",
            "Accept-Encoding": "identity",
            "User-Agent": USER_AGENT,
        }
        try:
            conn = self.connection()
            conn.request("GET", self.path, headers=headers)
            return conn.getresponse()
        except Exception:
            self.reset()
            raise


class RangeReader:
    """`read(offset, length)` and `stream(offset, length)` over HTTPS Range requests.

    The object never requests bytes past its end: `length` is clamped to the object
    size and a read that starts at (or after) the end returns nothing. Safe to share
    between threads: the only mutable state is the cached size.
    """

    def __init__(
        self,
        url: str,
        *,
        size: int | None = None,
        opener: Opener | None = None,
        chunk: int = CHUNK,
        retries: int = RETRIES,
        backoff: float = BACKOFF,
        min_speed: float = MIN_SPEED,
        stall_window: float = STALL_WINDOW,
        timeout: float = TIMEOUT,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.url = url
        self.chunk = chunk
        self.retries = retries
        self.backoff = backoff
        self.min_speed = min_speed
        self.stall_window = stall_window
        self.timeout = timeout
        self.sleep = sleep
        self.clock = clock
        self._size = size
        self._opener = opener
        self._local = threading.local()  # one keep-alive connection per thread

    # -- transport ---------------------------------------------------------------
    def _client(self) -> _KeepAlive:
        client = getattr(self._local, "client", None)
        if client is None:
            client = _KeepAlive(self.url, self.timeout)
            self._local.client = client
        return client

    def _open(self, offset: int, length: int, chunk: int | None = None) -> Iterator[bytes]:
        if self._opener is not None:
            yield from self._opener(offset, length)
            return
        step = chunk or self.chunk
        client = self._client()
        try:
            response = client.get(offset, length)
        except Exception:
            client.reset()
            raise
        status = getattr(response, "status", 206)
        if status not in (200, 206):
            response.close()
            client.reset()
            raise RangeError(f"unexpected HTTP status {status} at {offset}")
        if status == 200 and offset != 0:
            response.close()
            client.reset()
            raise RangeError(f"server ignored the Range header for {offset}")
        received = 0
        done = False
        try:
            while received < length:
                piece = response.read(min(step, length - received))
                if not piece:
                    break
                received += len(piece)
                yield piece
            done = received >= length
        finally:
            # an aborted body leaves the socket in an unknown state: drop it
            if not done:
                client.reset()
            response.close()

    def size(self) -> int:
        """Total object size, from a HEAD request the first time, then cached."""

        if self._size is None:
            delay = self.backoff
            last: Exception | None = None
            for attempt in range(self.retries + 1):
                try:
                    self._size = self._head()
                    break
                except Exception as exc:  # every transport error is worth a retry
                    last = exc
                    if attempt == self.retries:
                        raise RangeError(
                            f"cannot stat {self.url} after {attempt + 1} tries: {exc}"
                        ) from exc
                    log.warning("HEAD %s failed (%s), retry %d", self.url, exc, attempt + 1)
                    self.sleep(delay)
                    delay *= 2
            if self._size is None:  # pragma: no cover - defensive
                raise RangeError(f"cannot stat {self.url}: {last}")
        return self._size

    def _head(self) -> int:
        request = urllib.request.Request(
            self.url, method="HEAD", headers={"User-Agent": USER_AGENT}
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return int(response.headers["Content-Length"])

    # -- reads -------------------------------------------------------------------
    def _clamp(self, offset: int, length: int) -> tuple[int, int]:
        if offset < 0 or length < 0:
            raise ValueError(f"negative range: offset={offset} length={length}")
        size = self.size()
        if offset >= size:
            return size, size
        return offset, min(offset + length, size)

    def stream(self, offset: int, length: int, chunk: int | None = None) -> Iterator[bytes]:
        """Yield `length` bytes at `offset`, retrying transport failures and stalls.

        `chunk` is the socket read size (1 MiB by default), not the granularity of
        the retry: a retry always resumes at the last byte handed to the caller.
        """
        pos, end = self._clamp(offset, length)
        delay = self.backoff
        attempts = 0
        while pos < end:
            started = self.clock()
            received = 0
            failure: str | None = None
            try:
                for piece in self._open(pos, end - pos, chunk):
                    if not piece:
                        continue
                    if len(piece) > end - pos:
                        piece = piece[: end - pos]
                    yield piece
                    pos += len(piece)
                    received += len(piece)
                    if pos >= end:
                        return
                    elapsed = self.clock() - started
                    if elapsed > self.stall_window and received / elapsed < self.min_speed:
                        raise RangeError(
                            f"stalled at {received / elapsed / 1024:.0f} KiB/s for {elapsed:.0f} s"
                        )
            except Exception as exc:  # every transport error is worth a retry
                failure = str(exc) or type(exc).__name__
            if pos >= end:
                return
            if failure is None:
                failure = f"short read, {end - pos} bytes missing"
            attempts += 1
            if attempts > self.retries:
                raise RangeError(
                    f"cannot read {length} bytes at {offset} of {self.url} "
                    f"after {attempts} attempts: {failure}"
                )
            log.warning(
                "range read %s at %d failed (%s), retry %d/%d",
                self.url,
                pos,
                failure,
                attempts,
                self.retries,
            )
            self.sleep(delay)
            delay *= 2

    def read(self, offset: int, length: int) -> bytes:
        """Exactly `length` bytes at `offset`, or fewer only at the end of the object."""
        start, end = self._clamp(offset, length)
        if start >= end:
            return b""
        data = b"".join(self.stream(offset, end - start))
        if len(data) != end - start:  # pragma: no cover - stream guarantees this
            raise RangeError(f"short read at {offset}: {len(data)} of {end - start} bytes")
        return data
