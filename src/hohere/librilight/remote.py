"""HTTPS Range reads of the archive: a stall guard and retries.

One request has ~0.5 s of latency and 16 parallel streams reach ~366 MB/s, but a
single stream sometimes stalls for minutes, so every read is retried and aborted when
it trickles in below `MIN_SPEED` for `WINDOW` seconds. The transfer itself is a plain
callable of `(offset, n)` returning chunks, which keeps the retry loop testable
without a socket.
"""

from __future__ import annotations

import logging
import time
import urllib.request
from collections.abc import Callable, Iterable

log = logging.getLogger(__name__)

MIN_SPEED = 1_000_000  # bytes/s
WINDOW = 20.0  # seconds below MIN_SPEED before a transfer is aborted
RETRIES = 5
BACKOFF = 2.0
CHUNK = 1 << 16
USER_AGENT = "hohere-librilight/1"

Stream = Callable[[int, int], Iterable[bytes]]


class RangeError(RuntimeError):
    """A Range request could not be satisfied."""


class _Retryable(Exception):
    pass


def _read_once(
    stream: Stream, offset: int, n: int, min_speed: float, window: float, clock: Callable[[], float]
) -> bytes:
    started = clock()
    parts: list[bytes] = []
    total = 0
    for chunk in stream(offset, n):
        parts.append(chunk)
        total += len(chunk)
        if total >= n:
            break
        elapsed = clock() - started
        if elapsed > window and total / elapsed < min_speed:
            raise _Retryable(f"trickling at {total / elapsed / 1e6:.2f} MB/s for {elapsed:.0f} s")
    data = b"".join(parts)
    if len(data) < n:
        raise _Retryable(f"short read: {len(data)} of {n} bytes")
    return data[:n]


def read_with_retry(
    stream: Stream,
    offset: int,
    n: int,
    *,
    retries: int = RETRIES,
    min_speed: float = MIN_SPEED,
    window: float = WINDOW,
    backoff: float = BACKOFF,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> bytes:
    """Read exactly `n` bytes at `offset`, retrying transport failures and stalls."""
    delay = backoff
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return _read_once(stream, offset, n, min_speed, window, clock)
        except Exception as e:  # every transport error is worth a retry
            last = e
            if attempt == retries:
                break
            log.warning("range read at %d failed (%s), retry %d", offset, e, attempt + 1)
            sleep(delay)
            delay *= 2
    raise RangeError(f"cannot read {n} bytes at {offset} after {retries + 1} tries: {last}")


class RangeReader:
    """`read(offset, n)` over HTTPS Range requests; safe to share between threads."""

    def __init__(
        self, url: str, *, timeout: float = WINDOW, chunk: int = CHUNK, retries: int = RETRIES
    ) -> None:
        self.url = url
        self.timeout = timeout
        self.chunk = chunk
        self.retries = retries

    def _retry(self, call: Callable[[], bytes], what: str) -> bytes:
        delay = BACKOFF
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return call()
            except Exception as e:  # every transport error is worth a retry
                last = e
                if attempt == self.retries:
                    break
                log.warning("%s failed (%s), retry %d", what, e, attempt + 1)
                time.sleep(delay)
                delay *= 2
        raise RangeError(f"{what} failed after {self.retries + 1} tries: {last}")

    def _stream(self, offset: int, n: int) -> Iterable[bytes]:
        request = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={offset}-{offset + n - 1}", "User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            if getattr(response, "status", 206) == 200 and offset != 0:
                raise RangeError(f"server ignored the Range header for {offset}")
            while chunk := response.read(self.chunk):
                yield chunk

    def read(self, offset: int, n: int) -> bytes:
        return read_with_retry(self._stream, offset, n, retries=self.retries)

    def size(self) -> int:
        """Total size from a HEAD request."""

        def head() -> bytes:
            request = urllib.request.Request(
                self.url, method="HEAD", headers={"User-Agent": USER_AGENT}
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.headers["Content-Length"].encode()

        return int(self._retry(head, "HEAD"))
