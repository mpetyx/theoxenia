"""Tests for bounded HTTP byte-range access using synthetic local data."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from xenia.radio.remote import RangeMetrics, open_counted_http_file

TEST_DATA = bytes(range(256)) * 64


class _RangeServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], payload: bytes) -> None:
        self.payload = payload
        self.ranges: list[tuple[int, int]] = []
        super().__init__(address, _RangeHandler)


class _RangeHandler(BaseHTTPRequestHandler):
    server: _RangeServer
    protocol_version = "HTTP/1.1"

    def do_HEAD(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.server.payload)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self) -> None:
        header = self.headers.get("Range")
        if header is None:
            self.send_error(400, "Range header required")
            return
        try:
            unit, interval = header.split("=", maxsplit=1)
            start_text, end_text = interval.split("-", maxsplit=1)
            start = int(start_text)
            end = int(end_text)
        except (ValueError, TypeError):
            self.send_error(400, "invalid range")
            return
        if unit != "bytes" or start < 0 or end < start or end >= len(self.server.payload):
            self.send_error(416, "range not satisfiable")
            return

        self.server.ranges.append((start, end + 1))
        payload = self.server.payload[start : end + 1]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(self.server.payload)}")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_args: object) -> None:
        return


@contextmanager
def _range_server(payload: bytes) -> Iterator[tuple[str, _RangeServer]]:
    server = _RangeServer(("127.0.0.1", 0), payload)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}/synthetic.bin", server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_reads_are_seekable_and_only_network_fetches_are_counted() -> None:
    with _range_server(TEST_DATA) as (url, server):
        with open_counted_http_file(url, block_size=1024) as (remote_file, metrics):
            assert remote_file.read(100) == TEST_DATA[:100]
            remote_file.seek(2048 + 100)
            assert remote_file.read(1500) == TEST_DATA[2148:3648]
            remote_file.seek(0)
            assert remote_file.read(100) == TEST_DATA[:100]

            assert isinstance(metrics, RangeMetrics)
            assert metrics.source_size_bytes == len(TEST_DATA)
            assert metrics.bytes_transferred == sum(item.bytes_received for item in metrics.ranges)
            assert all(item.end_exclusive <= len(TEST_DATA) for item in metrics.ranges)
            assert metrics.bytes_transferred <= metrics.max_transfer_bytes

        assert len(server.ranges) == len(metrics.ranges)
        assert [item.bytes_received for item in metrics.ranges] == [
            end - start for start, end in server.ranges
        ]
        assert metrics.bytes_transferred < len(TEST_DATA)


def test_rejects_non_loopback_http_before_connecting() -> None:
    with (
        pytest.raises(ValueError, match="loopback"),
        open_counted_http_file("http://example.com/data"),
    ):
        pass


def test_rejects_credential_bearing_url() -> None:
    with (
        pytest.raises(ValueError, match="credentials"),
        open_counted_http_file("https://user:password@example.com/data"),
    ):
        pass


def test_validates_block_and_transfer_limits() -> None:
    with (
        pytest.raises(ValueError, match="block_size"),
        open_counted_http_file("https://example.com/data", block_size=0),
    ):
        pass
    with (
        pytest.raises(ValueError, match="max_transfer_bytes"),
        open_counted_http_file("https://example.com/data", max_transfer_bytes=0),
    ):
        pass
