"""Test-wide guards: no network access and the repository root on sys.path."""
from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class NetworkAccessBlocked(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Every test runs without network access: any attempt to open a socket connection fails loudly."""

    def _blocked(*args, **kwargs):
        raise NetworkAccessBlocked("network access is not allowed in tests")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    yield
