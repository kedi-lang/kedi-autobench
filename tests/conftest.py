from __future__ import annotations

from collections.abc import Iterator

import pytest
from kedi.telemetry import reset_backend


@pytest.fixture(autouse=True)
def clean_kedi_backend() -> Iterator[None]:
    reset_backend()
    try:
        yield
    finally:
        reset_backend()
