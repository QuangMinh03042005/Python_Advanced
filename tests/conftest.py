from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def temp_data_dir(request: pytest.FixtureRequest) -> Iterator[Path]:
    """Workspace-local temp directory.

    The Windows sandbox blocks pytest's default system temp directory, so tests
    use a normal directory under the project instead.
    """
    path = Path(".test-tmp") / request.node.name
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
