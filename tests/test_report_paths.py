from __future__ import annotations

from src.report_paths import (
    REPORTS_ROOT,
    RESULTS_PATH,
    TEXTCNN_SINGLE_DIR,
    resolve_results_path,
)


def test_reports_use_global_results_for_structured_dirs() -> None:
    assert TEXTCNN_SINGLE_DIR.is_relative_to(REPORTS_ROOT)
    assert resolve_results_path(TEXTCNN_SINGLE_DIR) == RESULTS_PATH


def test_custom_temp_reports_keep_local_results() -> None:
    temp_dir = REPORTS_ROOT.parent / ".tmp" / "custom_reports"
    assert resolve_results_path(temp_dir) == temp_dir / "results.csv"
