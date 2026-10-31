from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = PROJECT_ROOT / "reports"
RESULTS_PATH = REPORTS_ROOT / "results.csv"

DATA_DIR = REPORTS_ROOT / "data"
BASELINE_DIR = REPORTS_ROOT / "baselines" / "tfidf"
TEXTCNN_SINGLE_DIR = REPORTS_ROOT / "textcnn" / "single"
TEXTCNN_MULTITASK_DIR = REPORTS_ROOT / "textcnn" / "multitask"
TEXTCNN_DISTILL_DIR = REPORTS_ROOT / "textcnn" / "distilled"
PHOBERT_FROZEN_DIR = REPORTS_ROOT / "phobert" / "frozen"
PHOBERT_KEYWORD_DIR = REPORTS_ROOT / "phobert" / "keyword"
DISTILLATION_DIR = REPORTS_ROOT / "distillation"
EVALUATION_DIR = REPORTS_ROOT / "evaluation"
PLANS_DIR = REPORTS_ROOT / "plans"


def resolve_results_path(reports_dir: str | Path) -> Path:
    """Keep one global results.csv for normal runs, but isolate custom temp runs."""
    target = Path(reports_dir).resolve()
    try:
        target.relative_to(REPORTS_ROOT.resolve())
    except ValueError:
        return target / "results.csv"
    return RESULTS_PATH
