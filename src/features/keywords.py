from __future__ import annotations

from collections.abc import Sequence

import numpy as np

TOPIC_ORDER = ("lecturer", "training_program", "facility", "others")

TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "lecturer": (
        "giảng_viên",
        "giáo_viên",
        "thầy",
        "cô",
        "thầy_cô",
        "giảng_dạy",
        "truyền_đạt",
        "bài_giảng",
        "phương_pháp",
        "dạy",
    ),
    "training_program": (
        "chương_trình",
        "đào_tạo",
        "môn_học",
        "bài_tập",
        "giáo_trình",
        "tài_liệu",
        "đề_cương",
        "kiểm_tra",
        "thực_hành",
        "seminar",
        "đồ_án",
        "nội_dung",
    ),
    "facility": (
        "cơ_sở_vật_chất",
        "cơ_sở",
        "vật_chất",
        "phòng_học",
        "phòng_máy",
        "máy_chiếu",
        "wifi",
        "máy_tính",
        "thiết_bị",
        "ánh_sáng",
        "điều_hòa",
        "âm_thanh",
        "máy_lạnh",
    ),
    "others": (
        "áp_lực",
        "sức_khỏe",
        "thời_gian",
        "học_phí",
        "dịch_vụ",
        "hỗ_trợ",
    ),
}


def keyword_feature_names() -> list[str]:
    names: list[str] = []
    for topic in TOPIC_ORDER:
        names.extend(
            [
                f"{topic}_present",
                f"{topic}_density",
                f"{topic}_last3",
            ]
        )
    return names


def keyword_hits(tokens: Sequence[str]) -> dict[str, list[str]]:
    token_set = set(tokens)
    return {
        topic: [keyword for keyword in keywords if keyword in token_set]
        for topic, keywords in TOPIC_KEYWORDS.items()
    }


def extract_topic_keyword_features(tokens: Sequence[str]) -> np.ndarray:
    """Return presence, density, and last-3 presence for each topic."""
    token_list = list(tokens)
    token_set = set(token_list)
    token_count = max(len(token_list), 1)
    last_tokens = token_list[-3:]

    features: list[float] = []
    for topic in TOPIC_ORDER:
        hits = [keyword for keyword in TOPIC_KEYWORDS[topic] if keyword in token_set]
        features.extend(
            [
                1.0 if hits else 0.0,
                len(hits) / token_count,
                1.0 if any(keyword in last_tokens for keyword in hits) else 0.0,
            ]
        )
    return np.asarray(features, dtype=np.float32)
