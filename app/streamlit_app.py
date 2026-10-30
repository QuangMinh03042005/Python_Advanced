from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import DEFAULT_CHECKPOINT_PATH, PhoBERTMultiTaskPredictor  # noqa: E402

SENTIMENT_VN = {
    "negative": "Tiêu cực",
    "neutral": "Trung tính",
    "positive": "Tích cực",
}
TOPIC_VN = {
    "lecturer": "Giảng viên",
    "training_program": "Chương trình đào tạo",
    "facility": "Cơ sở vật chất",
    "others": "Khác",
}
EXAMPLES = {
    "Tích cực": "Thầy dạy rất nhiệt tình, bài tập vừa sức và dễ hiểu.",
    "Tiêu cực": "Cơ sở vật chất còn yếu, phòng học thiếu máy chiếu.",
    "Trung tính": "Môn học bình thường, không có gì nổi bật.",
}


st.set_page_config(
    page_title="Phân tích phản hồi sinh viên",
    page_icon="💬",
    layout="wide",
)


@st.cache_resource(show_spinner="Đang tải PhoBERT và checkpoint...")
def load_predictor() -> PhoBERTMultiTaskPredictor:
    return PhoBERTMultiTaskPredictor(DEFAULT_CHECKPOINT_PATH)


try:
    predictor = load_predictor()
except Exception as exc:  # pragma: no cover - UI guard
    st.error(f"Không tải được mô hình: {exc}")
    st.info("Hãy chắc chắn đã chạy `python -m src.train_phobert_multitask` trước.")
    st.stop()


def render_probabilities(title: str, probabilities: dict[str, float], labels_vn: dict[str, str]) -> None:
    st.markdown(f"**{title}**")
    for label, probability in sorted(probabilities.items(), key=lambda item: item[1], reverse=True):
        value = min(max(float(probability), 0.0), 1.0)
        st.progress(value, text=f"{labels_vn[label]}: {value:.1%}")


def suggestion_for(sentiment: str, topic: str) -> str | None:
    if sentiment != "negative":
        return None
    suggestions = {
        "facility": "Cần xem lại cơ sở vật chất và thiết bị học tập.",
        "lecturer": "Cần phản hồi và trao đổi với giảng viên.",
        "training_program": "Cần xem lại chương trình đào tạo và bài tập.",
        "others": "Cần ghi nhận thêm để làm rõ vấn đề.",
    }
    return suggestions.get(topic)


def render_prediction(result: dict[str, Any]) -> None:
    sentiment = result["sentiment"]
    topic = result["topic"]
    sentiment_label = SENTIMENT_VN[sentiment["label"]]
    topic_label = TOPIC_VN[topic["label"]]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Cảm xúc", sentiment_label, f"{sentiment['confidence']:.1%}")
    with col2:
        st.metric("Chủ đề", topic_label, f"{topic['confidence']:.1%}")

    if sentiment["confidence"] < 0.5 or sentiment["margin"] < 0.1:
        st.warning("Mô hình chưa chắc chắn về cảm xúc. Hãy xem xác suất chi tiết.")
    if topic["confidence"] < 0.5 or topic["margin"] < 0.1:
        st.warning("Mô hình chưa chắc chắn về chủ đề. Hãy xem xác suất chi tiết.")

    render_probabilities("Xác suất cảm xúc", sentiment["probabilities"], SENTIMENT_VN)
    render_probabilities("Xác suất chủ đề", topic["probabilities"], TOPIC_VN)

    with st.expander("Token sau tách từ"):
        st.code(" | ".join(result["tokens"]))

    keyword_hits = result.get("topic_keywords", {})
    detected = [
        f"{TOPIC_VN[topic_name]}: {', '.join(words)}"
        for topic_name, words in keyword_hits.items()
        if words
    ]
    if detected:
        st.caption("Từ khóa chủ đề phát hiện: " + "; ".join(detected))

    suggestion = suggestion_for(sentiment["label"], topic["label"])
    if suggestion:
        st.info(f"Gợi ý (rule-based): {suggestion}")

    st.caption(
        "Confidence là xác suất softmax, chưa được calibration. "
        "Không dùng kết quả này thay cho đánh giá của con người."
    )


st.title("💬 Phân tích phản hồi sinh viên")
st.caption("PhoBERT keyword-aware multi-task — dự đoán cảm xúc và chủ đề từ một câu tiếng Việt.")

with st.sidebar:
    st.header("Mô hình")
    st.write("**PhoBERT keyword-aware multi-task**")
    st.write("Encoder: `vinai/phobert-base`")
    st.write(f"Device: `{predictor.device}`")
    st.write(f"Max length: `{predictor.max_len}`")
    st.write("Keyword features: 12")
    st.divider()
    st.caption(
        "Demo chỉ dùng cho mục đích học tập. "
        "Không nhập dữ liệu sinh viên thật khi chưa có sự đồng ý."
    )

tab_predict, tab_compare, tab_about = st.tabs(["Dự đoán", "So sánh mô hình", "Giới thiệu"])

with tab_predict:
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    st.subheader("Nhập câu phản hồi")
    example_columns = st.columns(len(EXAMPLES))
    for column, (label, text) in zip(example_columns, EXAMPLES.items(), strict=True):
        if column.button(label, width="stretch"):
            st.session_state.input_text = text

    text = st.text_area(
        "Câu cần phân tích",
        key="input_text",
        height=110,
        placeholder="Ví dụ: Thầy dạy rất nhiệt tình nhưng bài tập hơi khó.",
        label_visibility="collapsed",
    )

    button_col, clear_col, _ = st.columns([1, 1, 4])
    predict_clicked = button_col.button("Dự đoán", type="primary", width="stretch")
    if clear_col.button("Xóa", width="stretch"):
        st.session_state.input_text = ""
        st.rerun()

    if predict_clicked:
        if not text.strip():
            st.warning("Vui lòng nhập một câu.")
        else:
            with st.spinner("Đang phân tích..."):
                try:
                    result = predictor.predict(text)
                except Exception as exc:  # pragma: no cover - UI guard
                    st.error(f"Không dự đoán được: {exc}")
                else:
                    render_prediction(result)

with tab_compare:
    st.subheader("Kết quả test của các mô hình")
    results_path = PROJECT_ROOT / "reports" / "results.csv"
    if not results_path.exists():
        st.info("Chưa có `reports/results.csv`.")
    else:
        results = pd.read_csv(results_path)
        test_results = results[
            (results["split"] == "test")
            & (results["task"].isin(["sentiment", "topic"]))
        ]
        if test_results.empty:
            st.info("Chưa có kết quả test.")
        else:
            table = test_results.pivot_table(
                index="model",
                columns="task",
                values="macro_f1",
                aggfunc="first",
            )
            table = table.rename(
                columns={
                    "sentiment": "Sentiment macro-F1",
                    "topic": "Topic macro-F1",
                }
            )
            st.dataframe(table.style.format("{:.4f}"), width="stretch")
            st.bar_chart(table)

        joint = results[
            (results["split"] == "test") & (results["task"] == "joint")
        ][["model", "joint_accuracy"]]
        if not joint.empty:
            st.markdown("**Joint accuracy**")
            st.dataframe(
                joint.set_index("model").style.format("{:.4f}"),
                width="stretch",
            )

with tab_about:
    st.subheader("Bài toán")
    st.write(
        "Mỗi câu phản hồi sinh viên được dự đoán đồng thời:"
        " cảm xúc (tiêu cực / trung tính / tích cực) và"
        " chủ đề (giảng viên / chương trình đào tạo / cơ sở vật chất / khác)."
    )
    st.subheader("Dữ liệu")
    st.write("UIT-VSFC — khoảng 16.000 câu phản hồi sinh viên tiếng Việt.")
    st.subheader("Mô hình")
    st.write(
        "PhoBERT được đóng băng; embedding `[CLS]` được trích một lần."
        " Topic head dùng thêm 12 keyword features và một keyword auxiliary loss"
        " để giảm lỗi shortcut theo cụm từ."
    )
    st.subheader("Hạn chế")
    st.write(
        "- Confidence chưa calibration.\n"
        "- Có thể sai với mỉa mai, teencode, câu quá dài.\n"
        "- Chủ đề là nhãn mức câu, không phải cảm xúc theo từng khía cạnh.\n"
        "- Không dùng để đánh giá cá nhân giảng viên."
    )
