from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import predict as teacher_module  # noqa: E402
from src import predict_textcnn as student_module  # noqa: E402

TEACHER_CHECKPOINT_PATH = teacher_module.DEFAULT_CHECKPOINT_PATH
STUDENT_CHECKPOINT_PATH = student_module.DEFAULT_CHECKPOINT_PATH
TEACHER_LABEL = "PhoBERT teacher"
STUDENT_LABEL = "TextCNN student"
CALIBRATION_PATH = PROJECT_ROOT / "reports" / "evaluation" / "calibration_temperatures.json"
BENCHMARK_PATH = PROJECT_ROOT / "reports" / "distillation" / "distillation_benchmark.json"

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


@st.cache_data
def load_temperatures() -> dict[str, dict[str, float]]:
    defaults = {
        "teacher": {"sentiment": 1.0, "topic": 1.0},
        "student": {"sentiment": 1.0, "topic": 1.0},
    }
    if not CALIBRATION_PATH.exists():
        return defaults
    payload = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    temperatures = payload.get("temperatures", {})
    return {
        model: {
            "sentiment": float(temperatures.get(model, {}).get("sentiment", 1.0)),
            "topic": float(temperatures.get(model, {}).get("topic", 1.0)),
        }
        for model in ("teacher", "student")
    }


@st.cache_resource(show_spinner="Đang tải PhoBERT teacher...")
def load_teacher(
    sentiment_temperature: float,
    topic_temperature: float,
) -> teacher_module.PhoBERTMultiTaskPredictor:
    predictor = teacher_module.PhoBERTMultiTaskPredictor(
        TEACHER_CHECKPOINT_PATH,
        sentiment_temperature=sentiment_temperature,
        topic_temperature=topic_temperature,
    )
    predictor.predict("câu khởi động")
    return predictor


@st.cache_resource(show_spinner="Đang tải TextCNN student...")
def load_student(
    sentiment_temperature: float,
    topic_temperature: float,
) -> student_module.TextCNNMultiTaskPredictor:
    predictor = student_module.TextCNNMultiTaskPredictor(
        STUDENT_CHECKPOINT_PATH,
        sentiment_temperature=sentiment_temperature,
        topic_temperature=topic_temperature,
    )
    predictor.predict("câu khởi động")
    return predictor


temperatures = load_temperatures()

with st.sidebar:
    st.header("Mô hình")
    selected_model = st.selectbox(
        "Chọn mô hình",
        [TEACHER_LABEL, STUDENT_LABEL],
        index=0,
    )

if selected_model == TEACHER_LABEL:
    model_key = "teacher"
    load_selected = load_teacher
elif selected_model == STUDENT_LABEL:
    model_key = "student"
    load_selected = load_student
else:
    st.error("Mô hình không hợp lệ.")
    st.stop()

try:
    predictor = load_selected(
        temperatures[model_key]["sentiment"],
        temperatures[model_key]["topic"],
    )
except Exception as exc:  # pragma: no cover - UI guard
    st.error(f"Không tải được mô hình: {exc}")
    if model_key == "student":
        st.info("Hãy chạy `python -m src.train_textcnn_distill` để tạo checkpoint student.")
    else:
        st.info("Hãy chạy `python -m src.train_phobert_keyword_multitask` trước.")
    st.stop()


def render_probabilities(title: str, probabilities: dict[str, float], labels_vn: dict[str, str]) -> None:
    st.markdown(f"**{title}**")
    for label, probability in sorted(probabilities.items(), key=lambda item: item[1], reverse=True):
        value = min(max(float(probability), 0.0), 1.0)
        st.progress(value, text=f"{labels_vn[label]}: {value:.1%}")


def top_topics(probabilities: dict[str, float], limit: int = 2) -> list[tuple[str, float]]:
    return sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:limit]


def mentioned_topic_labels(topic_keywords: dict[str, list[str]]) -> list[str]:
    return [
        TOPIC_VN[topic]
        for topic, words in topic_keywords.items()
        if words
    ]


def render_top_topics(probabilities: dict[str, float]) -> None:
    top = top_topics(probabilities)
    summary = " · ".join(
        f"{TOPIC_VN[label]} {probability:.1%}"
        for label, probability in top
    )
    st.caption(f"Top-2 chủ đề: {summary}")


def render_multi_topic_warning(result: dict[str, Any]) -> None:
    mentioned = mentioned_topic_labels(result.get("topic_keywords", {}))
    if len(mentioned) < 2:
        return
    st.warning(
        "Câu này nhắc đến nhiều chủ đề: "
        + ", ".join(mentioned)
        + ". Mô hình hiện là single-label nên chỉ chọn chủ đề trội; "
        "các xác suất bên dưới không phải xác suất độc lập của từng chủ đề."
    )
    top = top_topics(result["topic"]["probabilities"])
    columns = st.columns(len(top))
    for column, (label, probability) in zip(columns, top, strict=True):
        column.metric(f"Top {TOPIC_VN[label]}", f"{probability:.1%}")


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

    render_multi_topic_warning(result)

    render_probabilities("Xác suất cảm xúc", sentiment["probabilities"], SENTIMENT_VN)
    render_probabilities("Xác suất chủ đề", topic["probabilities"], TOPIC_VN)
    render_top_topics(topic["probabilities"])

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
        f"{result.get('model_name', 'Mô hình')} · "
        f"{result.get('parameter_count', 0):,} tham số · "
        f"{result.get('latency_ms', 0.0):.2f} ms/câu"
    )
    st.caption(
        "Confidence đã calibration bằng temperature scaling trên dev. "
        "Không dùng kết quả này thay cho đánh giá của con người."
    )


st.title("💬 Phân tích phản hồi sinh viên")
st.caption("So sánh PhoBERT teacher và TextCNN student — dự đoán cảm xúc và chủ đề từ một câu tiếng Việt.")

with st.sidebar:
    st.write(f"**{predictor.model_name}**")
    st.write(f"Device: `{predictor.device}`")
    st.write(f"Max length: `{predictor.max_len}`")
    st.write(f"Parameters: `{predictor.parameter_count:,}`")
    st.write(
        "Calibration: "
        f"`T_sent={predictor.sentiment_temperature:.3f}`, "
        f"`T_topic={predictor.topic_temperature:.3f}`"
    )
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

    st.subheader("Teacher vs Student trên CPU")
    if not BENCHMARK_PATH.exists():
        st.info("Chưa có `reports/distillation/distillation_benchmark.json`.")
    else:
        benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        teacher = benchmark["teacher"]
        student = benchmark["student"]
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Teacher latency",
            f"{teacher['ms_per_sentence']:.2f} ms",
            f"{teacher['parameter_count']:,} tham số",
        )
        col2.metric(
            "Student latency",
            f"{student['ms_per_sentence']:.2f} ms",
            f"{student['parameter_count']:,} tham số",
        )
        col3.metric("Speedup", f"{benchmark['speedup']:.1f}×")
        st.caption(
            f"Đo trên {benchmark['n_sentences']} câu test, device "
            f"`{benchmark['device']}`."
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
        "**Teacher:** PhoBERT keyword-aware multi-task."
        " **Student:** TextCNN multi-task được distillation từ teacher."
        " Student nhỏ hơn ~332 lần và nhanh hơn ~26 lần trên CPU."
    )
    st.write(
        "Confidence được calibration bằng temperature scaling trên dev."
        " Calibration không làm thay đổi nhãn dự đoán, chỉ sửa độ tin cậy."
    )
    st.subheader("Hạn chế")
    st.write(
        "- Có thể sai với mỉa mai, teencode, câu quá dài.\n"
        "- Mất dấu là lỗi nghiêm trọng nhất với cả teacher và student.\n"
        "- Chủ đề là nhãn mức câu, không phải cảm xúc theo từng khía cạnh.\n"
        "- Không dùng để đánh giá cá nhân giảng viên."
    )
