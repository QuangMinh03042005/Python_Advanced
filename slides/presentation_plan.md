# Kế hoạch thuyết trình

## 1. Phân công

| Thành viên | Slide | Nội dung | Thời lượng |
|---|---|---|---:|
| Đinh Quang Minh | 1–5, 18, 20–21 | Mở đầu, bài toán, mục tiêu, pipeline; demo; kết luận; Q&A | ~7 phút |
| Dương Tấn Cường | 6–10, 19 | UIT-VSFC, EDA, TextCNN, PhoBERT, keyword-aware; hạn chế | ~6 phút |
| Lê Thành Duy | 11–17 | Thực nghiệm, kết quả, kiểm định, distillation, calibration, robustness | ~5 phút |
| Cả nhóm | 21 | Q&A | 3–5 phút |

Tổng: khoảng **18–20 phút** + Q&A.

## 2. Kịch bản theo thời gian

### Minh — 0:00–5:00

**Slide 1 — Bìa**
- Chào thầy và các bạn.
- Giới thiệu tên đề tài, nhóm, GVHD.
- Câu dẫn: “Phản hồi sinh viên rất nhiều, nhưng làm sao biến nó thành tín hiệu có thể hành động?”

**Slide 2 — Phân công**
- Nói ngắn gọn 3 phần: mở đầu, dữ liệu/mô hình, kết quả.
- Không đọc chi tiết bảng; chỉ nêu thứ tự.

**Slide 3 — Bài toán**
- Một câu phản hồi có thể vừa tích cực về giảng viên, vừa tiêu cực về cơ sở vật chất.
- Hai output: sentiment + topic.
- Nhấn mạnh: single-label topic nên câu nhiều chủ đề là giới hạn.

**Slide 4 — Mục tiêu và đóng góp**
- Nêu 4 việc: so sánh TextCNN/PhoBERT, keyword-aware, 3 seed + kiểm định, distillation.
- Không đọc hết bullet; chọn ý chính.

**Slide 5 — Pipeline**
- Chỉ vào sơ đồ: dữ liệu → tiền xử lý → encoder → hai head.
- Nói keyword đi vào topic head.
- Chuyển: “Để hiểu dữ liệu, mời Cường.”

### Cường — 5:00–11:00

**Slide 6 — UIT-VSFC**
- 16.175 câu; train/dev/test.
- 3 nhãn sentiment, 4 nhãn topic.
- Vì sao dùng macro-F1: neutral 4%, lecturer 71.5%.

**Slide 7 — EDA**
- 4 con số: p95 = 33 token, neutral 4%, lecturer 71.5%, facility 474/497 negative.
- Hệ quả: class weight, per-class, cẩn thận shortcut.

**Slide 8 — Multi-task**
- Công thức `L = L_s + λL_t`.
- Lợi ích và rủi ro negative transfer.
- Nói rõ: TextCNN multi chưa chắc tốt hơn single.

**Slide 9 — TextCNN**
- Kiến trúc: embedding 100, conv 3/4/5, max-pool, shared 256, hai head.
- 0.41M tham số; nhẹ, CPU-friendly.

**Slide 10 — PhoBERT + keyword-aware**
- PhoBERT frozen `[CLS]` 768 → shared 256.
- 12 keyword features; auxiliary loss.
- Kết quả then chốt: topic +0.0205, joint +0.0199 so với frozen.

Chuyển: “Kết quả thực nghiệm, mời Duy.”

### Duy — 11:00–16:00

**Slide 11 — Setup**
- CPU, 3 seed, chọn model bằng dev, test một lần.

**Slide 12 — Kết quả seed 42**
- Đọc 3 dòng quan trọng: TF-IDF, PhoBERT frozen, keyword-aware.
- Keyword-aware: 0.7788 / 0.7633 / 0.8054.

**Slide 13 — 3 seed**
- Mean ± std.
- Nhấn mạnh: keyword-aware tốt nhất cả 3 metric; phương sai joint thấp hơn frozen.

**Slide 14 — Kiểm định**
- McNemar + bootstrap.
- Kết luận: topic/joint so với frozen nhất quán; joint so với TextCNN nhất quán; sentiment chưa đủ bằng chứng.

**Slide 15 — Distillation**
- Teacher vs hard-label vs distilled.
- 407K tham số, 1.55MB, 2.30ms/câu, nhanh hơn 27.6×.
- Student gần ngang teacher ở topic.

**Slide 16 — Calibration**
- Student overconfident trước calibration.
- Temperature scaling giảm ECE mạnh.
- Accuracy không đổi.

**Slide 17 — Robustness**
- Mất dấu là điểm yếu lớn nhất.
- Teacher nhạy với emoji/shuffle; student bền vững hơn.
- Hàm ý: cần chuẩn hóa input/augmentation.

Chuyển: “Minh sẽ demo và nói ý nghĩa thực tế.”

### Minh — 16:00–18:00

**Slide 18 — Demo và ý nghĩa**
- Model selector teacher/student.
- Confidence đã calibration, top-2, keyword, latency, cảnh báo multi-topic.
- Bảng triage: negative + lecturer/facility/training.
- Thông điệp: hỗ trợ con người, không thay thế.

Chuyển: “Cường bổ sung hạn chế.”

### Cường — 18:00–19:00

**Slide 19 — Hạn chế**
- Chưa fine-tune PhoBERT.
- Dataset nhỏ, một miền, chưa cross-domain.
- Single-label topic.
- Mất dấu/teencode còn yếu.
- Không dùng đánh giá cá nhân giảng viên.

Chuyển: “Minh chốt lại.”

### Minh — 19:00–20:00

**Slide 20 — Kết luận**
- 3 kết quả chính.
- 4 hướng phát triển.
- Câu chốt: “Từ bài toán phân loại, nhóm đã xây pipeline có thể đo, kiểm định, nén và triển khai trên CPU.”

**Slide 21 — Q&A**
- Cảm ơn.
- Mời câu hỏi.

## 3. Câu hỏi dự phòng

**Vì sao không fine-tune PhoBERT?**
> CPU-only, fine-tune toàn bộ rất chậm; nhóm chọn frozen + head để đảm bảo tái lập. Fine-tune 1–2 layer là hướng phát triển.

**Vì sao TextCNN multi-task không tốt hơn single-task?**
> Hai task có tương quan nhưng không đồng nhất; hidden 256 nhỏ, có thể negative transfer. Kết quả này được báo cáo trung thực, không tô hồng multi-task.

**Vì sao câu hai chủ đề không ra 50/50?**
> Bài toán là single-label, softmax buộc chọn một. 50/50 cần multi-label annotation và sigmoid head. Demo đã cảnh báo multi-topic + top-2.

**Vì sao test chỉ đánh giá một lần?**
> Để tránh tune trên test. Model chọn bằng dev; test chỉ dùng cho cấu hình cuối.

**Vì sao distillation?**
> Teacher chính xác nhưng nặng; student nhỏ hơn 332×, nhanh hơn 27.6×, phù hợp CPU. Mục tiêu là trade-off, không phải vượt teacher.

**Điểm mạnh nhất của đề tài?**
> Không chỉ train model: có baseline, 3 seed, kiểm định, calibration, robustness, distillation và demo. Biết rõ giới hạn của mình.

## 4. Nếu chỉ có 15 phút

Cắt hoặc rút gọn:
- Slide 4: chỉ nói 2 đóng góp.
- Slide 9: chỉ nói TextCNN nhẹ, 0.41M.
- Slide 16: chỉ nói “ECE giảm mạnh”.
- Slide 20: chỉ nói 3 kết quả chính.

Giữ nguyên:
- Slide 3, 5, 10, 12, 13, 15, 18.
