# Kế hoạch thuyết trình

## 1. Phân công

| Thành viên | Slide | Nội dung | Thời lượng |
|---|---|---|---:|
| Đinh Quang Minh | 1–6, 19, 21–22 | Mở đầu, bài toán, mục tiêu, pipeline; demo; kết luận; Q&A | ~8 phút |
| Dương Tấn Cường | 7–11, 20 | UIT-VSFC, EDA, multi-task, TextCNN, PhoBERT, keyword-aware; hạn chế | ~6 phút |
| Lê Thành Duy | 12–18 | Thực nghiệm, kết quả, kiểm định, distillation, calibration, robustness | ~5–6 phút |
| Cả nhóm | 22 | Q&A | 3–5 phút |

Tổng: khoảng **19–20 phút** + Q&A.

## 2. Luồng bàn giao

| Từ | Slide | Nội dung | Bàn giao cho |
|---|---|---|---|
| Minh | 1–6 | Mở đầu, bài toán, mục tiêu, pipeline | Cường |
| Cường | 7–11 | Dữ liệu, EDA, mô hình | Duy |
| Duy | 12–18 | Kết quả, kiểm định, distillation, calibration, robustness | Minh |
| Minh | 19 | Demo và ý nghĩa thực tế | Cường |
| Cường | 20 | Hạn chế | Minh |
| Minh | 21–22 | Kết luận và Q&A | Kết thúc |

## 3. Kịch bản theo thời gian

### Minh — 0:00–6:00

**Slide 1 — Bìa**
- Chào thầy và các bạn.
- Giới thiệu đề tài, nhóm, GVHD.
- Câu dẫn: “Phản hồi sinh viên rất nhiều, nhưng làm sao biến nó thành tín hiệu có thể hành động?”

**Slide 2 — Phân công**
- Nói ngắn gọn 3 phần: mở đầu, dữ liệu/mô hình, kết quả.
- Không đọc chi tiết bảng.

**Slide 3 — Luồng bàn giao**
- Nói rõ ai nói phần nào, chuyển cho ai.
- Câu chuyển: “Minh mở đầu, Cường dữ liệu/mô hình, Duy kết quả, Minh demo, Cường hạn chế, Minh kết luận.”

**Slide 4 — Bài toán**
- Một câu có thể vừa tích cực về giảng viên, vừa tiêu cực về cơ sở vật chất.
- Hai output: sentiment + topic.
- Nhấn mạnh: single-label topic nên câu nhiều chủ đề là giới hạn.

**Slide 5 — Mục tiêu và đóng góp**
- Nêu 4 việc: so sánh TextCNN/PhoBERT, keyword-aware, 3 seed + kiểm định, distillation.
- Không đọc hết bullet.

**Slide 6 — Pipeline**
- Dữ liệu → tiền xử lý → encoder → hai head.
- Keyword đi vào topic head.
- Chuyển: “Để hiểu dữ liệu, mời Cường.”

### Cường — 6:00–12:00

**Slide 7 — UIT-VSFC**
- 16.175 câu; train/dev/test.
- 3 nhãn sentiment, 4 nhãn topic.
- Vì sao dùng macro-F1: neutral 4%, lecturer 71.5%.

**Slide 8 — EDA**
- 4 con số: p95 = 33 token, neutral 4%, lecturer 71.5%, facility 474/497 negative.
- Hệ quả: class weight, per-class, cẩn thận shortcut.

**Slide 9 — Multi-task**
- Công thức `L = L_s + λL_t`.
- Lợi ích và rủi ro negative transfer.
- TextCNN multi chưa chắc tốt hơn single.

**Slide 10 — TextCNN**
- Embedding 100, conv 3/4/5, max-pool, shared 256, hai head.
- 0.41M tham số; nhẹ, CPU-friendly.

**Slide 11 — PhoBERT + keyword-aware**
- PhoBERT frozen `[CLS]` 768 → shared 256.
- 12 keyword features; auxiliary loss.
- Kết quả then chốt: topic +0.0205, joint +0.0199 so với frozen.

Chuyển: “Kết quả thực nghiệm, mời Duy.”

### Duy — 12:00–17:30

**Slide 12 — Setup**
- CPU, 3 seed, chọn model bằng dev, test một lần.

**Slide 13 — Kết quả seed 42**
- Đọc 3 dòng quan trọng: TF-IDF, PhoBERT frozen, keyword-aware.
- Keyword-aware: 0.7788 / 0.7633 / 0.8054.

**Slide 14 — 3 seed**
- Mean ± std.
- Keyword-aware tốt nhất cả 3 metric; phương sai joint thấp hơn frozen.

**Slide 15 — Kiểm định**
- McNemar + bootstrap.
- Topic/joint so với frozen nhất quán; joint so với TextCNN nhất quán; sentiment chưa đủ bằng chứng.

**Slide 16 — Distillation**
- Teacher vs hard-label vs distilled.
- 407K tham số, 1.55MB, 2.30ms/câu, nhanh hơn 27.6×.
- Student gần ngang teacher ở topic.

**Slide 17 — Calibration**
- Student overconfident trước calibration.
- Temperature scaling giảm ECE mạnh.
- Accuracy không đổi.

**Slide 18 — Robustness**
- Mất dấu là điểm yếu lớn nhất.
- Teacher nhạy với emoji/shuffle; student bền vững hơn.
- Hàm ý: cần chuẩn hóa input/augmentation.

Chuyển: “Minh sẽ demo và nói ý nghĩa thực tế.”

### Minh — 17:30–18:30

**Slide 19 — Demo và ý nghĩa**
- Model selector teacher/student.
- Confidence đã calibration, top-2, keyword, latency, cảnh báo multi-topic.
- Bảng triage: negative + lecturer/facility/training.
- Thông điệp: hỗ trợ con người, không thay thế.

Chuyển: “Cường bổ sung hạn chế.”

### Cường — 18:30–19:30

**Slide 20 — Hạn chế**
- Chưa fine-tune PhoBERT.
- Dataset nhỏ, một miền, chưa cross-domain.
- Single-label topic.
- Mất dấu/teencode còn yếu.
- Không dùng đánh giá cá nhân giảng viên.

Chuyển: “Minh chốt lại.”

### Minh — 19:30–20:30

**Slide 21 — Kết luận**
- 3 kết quả chính.
- 4 hướng phát triển.
- Câu chốt: “Từ bài toán phân loại, nhóm đã xây pipeline có thể đo, kiểm định, nén và triển khai trên CPU.”

**Slide 22 — Q&A**
- Cảm ơn.
- Mời câu hỏi.

## 4. Câu hỏi dự phòng

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

## 5. Nếu chỉ có 15 phút

Cắt hoặc rút gọn:
- Slide 5: chỉ nói 2 đóng góp.
- Slide 10: chỉ nói TextCNN nhẹ, 0.41M.
- Slide 17: chỉ nói “ECE giảm mạnh”.
- Slide 21: chỉ nói 3 kết quả chính.

Giữ nguyên:
- Slide 4, 6, 11, 13, 14, 16, 19.
