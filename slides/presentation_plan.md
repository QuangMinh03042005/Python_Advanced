# Kế hoạch thuyết trình

## 1. Phân công liền mạch

| Thành viên | Phần | Slide | Nội dung | Thời lượng |
|---|---|---|---|---:|
| Đinh Quang Minh | Phần 1 | 1–6 | Mở đầu, bài toán, mục tiêu, pipeline, demo | ~6 phút |
| Dương Tấn Cường | Phần 2 | 7–14 | UIT-VSFC, EDA, mô hình, calibration, robustness | ~7 phút |
| Lê Thành Duy | Phần 3 | 15–23 | Thực nghiệm, kết quả, kiểm định, distillation, hạn chế, kết luận, Q&A | ~7 phút |

Tổng: khoảng **20 phút** + Q&A 3–5 phút.

Mỗi người trình bày **một mạch liên tục**, không quay lại giữa chừng.

## 2. Cấu trúc 3 phần

### Phần 1 — Bài toán & sản phẩm (Minh)

1. Bìa
2. **Slide phân đoạn: Phần 1**
3. Bài toán và động lực
4. Mục tiêu và đóng góp
5. Pipeline tổng quan
6. Demo và ý nghĩa thực tế

**Thông điệp:** “Hệ thống làm gì, vì sao cần, và đầu ra trông như thế nào?”

Chuyển: “Sang phần dữ liệu và mô hình, mời Cường.”

### Phần 2 — Dữ liệu, mô hình & độ tin cậy (Cường)

7. **Slide phân đoạn: Phần 2**
8. UIT-VSFC
9. EDA
10. Multi-task learning
11. TextCNN
12. PhoBERT + keyword-aware
13. Calibration
14. Robustness

**Thông điệp:** “Dữ liệu có gì khó, mô hình được xây thế nào, và độ tin cậy ra sao?”

Chuyển: “Kết quả thực nghiệm, mời Duy.”

### Phần 3 — Thực nghiệm, kết quả & kết luận (Duy)

15. **Slide phân đoạn: Phần 3**
16. Thiết lập thực nghiệm
17. Kết quả seed 42
18. Kết quả 3 seed
19. Kiểm định thống kê
20. Distillation
21. Hạn chế
22. Kết luận và hướng phát triển
23. Q&A

**Thông điệp:** “Kết quả có tốt không, có bền vững không, và giới hạn ở đâu?”

## 3. Kịch bản chi tiết

### Minh — 0:00–6:00

**Slide 1 — Bìa**
- Chào thầy và các bạn.
- Giới thiệu đề tài, nhóm, GVHD.
- Câu dẫn: “Phản hồi sinh viên rất nhiều, nhưng làm sao biến nó thành tín hiệu có thể hành động?”

**Slide 2 — Phần 1**
- Chuyển ý: “Phần 1 trả lời hệ thống làm gì và vì sao cần.”

**Slide 3 — Bài toán**
- Một câu có thể vừa tích cực về giảng viên, vừa tiêu cực về cơ sở vật chất.
- Hai output: sentiment + topic.
- Single-label topic là giới hạn.

**Slide 4 — Mục tiêu và đóng góp**
- Nêu 4 việc: so sánh TextCNN/PhoBERT, keyword-aware, 3 seed + kiểm định, distillation.
- Không đọc hết bullet.

**Slide 5 — Pipeline**
- Dữ liệu → tiền xử lý → encoder → hai head.
- Keyword đi vào topic head.
- Demo là sản phẩm của pipeline này.

**Slide 6 — Demo và ý nghĩa**
- Model selector teacher/student.
- Confidence đã calibration, top-2, keyword, latency, cảnh báo multi-topic.
- Bảng triage: negative + lecturer/facility/training.
- Thông điệp: hỗ trợ con người, không thay thế.

Chuyển: “Sang phần dữ liệu và mô hình, mời Cường.”

### Cường — 6:00–13:00

**Slide 7 — Phần 2**
- Chuyển ý: “Phần 2 đi vào dữ liệu, mô hình và độ tin cậy.”

**Slide 8 — UIT-VSFC**
- 16.175 câu; train/dev/test.
- 3 nhãn sentiment, 4 nhãn topic.
- Vì sao dùng macro-F1: neutral 4%, lecturer 71.5%.

**Slide 9 — EDA**
- p95 = 33 token, neutral 4%, lecturer 71.5%, facility 474/497 negative.
- Hệ quả: class weight, per-class, cẩn thận shortcut.

**Slide 10 — Multi-task**
- Công thức `L = L_s + λL_t`.
- Lợi ích và rủi ro negative transfer.
- TextCNN multi chưa chắc tốt hơn single.

**Slide 11 — TextCNN**
- Embedding 100, conv 3/4/5, max-pool, shared 256, hai head.
- 0.41M tham số; nhẹ, CPU-friendly.

**Slide 12 — PhoBERT + keyword-aware**
- PhoBERT frozen `[CLS]` 768 → shared 256.
- 12 keyword features; auxiliary loss.
- Kết quả then chốt: topic +0.0205, joint +0.0199 so với frozen.

**Slide 13 — Calibration**
- Student overconfident trước calibration.
- Temperature scaling giảm ECE mạnh.
- Accuracy không đổi.

**Slide 14 — Robustness**
- Mất dấu là điểm yếu lớn nhất.
- Teacher nhạy với emoji/shuffle; student bền vững hơn.
- Hàm ý: cần chuẩn hóa input/augmentation.

Chuyển: “Kết quả thực nghiệm, mời Duy.”

### Duy — 13:00–20:00

**Slide 15 — Phần 3**
- Chuyển ý: “Phần 3 trả lời kết quả có tốt và bền vững không.”

**Slide 16 — Setup**
- CPU, 3 seed, chọn model bằng dev, test một lần.

**Slide 17 — Kết quả seed 42**
- Đọc 3 dòng quan trọng: TF-IDF, PhoBERT frozen, keyword-aware.
- Keyword-aware: 0.7788 / 0.7633 / 0.8054.

**Slide 18 — 3 seed**
- Mean ± std.
- Keyword-aware tốt nhất cả 3 metric; phương sai joint thấp hơn frozen.

**Slide 19 — Kiểm định**
- McNemar + bootstrap.
- Topic/joint so với frozen nhất quán; joint so với TextCNN nhất quán; sentiment chưa đủ bằng chứng.

**Slide 20 — Distillation**
- Teacher vs hard-label vs distilled.
- 407K tham số, 1.55MB, 2.30ms/câu, nhanh hơn 27.6×.
- Student gần ngang teacher ở topic.

**Slide 21 — Hạn chế**
- Chưa fine-tune PhoBERT.
- Dataset nhỏ, một miền, chưa cross-domain.
- Single-label topic.
- Mất dấu/teencode còn yếu.
- Không dùng đánh giá cá nhân giảng viên.

**Slide 22 — Kết luận**
- 3 kết quả chính.
- 4 hướng phát triển.
- Câu chốt: “Từ bài toán phân loại, nhóm đã xây pipeline có thể đo, kiểm định, nén và triển khai trên CPU.”

**Slide 23 — Q&A**
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
- Slide 4: chỉ nói 2 đóng góp.
- Slide 11: chỉ nói TextCNN nhẹ, 0.41M.
- Slide 13: chỉ nói “ECE giảm mạnh”.
- Slide 22: chỉ nói 3 kết quả chính.

Giữ nguyên:
- Slide 3, 5, 6, 12, 17, 18, 20.
