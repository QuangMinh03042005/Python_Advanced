# Kế hoạch nâng cấp độ ấn tượng

## 1. Điểm xuất phát

Hiện tại đề tài đã có:

- UIT-VSFC loader, label mapping, EDA.
- Baseline TF-IDF.
- TextCNN single-task và multi-task.
- PhoBERT frozen multi-task.
- PhoBERT keyword-aware multi-task.
- 3 seed, mean ± std.
- McNemar + bootstrap.
- Demo Streamlit.
- Báo cáo thực nghiệm và Git history.

Kết quả chính (test, 3 seed):

| Model | Sentiment F1 | Topic F1 | Joint |
|---|---:|---:|---:|
| TextCNN multi | 0.7559 ± 0.0024 | 0.7508 ± 0.0103 | 0.7654 ± 0.0090 |
| PhoBERT frozen | 0.7679 ± 0.0101 | 0.7388 ± 0.0082 | 0.7612 ± 0.0291 |
| PhoBERT keyword-aware | **0.7760 ± 0.0060** | **0.7597 ± 0.0067** | **0.7989 ± 0.0059** |

Điểm mạnh: quy trình đầy đủ, có kiểm định thống kê, có demo.

Điểm còn mỏng nếu muốn “ấn tượng”:

- PhoBERT vẫn **frozen**, chưa fine-tune.
- Chưa có robustness test.
- Chưa có calibration.
- Chưa có cross-domain.
- Demo chưa giải thích dự đoán theo token.
- Chưa có so sánh chi phí/latency.

## 2. Mục tiêu nâng cấp

Sau khi nâng cấp, đề tài nên trả lời được 3 câu:

1. **Học sâu thật sự nằm ở đâu?**
2. **Cải thiện có bền vững không?**
3. **Demo có giúp người xem tin mô hình không?**

## 3. Ba hướng nâng cấp

### Hướng A — Fine-tune nhẹ PhoBERT

**Ấn tượng nhất, nhưng cần GPU/Colab.**

Làm:

- Freeze embeddings + 8 layer đầu.
- Unfreeze 1–2 layer cuối + head.
- Max length 64–128.
- Batch 8–16.
- Learning rate `1e-5` đến `2e-5`.
- 2–3 epoch, warmup 10%, early stopping.
- 3 seed: 42, 123, 2024.

So sánh:

| Cấu hình | Mục đích |
|---|---|
| Frozen | Baseline hiện tại |
| Fine-tune last 1 layer | Kiểm tra lợi ích tối thiểu |
| Fine-tune last 2 layers | Cấu hình chính |
| Fine-tune + keyword features | Xem có cộng hưởng không |

Đo:

- Macro-F1 sentiment/topic.
- Joint accuracy.
- Train time.
- Inference latency.
- Model size.

Rủi ro:

- CPU rất chậm.
- Dễ overfit vì dataset nhỏ.
- Kết quả có thể không hơn frozen.

**Kết luận:** chỉ chọn nếu có Colab/Kaggle GPU và còn ít nhất 2 tuần.

---

### Hướng B — Knowledge distillation PhoBERT → TextCNN

**Ấn tượng, chạy được trên CPU, phù hợp bài toán deployment.**

Ý tưởng:

- Teacher: PhoBERT keyword-aware (đã có).
- Student: TextCNN multi-task hoặc TextCNN nhỏ hơn.
- Train student trên:
  - hard label,
  - soft label của teacher.

Loss:

```text
L = alpha * CE(student, hard_label)
  + (1 - alpha) * T^2 * KL(student/T || teacher/T)
```

Gợi ý:

- `T = 2.0`
- `alpha = 0.5`
- 3 seed.

Đo:

- Student vs teacher: F1, joint.
- Model size.
- CPU latency.
- Tốc độ inference.

Điểm mạnh:

- Có câu chuyện “model nhỏ, nhanh, gần bằng teacher”.
- Không cần GPU.
- Rất phù hợp môn Học sâu.
- Có paper Hinton 2015 để trích dẫn.

Rủi ro:

- Student có thể kém teacher khá xa.
- Cần thêm thời gian implement.

**Kết luận:** đây là hướng “ấn tượng” nhất nếu không có GPU.

---

### Hướng C — Robustness + calibration + explainability

**An toàn nhất, tăng chiều sâu đánh giá rõ rệt.**

Làm:

1. **Robustness test**
   - Bỏ dấu.
   - Teencode.
   - Typo.
   - Emoji.
   - Đảo trật tự từ.
   - Đo macro-F1 drop.

2. **Calibration**
   - Temperature scaling trên dev.
   - ECE.
   - Reliability diagram.
   - Hiển thị confidence đã calibration trong demo.

3. **Explainability**
   - Occlusion: bỏ từng token, đo Δp.
   - Highlight token ủng hộ/chống lại dự đoán.
   - Hiển thị keyword hits.

4. **Error taxonomy**
   - Phủ định.
   - Mỉa mai.
   - Câu nhiều chủ đề.
   - Nhãn mơ hồ.
   - Teencode/không dấu.

Điểm mạnh:

- Không cần train lại nhiều.
- Báo cáo rất chuyên nghiệp.
- Demo thuyết phục hơn.
- Trả lời tốt câu hỏi “mô hình yếu ở đâu?”.

Rủi ro:

- Không làm model mạnh hơn.
- Cần công sức tạo perturbed test set.

**Kết luận:** nên làm bất kể chọn hướng nào.

## 4. Khuyến nghị theo thời gian

### Nếu còn 1 tuần

Chọn **Hướng C**:

- Calibration.
- Robustness test.
- Occlusion explainability.
- Cập nhật demo + báo cáo.

Không fine-tune, không distillation.

### Nếu còn 2 tuần

Chọn **B + C**:

- Distillation PhoBERT → TextCNN.
- Calibration + robustness.
- 3 seed cho student.
- Demo model selector.

### Nếu còn 3–4 tuần và có GPU

Chọn **A + C**:

- Fine-tune last 1–2 layer.
- 3 seed.
- Ablation frozen vs fine-tuned.
- Calibration + robustness.
- Demo model selector.

## 5. Kế hoạch 3 tuần mẫu

### Tuần 1 — Model mới

- Ngày 1–2: đọc paper, chuẩn bị code fine-tune/distill.
- Ngày 3–4: train thử 1 seed.
- Ngày 5–7: chạy 3 seed, lưu checkpoint, metric.

Sản phẩm:

```text
models/phobert_finetuned/...
hoặc
models/textcnn_distilled/...
```

### Tuần 2 — Đánh giá

- Calibration.
- Robustness test.
- Ablation.
- Error taxonomy.
- Statistical test cho model mới.

Sản phẩm:

```text
reports/phobert/finetuned/report.md
reports/evaluation/robustness_report.md
reports/evaluation/calibration_report.md
```

### Tuần 3 — Demo + báo cáo

- Model selector trong Streamlit.
- Tab explainability.
- Tab robustness.
- Cập nhật báo cáo, slide, video.

## 6. Phân công 3 người

| Thành viên | Việc |
|---|---|
| A | Fine-tune/distillation, checkpoint, 3 seed |
| B | Calibration, robustness, error taxonomy |
| C | Demo, biểu đồ, báo cáo, kiểm tra tái lập |
| Cả nhóm | Đọc paper, review kết quả, viết báo cáo |

## 7. Tiêu chí “đủ ấn tượng”

- [ ] Có ít nhất một model mới, không chỉ frozen.
- [ ] Có 3 seed và mean ± std.
- [ ] Có ít nhất một kiểm định thống kê.
- [ ] Có calibration hoặc robustness.
- [ ] Demo giải thích được dự đoán.
- [ ] Báo cáo có ablation và limitations.
- [ ] Có so sánh chi phí/latency.

## 8. Không nên làm

- Không train PhoBERT from scratch.
- Không chạy theo SOTA.
- Không thêm 5–6 model mới mà không ablation.
- Không fine-tune toàn bộ trên CPU.
- Không dùng test để chọn hyperparameter.
- Không làm UI quá phức tạp trước khi có kết quả.

## 9. Kết luận

Nếu chỉ chọn một việc để làm đề tài ấn tượng hơn:

> **Distillation PhoBERT → TextCNN** nếu không có GPU.  
> **Fine-tune 1–2 layer cuối PhoBERT** nếu có GPU.

Sau đó bắt buộc thêm **calibration + robustness + demo explainability** để câu chuyện thuyết phục.
