# ĐỀ CƯƠNG ĐỒ ÁN MÔN HỌC SÂU

## Multi-task Learning cho Phân loại Cảm xúc và Chủ đề Phản hồi Sinh viên Tiếng Việt: So sánh TextCNN và PhoBERT

**Tên tiếng Anh:** Multi-Task Learning for Vietnamese Student Feedback: Joint Sentiment and Topic Classification with TextCNN and PhoBERT

| Thông tin | Nội dung |
|---|---|
| Nhóm | 3 sinh viên |
| Thời gian đề xuất | 6 tuần (có thể nén còn 4 tuần) |
| Điều kiện tính toán | CPU; Colab/Kaggle miễn phí là tùy chọn |
| Ngôn ngữ | Python |
| Thư viện chính | PyTorch, Hugging Face Transformers, scikit-learn, pandas, Streamlit |
| Dataset | UIT-VSFC |
| Bài toán | Multi-task classification: cảm xúc 3 lớp + chủ đề 4 lớp |

---

## 1. Tóm tắt

Phản hồi của sinh viên cuối mỗi học kỳ là nguồn dữ liệu quan trọng để đánh giá chất lượng giảng dạy và cơ sở vật chất. Mỗi phản hồi thường chứa hai thông tin: **cảm xúc** (tích cực, trung tính, tiêu cực) và **chủ đề** (giảng viên, chương trình đào tạo, cơ sở vật chất, khác).

Đề tài xây dựng một mô hình **multi-task learning** dùng chung encoder để giải đồng thời hai bài toán trên, thay vì huấn luyện hai mô hình riêng. Hai kiến trúc được so sánh:

1. **TextCNN** — nhẹ, huấn luyện được trên CPU.
2. **PhoBERT** — mạnh hơn nhờ ngữ cảnh, nhưng cần chiến lược phù hợp để chạy trên CPU.

Ngoài việc báo cáo F1 của từng tác vụ, đề tài tập trung phân tích:

- Multi-task có giúp ích so với single-task không?
- Có xảy ra **negative transfer** không?
- Trọng số loss `λ` ảnh hưởng thế nào?
- Multi-task có hữu ích hơn khi dữ liệu ít không?
- Chi phí tính toán và kích thước mô hình đánh đổi thế nào?

Kết quả kỳ vọng là một pipeline tái lập được, một bộ thực nghiệm có đối chứng, và một demo Streamlit cho phép nhập một câu để nhận về cảm xúc + chủ đề.

---

## 2. Bối cảnh và động lực

### 2.1. Bài toán thực tế

Trong khảo sát cuối kỳ, sinh viên thường để lại nhận xét ngắn:

> “Cơ sở vật chất còn yếu, phòng học thiếu máy chiếu.”

Một câu như vậy vừa thể hiện **cảm xúc tiêu cực**, vừa thuộc **chủ đề cơ sở vật chất**. Nếu chỉ phân loại cảm xúc, nhà trường biết “có vấn đề” nhưng không biết vấn đề ở đâu. Nếu chỉ phân loại chủ đề, nhà trường biết “cơ sở vật chất” nhưng không biết mức độ hài lòng.

Multi-task learning giải quyết cả hai trong một lần dự đoán.

### 2.2. Động lực kỹ thuật

- Hai tác vụ có chung đầu vào và có thể chia sẻ biểu diễn ngôn ngữ.
- Nhãn chủ đề có sẵn trong UIT-VSFC, không cần gán nhãn thêm.
- Multi-task có thể hoạt động như một dạng regularization, đặc biệt hữu ích khi dữ liệu nhỏ.
- So sánh TextCNN và PhoBERT cho thấy rõ đánh đổi giữa **chi phí tính toán** và **độ chính xác**.

### 2.3. Khoảng trống trong phạm vi môn học

Multi-task learning không phải là kỹ thuật mới. Đóng góp của đề tài không nằm ở việc phát minh kiến trúc, mà ở:

- Đánh giá thực nghiệm có kiểm soát trên **UIT-VSFC**.
- Phân tích **negative transfer** và **loss weighting**.
- Phân tích **data efficiency** — multi-task có lợi khi ít dữ liệu không.
- Cung cấp pipeline **CPU-friendly** và demo tái lập được.

---

## 3. Mục tiêu

### 3.1. Mục tiêu tổng quát

Xây dựng và đánh giá mô hình multi-task cho phân loại cảm xúc và chủ đề trên phản hồi sinh viên tiếng Việt, sử dụng TextCNN và PhoBERT trong điều kiện không có GPU.

### 3.2. Mục tiêu cụ thể

1. Xây dựng pipeline tiền xử lý tiếng Việt và chuẩn hóa nhãn cho UIT-VSFC.
2. Cài đặt baseline single-task cho cảm xúc và chủ đề.
3. Cài đặt multi-task TextCNN với shared encoder và hai head.
4. Cài đặt multi-task PhoBERT ở chế độ frozen; fine-tune là tùy chọn.
5. So sánh multi-task với single-task trên cùng split.
6. Khảo sát ảnh hưởng của `λ`, class weight và lượng dữ liệu.
7. Phân tích lỗi và hiện tượng negative transfer.
8. Xây dựng demo Streamlit.
9. Viết báo cáo và tài liệu tái lập.

---

## 4. Câu hỏi nghiên cứu

**RQ1.** Multi-task có cải thiện macro-F1 của bài toán cảm xúc so với single-task không?

**RQ2.** Chủ đề là tác vụ phụ có lợi, hay gây negative transfer cho cảm xúc?

**RQ3.** TextCNN và PhoBERT, mô hình nào hưởng lợi nhiều hơn từ multi-task?

**RQ4.** Trọng số loss `λ` ảnh hưởng thế nào đến sự cân bằng giữa hai tác vụ?

**RQ5.** Khi giảm lượng dữ liệu huấn luyện, multi-task có cải thiện độ chính xác nhiều hơn single-task không?

**RQ6.** Đánh đổi giữa độ chính xác, kích thước mô hình và thời gian suy luận trên CPU là gì?

---

## 5. Phạm vi

### 5.1. Trong phạm vi

- Dataset UIT-VSFC.
- Phân loại cảm xúc mức câu: 3 lớp.
- Phân loại chủ đề mức câu: 4 lớp.
- Hai kiến trúc: TextCNN và PhoBERT.
- Multi-task hard parameter sharing.
- Huấn luyện và suy luận trên CPU.
- Demo một câu và so sánh mô hình.

### 5.2. Ngoài phạm vi

- Aspect-based sentiment (cảm xúc theo từng khía cạnh).
- Multi-label (một câu nhiều chủ đề).
- Fine-tune LLM.
- Dữ liệu đa ngôn ngữ.
- Triển khai production quy mô lớn.
- Streaming/real-time.

---

## 6. Dataset

### 6.1. UIT-VSFC

- Khoảng 16.000 câu phản hồi sinh viên.
- Có split chính thức: train / dev / test.
- Mỗi câu có:
  - 1 nhãn cảm xúc: `negative`, `neutral`, `positive`.
  - 1 nhãn chủ đề: `lecturer`, `training_program` (curriculum), `facility`, `others`.

### 6.2. Lưu ý về label mapping

Thứ tự nhãn trong file gốc có thể khác thứ tự trong tài liệu. **Không hardcode** theo cảm tính.

Cần làm:

- Đọc file nhãn gốc.
- Tạo `label_map.json`.
- Viết unit test nhỏ kiểm tra số lớp và phân bố.
- In ra 5 ví dụ mỗi lớp để kiểm tra thủ công.

### 6.3. EDA cần làm

- Phân bố cảm xúc.
- Phân bố chủ đề.
- Phân bố chéo cảm xúc × chủ đề.
- Độ dài câu: min, max, mean, percentile 95.
- Câu trùng lặp hoặc gần trùng.
- Câu rỗng, câu chỉ có emoji, câu không dấu.
- Kiểm tra rò rỉ dữ liệu giữa train/dev/test.

### 6.4. Tiền xử lý

- Chuẩn hóa Unicode NFC.
- Chuẩn hóa khoảng trắng.
- Xử lý teencode bằng từ điển nhỏ.
- Tách từ tiếng Việt bằng `underthesea` hoặc `pyvi`.
- Với PhoBERT: dùng văn bản đã tách từ theo hướng dẫn của tokenizer.
- Giữ emoji ở một thí nghiệm, loại bỏ ở thí nghiệm khác để so sánh.
- Xây dựng vocabulary **chỉ trên train**.
- Padding/truncation về `max_len`.

### 6.5. Giấy phép và đạo đức

- Dataset có giấy phép chưa rõ ràng trong một số bản phân phối.
- Chỉ dùng cho mục đích học tập/nghiên cứu.
- Không công bố lại dữ liệu thô.
- Không dùng demo với dữ liệu thật của sinh viên khi chưa có sự đồng ý.

---

## 7. Phát biểu bài toán

Cho câu đầu vào:

\[
x = (x_1, x_2, \dots, x_L)
\]

Mô hình dự đoán:

\[
\hat{y}_s = f_s(x), \quad \hat{y}_t = f_t(x)
\]

Trong đó:

- \(\hat{y}_s \in \{0,1,2\}\) là cảm xúc.
- \(\hat{y}_t \in \{0,1,2,3\}\) là chủ đề.

Encoder dùng chung:

\[
h = E_\theta(x)
\]

Hai head:

\[
z_s = W_s h + b_s
\]

\[
z_t = W_t h + b_t
\]

Loss:

\[
\mathcal{L} = \mathcal{L}_{sent} + \lambda \mathcal{L}_{topic}
\]

Với:

\[
\mathcal{L}_{sent} = \text{CrossEntropy}(z_s, y_s)
\]

\[
\mathcal{L}_{topic} = \text{CrossEntropy}(z_t, y_t)
\]

---

## 8. Kiến trúc mô hình

### 8.1. Multi-task TextCNN

```text
Câu
 ↓ tách từ
Token IDs
 ↓ Embedding
Ma trận (L × D)
 ↓ Conv1D nhiều kernel size (3, 4, 5)
Feature maps
 ↓ ReLU + max-pooling theo thời gian
Vector đặc trưng
 ↓ Concatenate + Dropout
Shared fully-connected layer
 ├── Sentiment head → 3 logits
 └── Topic head     → 4 logits
```

Thiết kế đề xuất:

| Thành phần | Giá trị |
|---|---|
| Embedding dim | 100 hoặc 128 |
| Kernel sizes | 3, 4, 5 |
| Số filter mỗi kernel | 100 |
| Dropout | 0.5 |
| Shared FC | 256 |
| Sentiment head | 256 → 3 |
| Topic head | 256 → 4 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 32 |
| Max epochs | 20 |
| Early stopping | patience 3 |

### 8.2. Multi-task PhoBERT

```text
Câu
 ↓ tách từ + tokenizer
input_ids, attention_mask
 ↓ PhoBERT
last_hidden_state
 ↓ lấy vector [CLS] hoặc mean-pooling
Vector 768 chiều
 ↓ Dropout
 ├── Sentiment head → 3 logits
 └── Topic head     → 4 logits
```

Hai chế độ:

| Chế độ | Trainable | CPU |
|---|---:|---|
| Frozen PhoBERT + 2 head | ~2.3k tham số | Rất nhẹ |
| Fine-tune toàn bộ | ~135M tham số | Nặng |

Chiến lược CPU:

1. Chạy PhoBERT một lần để trích `[CLS]` cho toàn bộ dataset.
2. Lưu thành `.npy`.
3. Train hai head bằng MLP nhỏ.
4. Nếu có Colab, fine-tune nhẹ để so sánh.

---

## 9. Baseline và thực nghiệm

### 9.1. Baseline

| Mã | Mô hình | Tác vụ |
|---|---|---|
| B0 | Majority class | Sentiment |
| B1 | TF-IDF + Logistic Regression | Sentiment |
| B2 | TF-IDF + Logistic Regression | Topic |
| B3 | TextCNN single-task | Sentiment |
| B4 | TextCNN single-task | Topic |
| B5 | PhoBERT frozen + head | Sentiment |
| B6 | PhoBERT frozen + head | Topic |

### 9.2. Mô hình đề xuất

| Mã | Mô hình | Ghi chú |
|---|---|---|
| M1 | Multi-task TextCNN | `λ = 0.5` |
| M2 | Multi-task TextCNN | Tune `λ` |
| M3 | Multi-task PhoBERT frozen | `λ = 0.5` |
| M4 | Multi-task PhoBERT frozen | Tune `λ` |
| M5 | Multi-task PhoBERT fine-tune | Tùy chọn, Colab |

### 9.3. Ablation

- `λ ∈ {0.1, 0.3, 0.5, 1.0}`.
- Shared encoder vs hai encoder riêng.
- Có shared FC vs head trực tiếp từ pooled vector.
- Kernel 3/4/5 vs chỉ kernel 3.
- Dropout 0.3 / 0.5 / 0.7.
- Frozen vs fine-tune PhoBERT.
- Lowercase vs giữ nguyên chữ hoa.
- Có emoji vs bỏ emoji.

### 9.4. Data efficiency

Train với:

- 10%
- 25%
- 50%
- 100%

dữ liệu train, giữ nguyên dev/test.

Mục tiêu: kiểm tra multi-task có lợi hơn khi dữ liệu ít không.

### 9.5. Chế độ suy luận

- **Both-task inference:** dự đoán cả cảm xúc và chủ đề.
- **Sentiment-only inference:** chỉ dự đoán cảm xúc, nhưng train multi-task.

Nếu sentiment-only inference tốt hơn single-task, chứng tỏ chủ đề là tác vụ phụ hữu ích.

---

## 10. Đánh giá

### 10.1. Metric

- Accuracy.
- Macro-F1 cho sentiment.
- Macro-F1 cho topic.
- Precision/recall/F1 từng lớp.
- Confusion matrix.
- **Joint accuracy:** cả hai nhãn đều đúng.
- **Per-topic sentiment F1:** F1 cảm xúc trên từng nhóm chủ đề.

### 10.2. Cách báo cáo

- Chạy 3 seed.
- Báo cáo mean ± std.
- So sánh paired theo cùng split.
- Có thể dùng bootstrap 95% CI cho F1.
- Có thể dùng McNemar test giữa single-task và multi-task.

### 10.3. Phân tích lỗi

Phân loại lỗi:

- Nhầm giữa `positive` và `neutral`.
- Nhầm giữa `negative` và `neutral`.
- Nhầm chủ đề `lecturer` và `training_program`.
- Nhầm `facility` và `others`.
- Câu có nhiều chủ đề.
- Câu mỉa mai.
- Teencode/không dấu.
- Câu quá ngắn.
- Nhãn gốc mơ hồ.

Mỗi loại lấy 3–5 ví dụ, kèm xác suất dự đoán.

---

## 11. Demo

### 11.1. Chức năng

- Nhập một câu.
- Chọn TextCNN / PhoBERT / So sánh.
- Hiển thị:
  - Cảm xúc + confidence.
  - Chủ đề + confidence.
  - Xác suất từng lớp.
  - Token sau tách từ.
- Cảnh báo nếu confidence thấp.
- Tab so sánh hai mô hình.
- Tab lỗi mô hình.

### 11.2. Output mẫu

```json
{
  "text": "Cơ sở vật chất còn yếu, phòng học thiếu máy chiếu",
  "sentiment": {
    "label": "negative",
    "confidence": 0.86,
    "probabilities": {
      "negative": 0.86,
      "neutral": 0.09,
      "positive": 0.05
    }
  },
  "topic": {
    "label": "facility",
    "confidence": 0.78,
    "probabilities": {
      "lecturer": 0.05,
      "training_program": 0.10,
      "facility": 0.78,
      "others": 0.07
    }
  }
}
```

### 11.3. Gợi ý hành động

Có thể dùng rule đơn giản:

| Cảm xúc | Chủ đề | Gợi ý |
|---|---|---|
| negative | facility | Cần cải thiện cơ sở vật chất |
| negative | lecturer | Cần phản hồi về giảng viên |
| negative | training_program | Cần xem lại chương trình đào tạo |
| positive | any | Ghi nhận điểm tích cực |

Ghi rõ đây là gợi ý rule-based, không phải mô hình sinh văn bản.

---

## 12. Cấu trúc mã nguồn

```text
project/
├── data/
│   ├── raw/
│   └── processed/
├── configs/
│   ├── textcnn.yaml
│   └── phobert.yaml
├── src/
│   ├── preprocess.py
│   ├── dataset.py
│   ├── models/
│   │   ├── textcnn.py
│   │   └── phobert.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
├── app/
│   └── streamlit_app.py
├── notebooks/
│   └── 01_eda.ipynb
├── models/
├── reports/
│   ├── figures/
│   └── results.csv
├── requirements.txt
└── README.md
```

---

## 13. Kế hoạch 6 tuần

| Tuần | Nội dung | Sản phẩm |
|---|---|---|
| 1 | Đọc paper, tải data, EDA, label mapping, baseline TF-IDF | Báo cáo EDA + baseline |
| 2 | TextCNN single-task, multi-task, tune `λ` | Kết quả TextCNN |
| 3 | PhoBERT frozen, multi-task, so sánh | Kết quả PhoBERT |
| 4 | Ablation, data efficiency, 3 seed | Bảng kết quả đầy đủ |
| 5 | Phân tích lỗi, demo Streamlit | Demo chạy được |
| 6 | Viết báo cáo, slide, kiểm tra tái lập | Báo cáo + slide |

### Phiên bản tối thiểu 4 tuần

- Tuần 1: data + baseline.
- Tuần 2: TextCNN multi-task.
- Tuần 3: PhoBERT frozen + so sánh.
- Tuần 4: demo + báo cáo.

Bỏ fine-tune PhoBERT, bỏ một phần ablation nếu thiếu thời gian.

---

## 14. Phân công 3 người

| Thành viên | Trách nhiệm chính | Sản phẩm |
|---|---|---|
| **A — Data & Baseline** | EDA, tiền xử lý, label mapping, TF-IDF baseline, kiểm tra dữ liệu | Dataset sạch, baseline, báo cáo EDA |
| **B — TextCNN** | TextCNN single/multi-task, tune `λ`, ablation, learning curve | Kết quả TextCNN |
| **C — PhoBERT & Demo** | PhoBERT frozen/fine-tune, metric, phân tích lỗi, Streamlit | Kết quả PhoBERT + demo |
| **Cả nhóm** | Đọc paper, review code, viết báo cáo, kiểm tra tái lập | Báo cáo + slide |

### Nguyên tắc làm việc

- Họp 1 buổi/tuần.
- Mỗi tuần có một sản phẩm chạy được.
- Không merge code khi chưa có kết quả tái lập.
- Mọi thí nghiệm phải ghi seed, config, metric, thời gian chạy.

---

## 15. Kết quả kỳ vọng

Đây là kỳ vọng để định hướng, không phải cam kết:

| Mô hình | Sentiment F1 | Topic F1 |
|---|---:|---:|
| TF-IDF + LogReg | 0.80–0.85 | 0.78–0.84 |
| TextCNN single-task | 0.82–0.86 | 0.80–0.84 |
| TextCNN multi-task | 0.83–0.87 | 0.80–0.85 |
| PhoBERT frozen multi-task | 0.86–0.90 | 0.84–0.88 |
| PhoBERT fine-tune multi-task | 0.88–0.92 | 0.85–0.90 |

Kỳ vọng về multi-task:

- Có thể tăng nhẹ F1, hoặc không tăng.
- Có lợi rõ hơn khi dữ liệu ít (10–25%).
- Có thể xảy ra negative transfer nếu `λ` quá lớn.
- TextCNN có thể hưởng lợi nhiều hơn PhoBERT vì ít tham số hơn.

---

## 16. Rủi ro và cách giảm thiểu

| Rủi ro | Mức độ | Cách xử lý |
|---|---|---|
| Negative transfer | Cao | Tune `λ`, early stopping, thử shared encoder nhỏ hơn |
| CPU quá chậm với PhoBERT | Cao | Frozen + trích embedding một lần; chỉ fine-tune trên Colab |
| Mất cân bằng nhãn | Trung bình | Class weight, macro-F1, per-class report |
| Label mapping sai | Trung bình | Unit test, in ví dụ từng lớp |
| Rò rỉ dữ liệu | Trung bình | Kiểm tra trùng lặp, dùng split chính thức |
| Overfitting | Trung bình | Dropout, early stopping, 3 seed |
| Demo lỗi khi bảo vệ | Trung bình | Video dự phòng, chạy local, cache model |
| Scope creep | Cao | Bám mục tiêu, bỏ tính năng không cần thiết |

---

## 17. Đạo đức và hạn chế

- Dataset có thể chứa phản hồi nhạy cảm; không công bố dữ liệu thô.
- Kết quả không dùng để đánh giá cá nhân giảng viên.
- Mô hình có thể sai với mỉa mai, teencode, câu nhiều chủ đề.
- Chủ đề là nhãn mức câu, không phải cảm xúc theo từng khía cạnh.
- Cần có con người kiểm tra trước khi dùng cho quyết định thực tế.
- Báo cáo rõ giấy phép dataset và giới hạn của nghiên cứu.

---

## 18. Tài liệu tham khảo

1. Nguyen, K. V., Nguyen, V. D., Nguyen, P. X. V., Truong, T. T. H., & Nguyen, N. L. T. (2018). **UIT-VSFC: Vietnamese Students’ Feedback Corpus for Sentiment Analysis**. KSE 2018.
2. Nguyen, D. Q., & Nguyen, A. T. (2020). **PhoBERT: Pre-trained language models for Vietnamese**. Findings of EMNLP 2020. https://arxiv.org/abs/2003.00744
3. Kim, Y. (2014). **Convolutional Neural Networks for Sentence Classification**. EMNLP 2014. https://arxiv.org/abs/1408.5882
4. Caruana, R. (1997). **Multitask Learning**. Machine Learning, 28, 41–75.
5. Ruder, S. (2017). **An Overview of Multi-Task Learning in Deep Neural Networks**. https://arxiv.org/abs/1706.05098
6. Kendall, A., Gal, Y., & Cipolla, R. (2018). **Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics**. CVPR 2018. https://arxiv.org/abs/1705.07115
7. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. NAACL 2019.
8. UIT-VSFC repository: https://github.com/kietnv/uit-vsfc

---

## 19. Phụ lục A — Bảng hyperparameter dự kiến

| Tham số | TextCNN | PhoBERT frozen | PhoBERT fine-tune |
|---|---:|---:|---:|
| Max length | 128 | 128 | 128 |
| Batch size | 32 | 32 | 8–16 |
| Learning rate | 1e-3 | 1e-3 (head) | 2e-5 |
| Epochs | 20 | 20 | 3–5 |
| Dropout | 0.5 | 0.3 | 0.3 |
| Optimizer | Adam | Adam | AdamW |
| Weight decay | 1e-4 | 1e-4 | 0.01 |
| Early stopping | 3 | 3 | 2 |

---

## 20. Phụ lục B — Bảng log thí nghiệm

| Run ID | Model | λ | Seed | Sent F1 | Topic F1 | Joint Acc | Time |
|---|---|---:|---:|---:|---:|---:|---:|
| T01 | TextCNN single | — | 42 |  |  |  |  |
| T02 | TextCNN multi | 0.5 | 42 |  |  |  |  |
| T03 | PhoBERT multi | 0.5 | 42 |  |  |  |  |

---

## 21. Phụ lục C — Tiêu chí hoàn thành

- [ ] Dataset được kiểm tra và label mapping đúng.
- [ ] Có baseline TF-IDF cho cả hai tác vụ.
- [ ] Có TextCNN single-task và multi-task.
- [ ] Có PhoBERT frozen multi-task.
- [ ] Có ít nhất 3 seed cho mô hình chính.
- [ ] Có bảng ablation `λ`.
- [ ] Có learning curve 10/25/50/100%.
- [ ] Có confusion matrix và phân tích lỗi.
- [ ] Có demo Streamlit chạy local.
- [ ] Có README tái lập.
- [ ] Có báo cáo và slide.
