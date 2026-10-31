# Distillation summary — PhoBERT → TextCNN

## Thiết lập

- Teacher: PhoBERT keyword-aware multi-task.
- Student: TextCNN multi-task.
- Temperature: `T = 2.0`.
- Distill weight: `0.5`.
- Lambda topic: `1.0`.
- Seeds: 42, 123, 2024.
- Test set: 3.166 câu; test chỉ đánh giá một lần mỗi seed.

## Test — mean ± std

| Model | Sentiment F1 | Topic F1 | Joint accuracy |
|---|---:|---:|---:|
| PhoBERT teacher | **0.7760 ± 0.0060** | 0.7597 ± 0.0067 | **0.7989 ± 0.0059** |
| TextCNN hard-label | 0.7559 ± 0.0024 | 0.7508 ± 0.0103 | 0.7654 ± 0.0090 |
| **TextCNN distilled** | 0.7627 ± 0.0047 | **0.7607 ± 0.0102** | 0.7766 ± 0.0087 |

## So với TextCNN hard-label

| Metric | Chênh lệch |
|---|---:|
| Sentiment F1 | **+0.0068** |
| Topic F1 | **+0.0099** |
| Joint accuracy | **+0.0112** |

Distillation giúp student tăng đều cả 3 metric so với train hard-label.

## So với teacher

| Metric | Chênh lệch student - teacher |
|---|---:|
| Sentiment F1 | -0.0134 |
| Topic F1 | +0.0010 |
| Joint accuracy | -0.0223 |

Student gần như ngang teacher ở topic, thấp hơn ở sentiment/joint, nhưng nhỏ và nhanh hơn rất nhiều.

## Hiệu năng CPU

Đo trên 100 câu test, cùng CPU:

| Metric | Teacher PhoBERT | Student TextCNN | Tỷ lệ |
|---|---:|---:|---:|
| Parameters | 135.198.571 | 407.055 | **332× ít hơn** |
| Estimated size | 515.74 MB | 1.55 MB | **332× nhỏ hơn** |
| Latency | 51.14 ms/câu | 1.98 ms/câu | **25.9× nhanh hơn** |
| Throughput | 19.6 câu/s | 505.9 câu/s | **25.9×** |

Lưu ý: checkpoint teacher trong repo chỉ chứa head; kích thước teacher ở bảng trên là ước lượng từ số tham số của encoder + head.

## Per-seed test

| Seed | Sentiment F1 | Topic F1 | Joint |
|---:|---:|---:|---:|
| 42 | 0.7674 | 0.7490 | 0.7669 |
| 123 | 0.7626 | 0.7672 | 0.7836 |
| 2024 | 0.7581 | 0.7660 | 0.7792 |

## Kết luận

- Distillation thành công về mục tiêu **nén mô hình**.
- Student nhỏ hơn ~332 lần, nhanh hơn ~26 lần trên CPU.
- F1 giảm không nhiều so với teacher; topic gần như tương đương.
- Student tốt hơn TextCNN hard-label trên cả 3 metric trung bình.
- Đây là trade-off hợp lý cho triển khai CPU.

## File

- [reports/textcnn_distill_report.md](textcnn_distill_report.md)
- [reports/distill_seed_results.csv](distill_seed_results.csv)
- [reports/distillation_benchmark.json](distillation_benchmark.json)
- [src/train_textcnn_distill.py](../src/train_textcnn_distill.py)
- [src/benchmark_distill.py](../src/benchmark_distill.py)
