# Robustness report

- Test sample: `800` câu, random seed 42.
- Nhãn giữ nguyên; chỉ thay đổi input.
- Teacher: PhoBERT keyword-aware.
- Student: TextCNN distilled.

## Kết quả

| Variant | Model | Sentiment F1 | Topic F1 | Joint | Δ Sent | Δ Topic | Δ Joint |
|---|---|---:|---:|---:|---:|---:|---:|
| clean | teacher | 0.7902 | 0.7727 | 0.8200 | +0.0000 | +0.0000 | +0.0000 |
| clean | student | 0.7640 | 0.7322 | 0.7712 | +0.0000 | +0.0000 | +0.0000 |
| no_diacritics | teacher | 0.3292 | 0.1517 | 0.0813 | -0.4610 | -0.6210 | -0.7387 |
| no_diacritics | student | 0.1923 | 0.1165 | 0.0338 | -0.5716 | -0.6157 | -0.7375 |
| teencode | teacher | 0.7535 | 0.6833 | 0.7137 | -0.0367 | -0.0894 | -0.1062 |
| teencode | student | 0.7201 | 0.6303 | 0.6238 | -0.0438 | -0.1019 | -0.1475 |
| typo | teacher | 0.7596 | 0.7699 | 0.7762 | -0.0306 | -0.0028 | -0.0437 |
| typo | student | 0.7158 | 0.7005 | 0.7300 | -0.0481 | -0.0317 | -0.0413 |
| emoji | teacher | 0.5506 | 0.6661 | 0.6525 | -0.2396 | -0.1066 | -0.1675 |
| emoji | student | 0.7582 | 0.7440 | 0.7675 | -0.0058 | +0.0118 | -0.0038 |
| shuffle | teacher | 0.6281 | 0.6678 | 0.6538 | -0.1621 | -0.1049 | -0.1662 |
| shuffle | student | 0.7116 | 0.7099 | 0.7288 | -0.0524 | -0.0223 | -0.0425 |

## Figures

- `reports/evaluation/figures/robustness_scores.png`

## Nhận xét

- `no_diacritics` gây sụt giảm nghiêm trọng nhất cho cả hai mô hình.
- `teencode` làm topic giảm mạnh hơn sentiment.
- `emoji` làm teacher sentiment giảm rất mạnh; student ổn định hơn.
- `shuffle` làm teacher giảm mạnh hơn student, cho thấy teacher phụ thuộc trật tự từ.

## Cách đọc

- Δ càng âm ⇒ mô hình càng kém bền vững với perturbation đó.
- `shuffle` kiểm tra phụ thuộc trật tự từ.
- `no_diacritics` kiểm tra thiếu dấu.
- `teencode` kiểm tra viết tắt.
- `typo` kiểm tra lỗi gõ.
- `emoji` kiểm tra nhiễu ký tự ngoài từ vựng.
