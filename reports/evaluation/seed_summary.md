# Three-seed results

Seeds: 42, 123, 2024. Mỗi seed train lại từ đầu; test chỉ đánh giá một lần.

## Test — mean ± std

| Model | Sentiment F1 | Topic F1 | Joint accuracy |
|---|---:|---:|---:|
| textcnn_multitask | 0.7559 ± 0.0024 | 0.7508 ± 0.0103 | 0.7654 ± 0.0090 |
| phobert_frozen_multitask | 0.7679 ± 0.0101 | 0.7388 ± 0.0082 | 0.7612 ± 0.0291 |
| phobert_keyword_multitask | 0.7760 ± 0.0060 | 0.7597 ± 0.0067 | 0.7989 ± 0.0059 |

## Dev — mean ± std

| Model | Sentiment F1 | Topic F1 | Joint accuracy |
|---|---:|---:|---:|
| textcnn_multitask | 0.7972 ± 0.0072 | 0.7504 ± 0.0062 | 0.7903 ± 0.0061 |
| phobert_frozen_multitask | 0.7663 ± 0.0078 | 0.7519 ± 0.0155 | 0.7734 ± 0.0293 |
| phobert_keyword_multitask | 0.7845 ± 0.0051 | 0.7746 ± 0.0077 | 0.8130 ± 0.0064 |

## Per-seed test

| Model | Seed | Sentiment F1 | Topic F1 | Joint | Best epoch |
|---|---:|---:|---:|---:|---:|
| phobert_frozen_multitask | 42 | 0.7785 | 0.7428 | 0.7855 | 15 |
| phobert_frozen_multitask | 123 | 0.7666 | 0.7443 | 0.7691 | 9 |
| phobert_frozen_multitask | 2024 | 0.7585 | 0.7294 | 0.7290 | 5 |
| phobert_keyword_multitask | 42 | 0.7788 | 0.7633 | 0.8054 | 13 |
| phobert_keyword_multitask | 123 | 0.7692 | 0.7638 | 0.7972 | 15 |
| phobert_keyword_multitask | 2024 | 0.7801 | 0.7519 | 0.7941 | 10 |
| textcnn_multitask | 42 | 0.7556 | 0.7390 | 0.7552 | 10 |
| textcnn_multitask | 123 | 0.7538 | 0.7575 | 0.7723 | 10 |
| textcnn_multitask | 2024 | 0.7585 | 0.7560 | 0.7688 | 5 |

## Nhận xét

- PhoBERT keyword-aware có mean topic-F1 và joint accuracy cao nhất.
- Phương sai giữa các seed là đáng kể, đặc biệt với PhoBERT frozen.
- Không dùng test để chọn seed hay hyperparameter.
