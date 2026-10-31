# Calibration report

- Temperature scaling fit trên dev; đánh giá trên test.
- ECE: Expected Calibration Error, 15 bins.
- Temperature không làm thay đổi argmax/accuracy.

## Test

| Model | Task | Temperature | ECE before | ECE after | NLL before | NLL after |
|---|---|---:|---:|---:|---:|---:|
| teacher | sentiment | 1.3660 | 0.0242 | 0.0215 | 0.2894 | 0.2673 |
| teacher | topic | 1.2301 | 0.0254 | 0.0097 | 0.3585 | 0.3504 |
| student | sentiment | 1.9300 | 0.0658 | 0.0166 | 0.4173 | 0.3047 |
| student | topic | 1.8234 | 0.0805 | 0.0145 | 0.4669 | 0.3886 |

## Figures

- `reports/figures/calibration_sentiment_reliability.png`
- `reports/figures/calibration_topic_reliability.png`

## Nhận xét

- Trước calibration, student overconfident hơn teacher rõ rệt.
- Sau temperature scaling, ECE của student giảm mạnh ở cả sentiment và topic.
- Teacher topic cải thiện rõ; teacher sentiment cải thiện ít hơn.
- Accuracy không đổi vì temperature scaling chỉ sửa confidence.

## Cách đọc

- ECE càng thấp càng tốt.
- Temperature > 1 thường làm giảm overconfidence.
- Nếu ECE giảm nhưng accuracy không đổi, calibration chỉ sửa confidence, không sửa dự đoán.
