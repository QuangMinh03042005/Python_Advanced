# Reports index

Thư mục này chứa báo cáo, metric, hình và kết quả thí nghiệm.

```text
reports/
├── README.md
├── results.csv                  # bảng tổng hợp mọi model
├── data/                        # data audit + EDA
├── baselines/tfidf/             # TF-IDF + Logistic Regression
├── textcnn/
│   ├── single/                  # TextCNN single-task
│   ├── multitask/               # TextCNN multi-task
│   └── distilled/               # TextCNN student distillation
├── phobert/
│   ├── frozen/                  # PhoBERT frozen multi-task
│   └── keyword/                 # PhoBERT keyword-aware
├── distillation/                # teacher labels, benchmark, tổng hợp
├── evaluation/                  # 3 seed, thống kê, calibration, robustness
└── plans/                       # kế hoạch nâng cấp
```

## Đọc nhanh

| Nội dung | Đường dẫn |
|---|---|
| Kết quả tổng hợp | [`results.csv`](results.csv) |
| Data audit | [`data/data_summary.md`](data/data_summary.md) |
| EDA | [`data/eda_summary.md`](data/eda_summary.md) |
| Baseline | [`baselines/tfidf/baseline_report.md`](baselines/tfidf/baseline_report.md) |
| TextCNN single | [`textcnn/single/textcnn_single_report.md`](textcnn/single/textcnn_single_report.md) |
| TextCNN multi | [`textcnn/multitask/textcnn_multitask_report.md`](textcnn/multitask/textcnn_multitask_report.md) |
| TextCNN distilled | [`textcnn/distilled/textcnn_distill_report.md`](textcnn/distilled/textcnn_distill_report.md) |
| PhoBERT frozen | [`phobert/frozen/phobert_multitask_report.md`](phobert/frozen/phobert_multitask_report.md) |
| PhoBERT keyword | [`phobert/keyword/phobert_keyword_multitask_report.md`](phobert/keyword/phobert_keyword_multitask_report.md) |
| Distillation | [`distillation/distillation_summary.md`](distillation/distillation_summary.md) |
| 3-seed | [`evaluation/seed_summary.md`](evaluation/seed_summary.md) |
| Statistical tests | [`evaluation/statistical_tests.md`](evaluation/statistical_tests.md) |
| Calibration | [`evaluation/calibration_report.md`](evaluation/calibration_report.md) |
| Robustness | [`evaluation/robustness_report.md`](evaluation/robustness_report.md) |
| Kế hoạch nâng cấp | [`plans/improvement_plan.md`](plans/improvement_plan.md) |

## Quy ước

- Mỗi model có thư mục riêng, chứa `report.md`, `metrics.json`, history, per-class CSV và `figures/`.
- `results.csv` chỉ nằm ở root để mọi trainer cùng ghi/đọc.
- Hình của báo cáo nằm trong `figures/` cùng thư mục với report để link tương đối ổn định.
