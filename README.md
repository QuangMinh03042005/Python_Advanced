# UIT-VSFC Multi-task Learning

Đồ án môn Học sâu: phân loại đồng thời **cảm xúc** và **chủ đề** phản hồi sinh viên tiếng Việt bằng TextCNN và PhoBERT.

## Trạng thái

- [x] Môi trường Python 3.12 + PyTorch CPU
- [x] Dependency đã cài trong `.venv`
- [x] Đã kiểm tra import, TextCNN forward pass, PhoBERT tokenizer/model
- [x] Đã scaffold thư mục dự án
- [x] Đã tải UIT-VSFC và viết data loader
- [x] Đã xác minh label mapping và data audit
- [x] EDA: phân bố nhãn, độ dài, trùng lặp, tương quan sentiment × topic
- [x] Baseline TF-IDF + Logistic Regression
- [ ] TextCNN multi-task
- [ ] PhoBERT multi-task
- [ ] Demo Streamlit
- [ ] Báo cáo

## Yêu cầu

- Windows PowerShell
- Python **3.12.x** — môi trường hiện tại đã ổn định với PyTorch CPU
- CPU là đủ; không cần CUDA

## Kích hoạt môi trường

```powershell
.\.venv\Scripts\Activate.ps1
```

Hoặc dùng script tiện ích:

```powershell
.\scripts\activate.ps1
```

Kiểm tra nhanh:

```powershell
python -c "import torch, transformers, sklearn, underthesea; print(torch.__version__)"
```

## Cài lại dependency

Nếu đã có Python 3.12 và muốn tạo lại môi trường:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

Nếu cần tái lập chính xác toàn bộ package:

```powershell
pip install -r requirements-lock.txt
```

## Biến môi trường cho Hugging Face

Trong môi trường sandbox hiện tại, nên dùng cache trong workspace và tắt symlink:

```powershell
$env:HF_HOME = (Resolve-Path .hf-cache).Path
$env:HF_HUB_DISABLE_SYMLINKS = "1"
$env:PYTHONIOENCODING = "utf-8"
```

Script `scripts\activate.ps1` đã set các biến này.

## Dữ liệu

Tải UIT-VSFC:

```powershell
python scripts\download_uit_vsfc.py
```

Kiểm tra file:

```powershell
python scripts\download_uit_vsfc.py --check
```

Load thử:

```powershell
python -m src.data.load
```

Kết quả mong đợi:

```text
train: 11426 rows
dev: 1583 rows
test: 3166 rows
total: 16175 rows
```

Label mapping nằm trong `configs/label_map.json`. Báo cáo kiểm tra dữ liệu nằm trong `reports/data_summary.md`.

## EDA

Chạy:

```powershell
python -m src.data.eda
```

Output:

```text
reports/eda_summary.md
reports/eda_sentiment_distribution.csv
reports/eda_topic_distribution.csv
reports/eda_sentiment_topic_crosstab.csv
reports/eda_length_stats.csv
reports/figures/
```

Kết quả chính:

- p95 độ dài câu train: **33 token**, gợi ý `max_len = 48`.
- `neutral` chỉ chiếm **4.0%** câu train.
- `lecturer` chiếm **71.5%** câu train.
- `facility` gần như luôn tiêu cực: **474/497** câu.
- Sentiment và topic có tương quan mạnh; cần cẩn thận shortcut khi multi-task.
- Có 1 câu trùng trong train, không có trùng giữa các split.

## Baseline TF-IDF

Chạy:

```powershell
python -m src.models.baseline
```

Output:

```text
reports/baseline_report.md
reports/baseline_metrics.json
reports/results.csv
reports/baseline_*_per_class.csv
reports/figures/baseline_*_confusion.png
```

Kết quả hiện tại:

| Task | Split | Accuracy | Macro-F1 |
|---|---|---:|---:|
| sentiment | dev | 0.8857 | 0.7579 |
| sentiment | test | 0.8733 | 0.7400 |
| topic | dev | 0.8244 | 0.7301 |
| topic | test | 0.8105 | 0.7247 |
| joint | test | 0.7227 | — |

Nhận xét: accuracy cao nhưng macro-F1 thấp hơn đáng kể do `neutral` và `others` ít mẫu.

## Cấu trúc dự kiến

```text
configs/        # YAML hyperparameters
data/raw/       # dữ liệu gốc, không sửa
data/processed/ # dữ liệu đã xử lý
src/            # code huấn luyện
app/            # Streamlit demo
tests/          # unit tests
notebooks/      # EDA
scripts/        # tiện ích
reports/        # bảng, hình, kết quả
models/         # checkpoint, không commit
runs/           # log thí nghiệm, không commit
```

## Kiểm tra môi trường

```powershell
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "from transformers import AutoTokenizer; print(type(AutoTokenizer.from_pretrained('vinai/phobert-base')).__name__)"
```

Kỳ vọng:

```text
torch 2.14.1+cpu cuda False
PhobertTokenizer
```

## Ghi chú

- Không commit `.venv`, `.python`, `.tools`, `.hf-cache`, dữ liệu, model hoặc file checkpoint.
- Quy trình làm việc và Definition of Done nằm trong `AGENTS.md`.
- Proposal chi tiết nằm trong `proposal_multi_task_uit_vsfc.md`.
