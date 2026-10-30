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
- [x] TextCNN single-task
- [x] TextCNN multi-task
- [x] PhoBERT multi-task
- [x] Demo Streamlit
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

## TextCNN single-task

Chạy:

```powershell
python -m src.train_textcnn --task both
```

Smoke test:

```powershell
python -m src.train_textcnn --task both --smoke-test
```

Kết quả hiện tại:

| Task | Split | Accuracy | Macro-F1 | So với baseline |
|---|---|---:|---:|---:|
| sentiment | dev | 0.9172 | 0.8077 | +0.0498 |
| sentiment | test | 0.8932 | 0.7621 | +0.0221 |
| topic | dev | 0.8345 | 0.7384 | +0.0083 |
| topic | test | 0.8301 | 0.7355 | +0.0108 |
| joint | test | 0.7536 | — | +0.0309 |

Điểm đáng chú ý:

- `neutral` F1 tăng từ 0.4087 lên 0.4490.
- `others` F1 tăng từ 0.4336 lên 0.4444.
- Best epoch: sentiment 9, topic 5.
- Thời gian train CPU: khoảng 75 giây cho sentiment, 49 giây cho topic.
- Checkpoint nằm trong `models/textcnn/` và không commit.

## TextCNN multi-task

Chạy:

```powershell
python -m src.train_textcnn_multitask
```

Kết quả với `lambda_topic = 1.0`:

| Task | Split | Accuracy | Macro-F1 |
|---|---|---:|---:|
| sentiment | dev | 0.9122 | 0.7997 |
| sentiment | test | 0.8866 | 0.7556 |
| topic | dev | 0.8465 | 0.7455 |
| topic | test | 0.8370 | 0.7390 |
| joint | test | 0.7552 | — |

So với single-task:

- Sentiment giảm nhẹ: **0.7621 → 0.7556**.
- Topic tăng nhẹ: **0.7355 → 0.7390**.
- Joint tăng nhẹ: **0.7536 → 0.7552**.

Lambda sweep:

| Lambda | Dev mean-F1 | Test sentiment | Test topic | Test joint |
|---:|---:|---:|---:|---:|
| 0.5 | 0.7693 | 0.7639 | 0.7175 | 0.7388 |
| 1.0 | **0.7726** | 0.7556 | 0.7390 | 0.7552 |
| 2.0 | 0.7724 | 0.7598 | 0.7496 | 0.7641 |

`lambda = 1.0` được chọn theo **dev mean-F1**, không theo test.

## PhoBERT frozen multi-task

Chạy:

```powershell
python -m src.train_phobert_multitask
```

Lần đầu sẽ trích `[CLS]` embedding cho toàn bộ dataset và cache trong `data/processed/uit_vsfc_phobert/`.

Kết quả:

| Task | Split | Accuracy | Macro-F1 |
|---|---|---:|---:|
| sentiment | dev | 0.9109 | 0.7665 |
| sentiment | test | 0.9040 | **0.7785** |
| topic | dev | 0.8598 | 0.7673 |
| topic | test | 0.8509 | **0.7428** |
| joint | test | 0.7855 | — |

So sánh test:

| Model | Sentiment | Topic | Joint |
|---|---:|---:|---:|
| TF-IDF | 0.7400 | 0.7247 | 0.7227 |
| TextCNN single | 0.7621 | 0.7355 | 0.7536 |
| TextCNN multi | 0.7556 | 0.7390 | 0.7552 |
| PhoBERT frozen multi | **0.7785** | **0.7428** | **0.7855** |

- Best epoch: 15.
- Chỉ train head: **17.9 giây** sau khi đã cache embedding.
- Không fine-tune PhoBERT.

## Demo Streamlit

Chạy:

```powershell
.\scripts\run_demo.ps1
```

Mở:

```text
http://127.0.0.1:8501
```

Script này đặt `USERPROFILE` và cache Streamlit vào `.demo-home/` để tránh lỗi
`Access is denied: C:\Users\<user>\.streamlit` trong môi trường bị giới hạn quyền.

Nếu chạy trực tiếp:

```powershell
.\scripts\activate.ps1
streamlit run app\streamlit_app.py
```

có thể gặp lỗi trên; khi đó dùng `scripts\run_demo.ps1`.

Demo gồm:

- Nhập một câu → dự đoán cảm xúc + chủ đề.
- Hiển thị confidence và xác suất từng lớp.
- Cảnh báo khi confidence thấp.
- Hiển thị token sau tách từ.
- Tab so sánh kết quả test của các mô hình.
- Tab giới thiệu và hạn chế.

Có thể test nhanh bằng CLI:

```powershell
python -m src.predict --text "Cơ sở vật chất còn yếu, phòng học thiếu máy chiếu"
```

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
