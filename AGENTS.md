# AGENTS.md

## 1. Mục đích

File này quy định **quy trình làm việc, tiêu chuẩn code, cách chạy thí nghiệm và Definition of Done** cho dự án.

Mọi agent (AI hoặc người) làm việc trong repo này phải đọc file này trước khi sửa code, chạy thí nghiệm hoặc tạo tài liệu.

Thứ tự ưu tiên khi có mâu thuẫn:

1. Yêu cầu trực tiếp của người dùng.
2. `AGENTS.md`.
3. `README.md`.
4. Code và tài liệu hiện có.

---

## 2. Bối cảnh dự án

**Tên đề tài:** Multi-task Learning cho Phân loại Cảm xúc và Chủ đề Phản hồi Sinh viên Tiếng Việt: So sánh TextCNN và PhoBERT.

**Bài toán:**

- Input: một câu phản hồi sinh viên tiếng Việt.
- Output 1: cảm xúc — `negative`, `neutral`, `positive`.
- Output 2: chủ đề — `lecturer`, `training_program`, `facility`, `others`.

**Dataset:** UIT-VSFC.

**Mô hình:**

- TextCNN multi-task.
- PhoBERT multi-task, ưu tiên frozen + head.
- Baseline: TF-IDF + Logistic Regression, single-task TextCNN, single-task PhoBERT.

**Điều kiện:**

- Nhóm 3 người.
- Không có GPU.
- Ưu tiên pipeline chạy được trên CPU.
- Có thể dùng Colab/Kaggle miễn phí cho fine-tune PhoBERT, nhưng không được xem là điều kiện bắt buộc.

**Ngoài phạm vi:**

- Aspect-based sentiment.
- Multi-label.
- Fine-tune LLM.
- Dữ liệu đa ngôn ngữ.
- Streaming/real-time.
- Triển khai production quy mô lớn.

---

## 3. Nguyên tắc bất biến

### 3.1. Không bịa kết quả

- Không được báo cáo metric, bảng kết quả hoặc nhận xét thực nghiệm nếu chưa chạy.
- Nếu chưa chạy, ghi rõ: `Chưa chạy`, `Chưa có kết quả`, hoặc `Cần kiểm chứng`.
- Không được tự tạo số liệu cho báo cáo.

### 3.2. Không rò rỉ dữ liệu

- Không dùng test set để tune hyperparameter.
- Vocabulary, tokenizer, vectorizer chỉ được fit trên train.
- Kiểm tra câu trùng lặp giữa train/dev/test.
- Không thay đổi split chính thức nếu không có lý do và ghi lại.

### 3.3. Không hardcode nhãn

- Không hardcode thứ tự nhãn theo cảm tính.
- Phải đọc label mapping từ file gốc hoặc `label_map.json`.
- Phải có unit test kiểm tra số lớp và mapping.

### 3.4. Không sửa dữ liệu thô

- `data/raw/` là bất biến.
- Mọi biến đổi dữ liệu phải nằm trong script ở `src/data/`.
- Không chỉnh tay file trong `data/processed/`.

### 3.5. Không commit dữ liệu hoặc model lớn

Không commit:

- `data/raw/`
- `data/processed/`
- `models/`
- `runs/`
- `.venv/`
- file `.pt`, `.pth`, `.bin`, `.npy` lớn
- API key, token, mật khẩu

### 3.6. CPU-first

- Không viết code chỉ chạy trên CUDA.
- Luôn có fallback CPU:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

- Không thêm model quá lớn nếu chưa hỏi người dùng.
- Không mặc định fine-tune toàn bộ PhoBERT trên CPU.

### 3.7. Thay đổi nhỏ, có kiểm chứng

- Mỗi lần chỉ giải quyết một việc.
- Không refactor lan sang file không liên quan.
- Không đổi API công khai nếu chưa thông báo.
- Không xóa file hoặc dữ liệu khi chưa kiểm tra đường dẫn tuyệt đối.

---

## 4. Cấu trúc repo mục tiêu

```text
project/
├── AGENTS.md
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── textcnn.yaml
│   └── phobert.yaml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load.py
│   │   ├── preprocess.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── textcnn.py
│   │   └── phobert.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
├── app/
│   └── streamlit_app.py
├── tests/
│   ├── test_preprocess.py
│   ├── test_dataset.py
│   ├── test_model_shapes.py
│   └── test_metrics.py
├── models/
├── runs/
├── reports/
│   ├── figures/
│   └── results.csv
└── scripts/
    ├── download_data.ps1
    └── smoke_test.ps1
```

Nếu cấu trúc thay đổi, phải cập nhật file này và `README.md`.

---

## 5. Quy trình làm việc của agent

### Bước 1 — Đọc ngữ cảnh

Trước khi làm:

- Đọc `AGENTS.md`.
- Đọc `README.md` nếu có.
- Đọc proposal `proposal_multi_task_uit_vsfc.md` nếu liên quan.
- Kiểm tra `git status`.
- Kiểm tra file hiện có trước khi tạo mới hoặc ghi đè.

### Bước 2 — Lập kế hoạch

Với việc có từ 3 bước trở lên:

- Tạo todo list.
- Ghi rõ file sẽ sửa.
- Ghi rõ lệnh sẽ chạy.
- Ghi rõ tiêu chí hoàn thành.

Không bắt đầu sửa code khi chưa xác định được phạm vi.

### Bước 3 — Thực hiện thay đổi nhỏ nhất

- Ưu tiên sửa file hiện có hơn tạo file mới.
- Giữ diff gọn.
- Không đổi format toàn bộ file nếu không cần.
- Không thêm dependency nếu thư viện hiện có làm được.

### Bước 4 — Kiểm tra

Tối thiểu phải chạy:

```powershell
python -m compileall src
pytest -q
```

Nếu có sửa model hoặc dữ liệu, chạy thêm smoke test:

```powershell
python -m src.train_textcnn --task both --smoke-test --reports-dir .tmp\smoke_reports --models-dir .tmp\smoke_models --runs-dir .tmp\smoke_runs
```

Nếu chưa có test hoặc chưa có script, ghi rõ phần nào chưa kiểm tra.

### Bước 5 — Chạy thí nghiệm

- Chạy baseline trước.
- Chỉ thay một biến mỗi lần.
- Lưu config, seed, metric, thời gian chạy.
- Không tune trên test.

### Bước 6 — Cập nhật tài liệu

Cập nhật:

- `README.md` nếu thay đổi cách chạy.
- `reports/results.csv` nếu có kết quả mới.
- `AGENTS.md` nếu quy trình thay đổi.
- Proposal/report nếu kết luận thay đổi.

### Bước 7 — Báo cáo

Câu trả lời cuối phải có:

- Đã làm gì.
- File nào thay đổi.
- Lệnh nào đã chạy.
- Kết quả/kiểm tra.
- Phần nào chưa làm hoặc còn rủi ro.
- Bước tiếp theo.

Không chỉ nói “đã xong” mà không có bằng chứng.

---

## 6. Quy trình thí nghiệm

### 6.1. Trước khi chạy

- Xác định baseline.
- Xác định metric.
- Xác định split.
- Xác định seed.
- Xác định thời gian dự kiến.
- Kiểm tra config.

### 6.2. Trong khi chạy

- Log ra file, không chỉ in màn hình.
- Lưu:

```text
run_id
model
task
lambda
seed
git_commit
sent_f1
topic_f1
joint_accuracy
train_time
inference_ms
```

- Lưu prediction của test set để phân tích lỗi.
- Lưu confusion matrix.
- Lưu learning curve nếu có.

### 6.3. Sau khi chạy

- So sánh với baseline.
- Chạy ít nhất 3 seed cho mô hình chính.
- Báo cáo mean ± std.
- Kiểm tra negative transfer.
- Ghi rõ nếu kết quả không như kỳ vọng.

### 6.4. Quy tắc đặt tên run

```text
runs/{model}_{task}_{lambda}_{seed}_{timestamp}
```

Ví dụ:

```text
runs/textcnn_multitask_l0.5_s42_20261026_1400
```

### 6.5. Quy tắc chọn model

- Chọn model bằng dev set.
- Chỉ đánh giá test **một lần** cho cấu hình cuối.
- Không chọn model dựa trên test.
- Không báo cáo kết quả test của nhiều cấu hình rồi chọn cái tốt nhất.

---

## 7. Quy trình dữ liệu

### 7.1. Load dữ liệu

- Dùng `pathlib.Path`.
- Không hardcode đường dẫn tuyệt đối.
- Kiểm tra file tồn tại trước khi đọc.
- Kiểm tra encoding UTF-8.
- Kiểm tra số dòng của sentences và labels khớp nhau.

### 7.2. Label mapping

- Tạo `label_map.json`.
- Ghi rõ index → tên nhãn.
- Kiểm tra bằng unit test.
- In 5 ví dụ mỗi lớp để kiểm tra thủ công.

### 7.3. Tiền xử lý

- Chuẩn hóa Unicode NFC.
- Chuẩn hóa khoảng trắng.
- Tách từ nhất quán giữa train và inference.
- Với PhoBERT, dùng đúng hướng dẫn tokenizer.
- Không làm sạch quá mức khiến mất tín hiệu cảm xúc.

### 7.4. Chia dữ liệu

- Dùng split chính thức nếu có.
- Không trộn test vào train.
- Kiểm tra trùng lặp trước khi train.
- Nếu phải chia lại, dùng stratified split và ghi lại lý do.

---

## 8. Tiêu chuẩn code

### 8.1. Python

- Python 3.10+.
- Dùng type hints cho hàm công khai.
- Dùng docstring ngắn gọn.
- Dùng `pathlib` thay cho chuỗi đường dẫn.
- Dùng `logging` thay cho `print` trong pipeline.
- Không dùng biến toàn cục tùy tiện.
- Không hardcode hyperparameter trong model; đưa vào config.

### 8.2. PyTorch

- `model.train()` khi train.
- `model.eval()` và `torch.no_grad()` khi evaluate/inference.
- Không quên `optimizer.zero_grad()`.
- Lưu `state_dict`, không chỉ lưu object.
- Khi load, map `location` phù hợp CPU.
- Đặt seed cho `random`, `numpy`, `torch`.

### 8.3. Config

- Dùng YAML hoặc dataclass.
- CLI chỉ override các giá trị cần thiết.
- Mỗi run phải lưu config thực tế đã dùng.

### 8.4. Notebook

- Notebook chỉ dùng cho EDA và phân tích.
- Code huấn luyện phải nằm trong `src/`.
- Không để logic quan trọng chỉ tồn tại trong notebook.

### 8.5. Format và lint

Khuyến nghị:

```powershell
ruff check .
ruff format .
```

Nếu repo chưa cài `ruff`, không bắt buộc, nhưng phải giữ code nhất quán.

---

## 9. Kiểm thử

### 9.1. Test bắt buộc

- `test_preprocess.py`: Unicode, tách từ, câu rỗng.
- `test_dataset.py`: số mẫu, shape, label range.
- `test_model_shapes.py`: forward pass trả đúng `(B, 3)` và `(B, 4)`.
- `test_metrics.py`: accuracy, F1, joint accuracy.
- `test_label_map.py`: mapping đúng, không thiếu lớp.

### 9.2. Smoke test

Chạy 1 epoch trên 100 mẫu:

```powershell
python -m src.train_textcnn --task both --smoke-test --reports-dir .tmp\smoke_reports --models-dir .tmp\smoke_models --runs-dir .tmp\smoke_runs
```

Smoke test phải:

- Load được data.
- Forward/backward được.
- Tính loss được.
- Lưu checkpoint được.
- Load lại checkpoint được.
- Predict được một câu.

### 9.3. Trước khi commit

```powershell
python -m compileall src
pytest -q
```

Nếu test fail, không commit.

---

## 10. Git

### 10.1. Branch

```text
feat/...
fix/...
exp/...
docs/...
```

Ví dụ:

```text
feat/textcnn-multitask
exp/lambda-sweep
fix/label-mapping
```

### 10.2. Commit message

Theo Conventional Commits:

```text
feat: add multi-task textcnn
fix: correct label mapping
exp: add lambda sweep results
docs: update experiment protocol
test: add model shape tests
```

### 10.3. Không làm

- Không commit trực tiếp lên `main` nếu repo có branch protection.
- Không force push.
- Không commit secrets.
- Không commit dữ liệu thô.
- Không commit model lớn.

### 10.4. Trước khi merge

- Test pass.
- Không có conflict.
- Có mô tả thay đổi.
- Có kết quả hoặc lý do nếu là thay đổi thí nghiệm.
- Không làm giảm chất lượng code hiện có.

---

## 11. Báo cáo và tài liệu

### 11.1. Ngôn ngữ

- Tài liệu, báo cáo, slide: tiếng Việt.
- Tên biến, hàm, class: tiếng Anh.
- Comment: tiếng Anh hoặc tiếng Việt, nhưng phải nhất quán trong một file.

### 11.2. Báo cáo kết quả

Mỗi bảng kết quả phải có:

- Tên mô hình.
- Split.
- Metric.
- Seed hoặc mean ± std.
- Thời gian chạy nếu liên quan.
- Ghi chú cấu hình đặc biệt.

### 11.3. Hình ảnh

- Confusion matrix.
- Learning curve.
- So sánh single-task vs multi-task.
- Ảnh demo.

---

## 12. Demo

- Demo load model đã train, không train lại.
- Preprocess lúc inference giống hệt lúc train.
- Dùng `model.eval()` và `torch.no_grad()`.
- Cache model bằng `st.cache_resource`.
- Xử lý câu rỗng.
- Cảnh báo khi confidence thấp.
- Không log dữ liệu thật của sinh viên.
- Có ví dụ mẫu và video dự phòng.

---

## 13. Bảo mật và đạo đức

- Không commit API key.
- Không đưa dữ liệu sinh viên thật lên demo công khai.
- Không dùng kết quả để đánh giá cá nhân giảng viên.
- Không khẳng định mô hình thay thế con người.
- Ghi rõ hạn chế với mỉa mai, teencode, câu nhiều chủ đề.
- Tuân thủ giấy phép dataset.

---

## 14. Lệnh chuẩn

Các lệnh dưới đây là quy ước mục tiêu. Nếu repo chưa có script tương ứng, agent phải tạo hoặc ghi rõ là chưa có.

### Cài đặt

Yêu cầu **Python 3.12.x**. Môi trường hiện tại dùng Python 3.12.15 cục bộ trong `.python/`; kích hoạt bằng `.\scripts\activate.ps1`. Không chuyển sang 3.15 khi chưa kiểm tra lại toàn bộ stack.

```powershell
# Nếu máy đã có Python 3.12:
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

Hoặc dùng script tiện ích:

```powershell
.\scripts\activate.ps1
```

### Kiểm tra

```powershell
python -m compileall src
pytest -q
```

### Train

```powershell
python -m src.train_textcnn --task both
python -m src.train_textcnn --task sentiment
python -m src.train_textcnn --task topic
python -m src.train_textcnn_multitask
python -m src.train_textcnn_multitask --lambda-topic 1.0
python -m src.train_phobert_multitask
python -m src.train_phobert_keyword_multitask
```

### Smoke test

```powershell
python -m src.train_textcnn --task both --smoke-test --reports-dir .tmp\smoke_reports --models-dir .tmp\smoke_models --runs-dir .tmp\smoke_runs
python -m src.train_textcnn_multitask --smoke-test --reports-dir .tmp\mt_smoke_reports --models-dir .tmp\mt_smoke_models --runs-dir .tmp\mt_smoke_runs
python -m src.train_phobert_multitask --smoke-test --reports-dir .tmp\phobert_smoke_reports --models-dir .tmp\phobert_smoke_models --runs-dir .tmp\phobert_smoke_runs
```

### Evaluate

Chưa có script evaluate riêng; các script train tự đánh giá dev/test và ghi báo cáo.

### Predict

```powershell
python -m src.predict --text "Cơ sở vật chất còn yếu"
```

### Demo

```powershell
.\scripts\run_demo.ps1
```

Script này đặt `USERPROFILE` và cache Streamlit vào `.demo-home/` để tránh lỗi quyền với `C:\Users\<user>\.streamlit`.

---

## 15. Definition of Done

Một task chỉ được coi là xong khi:

- [ ] Code chạy được từ môi trường sạch.
- [ ] Không có lỗi compile.
- [ ] Test liên quan pass.
- [ ] Có smoke test nếu sửa model/data.
- [ ] Không rò rỉ dữ liệu.
- [ ] Không hardcode label/path.
- [ ] Config và seed được lưu.
- [ ] Kết quả được ghi vào `reports/results.csv` nếu là thí nghiệm.
- [ ] README/AGENTS được cập nhật nếu cách chạy thay đổi.
- [ ] Không commit dữ liệu/model lớn/secrets.
- [ ] Báo cáo cuối có tóm tắt, file thay đổi, lệnh đã chạy, kết quả và rủi ro.

---

## 16. Những việc không được làm

- Không bịa metric hoặc kết quả.
- Không tune trên test.
- Không sửa `data/raw/`.
- Không xóa dữ liệu/model khi chưa xác nhận.
- Không force push.
- Không commit `.venv`, data, model, secrets.
- Không thêm CUDA-only code.
- Không fine-tune PhoBERT toàn bộ trên CPU mà không hỏi.
- Không thêm dependency nặng nếu không cần.
- Không refactor ngoài phạm vi task.
- Không tuyên bố “xong” khi chưa chạy kiểm tra.

---

## 17. Trạng thái hiện tại

- [x] Đã có proposal.
- [x] Đã scaffold thư mục cơ bản.
- [x] Đã tạo `.venv` với Python 3.12.15.
- [x] Đã cài PyTorch CPU, Transformers, scikit-learn, Streamlit.
- [x] Đã có `requirements.txt`, `requirements-lock.txt`, `.gitignore`, `README.md`.
- [x] Đã kiểm tra import, TextCNN forward pass, PhoBERT tokenizer/model.
- [x] Đã có test môi trường và `pytest` pass.
- [x] Đã tải UIT-VSFC và có script tải lại.
- [x] Đã có `src/data/load.py` và `configs/label_map.json`.
- [x] Đã xác minh label mapping và data audit.
- [x] Đã có EDA, hình và báo cáo `reports/eda_summary.md`.
- [x] Đã có baseline TF-IDF + Logistic Regression và `reports/baseline_report.md`.
- [x] Đã có TextCNN single-task và `reports/textcnn_single_report.md`.
- [x] Đã có TextCNN multi-task và `reports/textcnn_multitask_report.md`.
- [x] Đã có PhoBERT frozen multi-task và `reports/phobert_multitask_report.md`.
- [x] Đã có PhoBERT keyword-aware multi-task và `reports/phobert_keyword_multitask_report.md`.
- [x] Đã có demo Streamlit và `src.predict`.
- [ ] Chưa có báo cáo.

### Thứ tự ưu tiên tiếp theo

1. Chạy thí nghiệm 3 seed.
2. Viết báo cáo.

---

## 18. Ghi chú cập nhật

Khi thay đổi quy trình, cập nhật file này và ghi rõ:

- Ngày thay đổi.
- Lý do.
- Phần nào bị ảnh hưởng.
- Người/agent thực hiện.
