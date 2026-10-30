# Data audit — UIT-VSFC

Ngày kiểm tra: 2026-10-30

## Nguồn

- Dataset: UIT-VSFC — Vietnamese Students' Feedback Corpus.
- Paper: Nguyen et al., KSE 2018.
- Dữ liệu gốc: tải từ Google Drive theo các ID trong `scripts/download_uit_vsfc.py`.
- Giấy phép: chưa xác định rõ trong bản phân phối; chỉ dùng cho học tập/nghiên cứu, không công bố lại dữ liệu thô.

## Quy mô

| Split | Số câu |
|---|---:|
| train | 11.426 |
| dev | 1.583 |
| test | 3.166 |
| **Tổng** | **16.175** |

## Label mapping

### Sentiment

| ID | Nhãn |
|---:|---|
| 0 | negative |
| 1 | neutral |
| 2 | positive |

### Topic

| ID | Nhãn |
|---:|---|
| 0 | lecturer |
| 1 | training_program |
| 2 | facility |
| 3 | others |

Mapping được xác minh bằng cách in ví dụ từng lớp. Lưu ý: thứ tự `facility`/`others` khác với class order trong một số loader tham khảo.

## Phân bố nhãn

### Sentiment

| Split | negative | neutral | positive |
|---|---:|---:|---:|
| train | 5.325 | 458 | 5.643 |
| dev | 705 | 73 | 805 |
| test | 1.409 | 167 | 1.590 |

Nhận xét: lớp `neutral` rất ít so với hai lớp còn lại. Cần dùng macro-F1 và class weight.

### Topic

| Split | lecturer | training_program | facility | others |
|---|---:|---:|---:|---:|
| train | 8.166 | 2.201 | 497 | 562 |
| dev | 1.151 | 267 | 70 | 95 |
| test | 2.290 | 572 | 145 | 159 |

Nhận xét: `lecturer` chiếm đa số; `facility` và `others` ít hơn đáng kể.

## Kiểm tra chất lượng

- Không có câu rỗng.
- Không có câu trùng giữa train/dev/test.
- Trong train có **1 câu trùng lặp nội bộ**; chưa xử lý, sẽ ghi nhận khi làm sạch.
- Số dòng `sentences` = `sentiments` = `topics` ở cả 3 split.
- Encoding đọc bằng `utf-8-sig`; dữ liệu hiện tại là UTF-8.

## Cách tái lập

```powershell
.\scripts\activate.ps1
python scripts\download_uit_vsfc.py
python -m src.data.load
pytest
```
