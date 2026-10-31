# Data

## UIT-VSFC

Dữ liệu gốc được tải vào:

```text
data/raw/uit_vsfc/
├── train/
│   ├── sentences.txt
│   ├── sentiments.txt
│   └── topics.txt
├── dev/
└── test/
```

Tải lại:

```powershell
.\scripts\activate.ps1
python scripts\download_uit_vsfc.py
```

Kiểm tra file:

```powershell
python scripts\download_uit_vsfc.py --check
```

## Quy ước

- `data/raw/` là bất biến — không sửa tay.
- `data/processed/` dành cho dữ liệu do script tạo ra.
- Không commit dữ liệu thô hoặc dữ liệu đã xử lý.
- Label mapping nằm trong `configs/label_map.json`.
- Kiểm tra tổng quan nằm trong `reports/data/data_summary.md`.

## Giấy phép

Giấy phép của UIT-VSFC chưa được xác định rõ trong bản phân phối đang dùng. Chỉ sử dụng cho mục đích học tập/nghiên cứu; không công bố lại dữ liệu thô.
