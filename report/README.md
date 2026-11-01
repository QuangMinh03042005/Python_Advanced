# Báo cáo LaTeX

- `main.tex`: nguồn báo cáo.
- `main.pdf`: bản đã compile.
- Logo dùng từ `../reports/SGU-LOGO.png`.

## Compile

Yêu cầu XeLaTeX (MiKTeX/TeX Live) và font Times New Roman.

```powershell
cd report
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```

Chạy hai lần để cập nhật mục lục, danh mục hình/bảng và tham chiếu.

## Ghi chú

- Báo cáo dùng XeLaTeX + `fontspec` để hiển thị tiếng Việt.
- Các hình được tham chiếu trực tiếp từ `../reports/`.
- Không commit file aux/log; đã cấu hình trong `.gitignore`.
