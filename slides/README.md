# Slide thuyết trình

- `main.tex`: nguồn Beamer.
- `main.pdf`: bản đã compile.
- `presentation_plan.md`: phân công và kịch bản 3 người.

## Compile

Yêu cầu XeLaTeX và font Arial/Times New Roman.

```powershell
cd slides
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```

Chạy hai lần để cập nhật tổng số slide và tham chiếu.

## Nội dung

23 slide, 16:9:

1. Bìa có khung
2. **Phần 1 — Bài toán & sản phẩm**
3. Bài toán và động lực
4. Mục tiêu và đóng góp
5. Pipeline tổng quan
6. Demo và ý nghĩa thực tế
7. **Phần 2 — Dữ liệu, mô hình & độ tin cậy**
8. UIT-VSFC
9. EDA
10. Multi-task learning
11. TextCNN
12. PhoBERT + keyword-aware
13. Calibration
14. Robustness
15. **Phần 3 — Thực nghiệm, kết quả & kết luận**
16. Thiết lập thực nghiệm
17. Kết quả seed 42
18. Kết quả 3 seed
19. Kiểm định thống kê
20. Distillation
21. Hạn chế
22. Kết luận và hướng phát triển
23. Q&A
