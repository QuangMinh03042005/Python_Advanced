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

25 slide, 16:9:

1. Bìa có khung
2. Phân công thuyết trình
3. Luồng bàn giao giữa 3 người
4. **Phần 1 — Bài toán & sản phẩm**
5. Bài toán và động lực
6. Mục tiêu và đóng góp
7. Pipeline tổng quan
8. Demo và ý nghĩa thực tế
9. **Phần 2 — Dữ liệu, mô hình & độ tin cậy**
10. UIT-VSFC
11. EDA
12. Multi-task learning
13. TextCNN
14. PhoBERT + keyword-aware
15. Calibration
16. Robustness
17. **Phần 3 — Thực nghiệm, kết quả & kết luận**
18. Thiết lập thực nghiệm
19. Kết quả seed 42
20. Kết quả 3 seed
21. Kiểm định thống kê
22. Distillation
23. Hạn chế
24. Kết luận và hướng phát triển
25. Q&A
