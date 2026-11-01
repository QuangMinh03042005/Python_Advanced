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

22 slide, 16:9:

1. Bìa
2. Phân công thuyết trình
3. Luồng bàn giao giữa 3 người
4. Bài toán và động lực
5. Mục tiêu và đóng góp
6. Pipeline tổng quan
7. UIT-VSFC
8. EDA
9. Multi-task learning
10. TextCNN
11. PhoBERT + keyword-aware
12. Thiết lập thực nghiệm
13. Kết quả seed 42
14. Kết quả 3 seed
15. Kiểm định thống kê
16. Distillation
17. Calibration
18. Robustness
19. Demo và ý nghĩa thực tế
20. Hạn chế
21. Kết luận và hướng phát triển
22. Q&A
