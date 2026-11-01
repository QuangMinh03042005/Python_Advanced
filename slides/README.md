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

21 slide, 16:9:

1. Bìa
2. Phân công thuyết trình
3. Bài toán và động lực
4. Mục tiêu và đóng góp
5. Pipeline tổng quan
6. UIT-VSFC
7. EDA
8. Multi-task learning
9. TextCNN
10. PhoBERT + keyword-aware
11. Thiết lập thực nghiệm
12. Kết quả seed 42
13. Kết quả 3 seed
14. Kiểm định thống kê
15. Distillation
16. Calibration
17. Robustness
18. Demo và ý nghĩa thực tế
19. Hạn chế
20. Kết luận và hướng phát triển
21. Q&A
