# Statistical tests

- Test set: 3.166 câu.
- So sánh trên predictions của 3 seed: 42, 123, 2024.
- McNemar exact test cho correctness từng câu.
- Bootstrap percentile 95% CI với 2000 lần resample.
- p-value chưa hiệu chỉnh đa so sánh; Bonferroni alpha cho 18 test là 0.0028.

## McNemar

| Seed | Comparison | Task | b | c | p-value |
|---:|---|---|---:|---:|---:|
| 42 | keyword_vs_frozen | sentiment | 76 | 44 | 0.004455 |
| 42 | keyword_vs_frozen | topic | 131 | 72 | 4.174e-05 |
| 42 | keyword_vs_frozen | joint | 157 | 94 | 8.409e-05 |
| 42 | keyword_vs_textcnn | sentiment | 191 | 104 | 4.601e-07 |
| 42 | keyword_vs_textcnn | topic | 244 | 141 | 1.709e-07 |
| 42 | keyword_vs_textcnn | joint | 329 | 170 | 9.808e-13 |
| 123 | keyword_vs_frozen | sentiment | 105 | 57 | 0.0002013 |
| 123 | keyword_vs_frozen | topic | 133 | 61 | 2.536e-07 |
| 123 | keyword_vs_frozen | joint | 183 | 94 | 9.717e-08 |
| 123 | keyword_vs_textcnn | sentiment | 197 | 121 | 2.394e-05 |
| 123 | keyword_vs_textcnn | topic | 188 | 151 | 0.05039 |
| 123 | keyword_vs_textcnn | joint | 278 | 199 | 0.0003452 |
| 2024 | keyword_vs_frozen | sentiment | 145 | 47 | 7.966e-13 |
| 2024 | keyword_vs_frozen | topic | 218 | 70 | 7.462e-19 |
| 2024 | keyword_vs_frozen | joint | 285 | 79 | 1.853e-28 |
| 2024 | keyword_vs_textcnn | sentiment | 187 | 105 | 1.842e-06 |
| 2024 | keyword_vs_textcnn | topic | 164 | 146 | 0.3343 |
| 2024 | keyword_vs_textcnn | joint | 255 | 175 | 0.0001335 |

## Bootstrap difference

| Seed | Comparison | Task | Difference | 95% CI | CI excludes 0 |
|---:|---|---|---:|---|---|
| 42 | keyword_vs_frozen | sentiment | 0.0003 | [-0.0174, 0.0183] | no |
| 42 | keyword_vs_frozen | topic | 0.0205 | [0.0051, 0.0369] | yes |
| 42 | keyword_vs_frozen | joint | 0.0199 | [0.0101, 0.0300] | yes |
| 42 | keyword_vs_textcnn | sentiment | 0.0232 | [-0.0040, 0.0504] | no |
| 42 | keyword_vs_textcnn | topic | 0.0243 | [0.0027, 0.0462] | yes |
| 42 | keyword_vs_textcnn | joint | 0.0502 | [0.0366, 0.0638] | yes |
| 123 | keyword_vs_frozen | sentiment | 0.0026 | [-0.0202, 0.0249] | no |
| 123 | keyword_vs_frozen | topic | 0.0195 | [0.0047, 0.0346] | yes |
| 123 | keyword_vs_frozen | joint | 0.0281 | [0.0180, 0.0382] | yes |
| 123 | keyword_vs_textcnn | sentiment | 0.0154 | [-0.0143, 0.0434] | no |
| 123 | keyword_vs_textcnn | topic | 0.0063 | [-0.0157, 0.0281] | no |
| 123 | keyword_vs_textcnn | joint | 0.0250 | [0.0123, 0.0382] | yes |
| 2024 | keyword_vs_frozen | sentiment | 0.0216 | [0.0010, 0.0416] | yes |
| 2024 | keyword_vs_frozen | topic | 0.0225 | [0.0058, 0.0383] | yes |
| 2024 | keyword_vs_frozen | joint | 0.0651 | [0.0534, 0.0761] | yes |
| 2024 | keyword_vs_textcnn | sentiment | 0.0217 | [-0.0046, 0.0476] | no |
| 2024 | keyword_vs_textcnn | topic | -0.0040 | [-0.0267, 0.0192] | no |
| 2024 | keyword_vs_textcnn | joint | 0.0253 | [0.0123, 0.0376] | yes |

## Tổng hợp độ nhất quán

| Comparison | Task | McNemar p<0.05 / 3 seed | CI excludes 0 / 3 seed |
|---|---|---:|---:|
| keyword_vs_frozen | joint | 3/3 | 3/3 |
| keyword_vs_frozen | sentiment | 3/3 | 1/3 |
| keyword_vs_frozen | topic | 3/3 | 3/3 |
| keyword_vs_textcnn | joint | 3/3 | 3/3 |
| keyword_vs_textcnn | sentiment | 3/3 | 0/3 |
| keyword_vs_textcnn | topic | 1/3 | 1/3 |

## Cách đọc

- `b`: keyword đúng, candidate sai.
- `c`: keyword sai, candidate đúng.
- McNemar p nhỏ ⇒ tỷ lệ hai loại lỗi khác nhau có ý nghĩa.
- CI không chứa 0 ⇒ khác biệt có ý nghĩa ở mức 95%.
- Ưu tiên kết luận nhất quán trên cả 3 seed, không chỉ một seed.
