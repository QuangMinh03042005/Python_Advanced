# TextCNN multi-task — lambda sweep

Mỗi run dùng cùng seed 42, cùng split, cùng kiến trúc; chỉ đổi `lambda_topic` trong loss:

```text
L = L_sentiment + lambda_topic * L_topic
```

| Lambda | Best epoch | Dev mean-F1 | Dev joint | Test sentiment F1 | Test topic F1 | Test joint |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 7 | 0.7693 | 0.7726 | 0.7639 | 0.7175 | 0.7388 |
| 1.0 | 10 | **0.7726** | **0.7859** | 0.7556 | 0.7390 | 0.7552 |
| 2.0 | 12 | 0.7724 | 0.7814 | 0.7598 | 0.7496 | 0.7641 |

## Chọn lambda

- Chọn theo **dev mean-F1**, không theo test.
- `lambda = 1.0` có dev mean-F1 cao nhất (0.7726), joint dev cao nhất (0.7859).
- `lambda = 2.0` có test tốt hơn nhưng không được chọn, để tránh tune trên test.
- `lambda = 0.5` nghiêng về sentiment, làm topic giảm mạnh.

## So sánh với single-task

| Model | Test sentiment F1 | Test topic F1 | Test joint |
|---|---:|---:|---:|
| TF-IDF baseline | 0.7400 | 0.7247 | 0.7227 |
| TextCNN single-task | **0.7621** | 0.7355 | 0.7536 |
| TextCNN multi-task (λ=1.0) | 0.7556 | **0.7390** | **0.7552** |

Kết luận: multi-task cải thiện nhẹ topic và joint, nhưng giảm nhẹ sentiment. Đây là trade-off, chưa phải cải thiện rõ rệt trên mọi tác vụ.
