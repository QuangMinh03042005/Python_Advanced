# Teacher soft labels

- Checkpoint: `D:\workspace\python\Python_Advanced\models\phobert_keyword_multitask\phobert_keyword_multitask_best.pt`
- Model type: `keyword_aware`
- Device: `cpu`
- Max length: `128`
- Batch size: `32`
- Extraction time: `514.6s`
- Git commit: `bef47f1`

| Split | Rows | Sent acc | Sent F1 | Topic acc | Topic F1 |
|---|---:|---:|---:|---:|---:|
| train | 11426 | 0.9626 | 0.9204 | 0.9088 | 0.8557 |
| dev | 1583 | 0.9248 | 0.7899 | 0.8781 | 0.7697 |
| test | 3166 | 0.9141 | 0.7788 | 0.8696 | 0.7633 |

## Files

```text
train_sentiment_logits.npy
train_topic_logits.npy
train_sentiment_probs.npy
train_topic_probs.npy
dev_*
test_*
metadata.json
```
