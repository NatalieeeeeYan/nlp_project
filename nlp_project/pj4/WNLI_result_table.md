| Model                     | 训练参数                                                     | checkpoint地址                              | 分数 |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------- | ---- |
| TA标准模型                | epoch=3, lr=1e-3                                             | output/v2_epoch3_lr1e-3/checkpoint-1270     | 56.5 |
| TA标准模型                | epoch=3, lr=1e-3                                             | output/v2_epoch3_lr1e-3/checkpoint-1905     | 56.5 |
| bert-base-uncased         | epoch=3, lr=1e-3                                             | output/v3_bert-base-uncased/checkpoint-1270 | 56.5 |
| bert-base-uncased         | epoch=3,bs=8,  lr=2e-5, warmup_steps=100, weight_decay=0.01, eval_steps=50 | output/v4/checkpoint-80                     | 56.5 |
| bert-base-uncased         | epoch=3,bs=128,  lr=2e-5, warmup_steps=100, weight_decay=0.01, eval_steps=50, load_best_model_at_end, 使用了数据增强 | output/v5/checkpoint-6                      | 56.5 |
| bert-base-uncased         | epoch=10,bs=128,  lr=1e-5, warmup_steps=100, weight_decay=0.01, eval_steps=50, load_best_model_at_end, 使用了数据增强 | output/v6/checkpoint-20                     | 53.2 |
| microsoft/deberta-v3-base | epoch=3, bs=128, lr=1e-5, 使用了数据增强                     | output/v7/checkpoint-6                      |      |
| bert-base-uncased         | epoch=10, bs=32, lr=1e-5, warmup_steps=100, weight_decay=0.01, eval_steps=50, load_best_model_at_end, data augment | output/v8/checkpoint-100                    | 56.5 |

提交版本：output/v8/checkpoint-100

结果：

<img src="/Users/songwenyan/FudanCourses/NLP-自然语言处理和知识表示/nlp_project/pj4/v8-100result.png" alt="v8-100result" style="zoom:50%;" />
