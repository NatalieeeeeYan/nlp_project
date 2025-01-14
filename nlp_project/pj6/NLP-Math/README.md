# README

**对模型进行 sft 微调：**

```sh
python sft.py --model <Path to your model> --dataset <Path to your dataset> --output <Path to save trained model>
```

**对模型进行 lora 微调：**

```sh
python lora.py --model <Path to your model> --dataset <Path to your dataset> --output <Path to save trained model>
```

**对模型进行 prefix tuning 微调：**

```sh
python prefix_tune.py --model <Path to your model> --dataset <Path to your dataset> --output <Path to save trained model>
```

**对模型进行测试：**

```sh
python test.py --model <Path to your model> --result <Path to save testing result>
```

**对模型进行评估：**

```sh
python evaluation_*.py --result <Path to the testing result> --gt <Path to the ground truth>
# --gt is optional
```

- 由于 gsm8k 和 MATH 数据集的表现方式差异较大，分别针对二者写了脚本进行评估：

  - `evaluation_gsm8k.py`：针对数据集 gsm8k 的评估函数；

  - `evaluation_math.py`：针对数据集 MATH 的评估函数，基础逻辑是字符串处理和子串匹配，根据评估结果编写了一些特定规则，预估会产生较多FP，后续结合 GPT 一同进行判断和评估。

- 通过 GPT 的 API 进行结果评估：

  ```sh
  # 运行前请修改 FIXME 处：
  # 	line 57 的待评估的模型输出结果路径
  # 	line 5 的 OpenAI API 密钥
  nohup python gpt_eval.py > <path_to_log> 2>&1 &
  # 评估结束后打开log，搜索 Consistent: True
  # 搜索结果除以测试数据总数为正确率
  ```

测试 Qwen2.5-Math 模型：运行 `qwen2.5-math.ipynb` 文件



`./results/`：存放各类方法训练出的模型在 gsm8k 和 MATH 这两个数据集上的输出结果

- 结果文件格式为：

  ```json
  [
      {
          "question": str, 				// 问题
          "ground_truth": str, 		// 标准答案
          "prediction": str				// 模型输出
      }
  ]
  ```

`dataset/`：存放训练和测试数据。

`tools.py`：存放工具函数。
