import argparse
import os

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)
from transformers import TrainingArguments, Trainer
from nlpaug.augmenter.word import SynonymAug  # 用于数据增强
import nltk
nltk.download('averaged_perceptron_tagger_eng')


def main(args):
    dataset = load_dataset("SetFit/wnli")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # 数据增强
    augmenter = SynonymAug(aug_src='wordnet')

    def preprocess_function(examples):
        # 对输入文本进行增强
        text1_aug = augmenter.augment(examples["text1"])
        text2_aug = augmenter.augment(examples["text2"])
        
        return {
            # **tokenizer(examples["text1"], examples["text2"]),
            **tokenizer(text1_aug, text2_aug),
            "label": examples["label"],
        }

    dataset = dataset.map(preprocess_function, batched=True)
    print(tokenizer.decode(dataset["train"]["input_ids"][0]))

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        trust_remote_code=True,
    )
    # 使用自定义模型
    # model = CustomModelWithAttention(model_name=args.model, num_labels=2)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")

    metric = evaluate.load("accuracy")

    def compute_metrics(p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=-1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        return result

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        args=TrainingArguments(
            output_dir=args.output,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=10,
            learning_rate=args.lr,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            gradient_accumulation_steps=args.accum,
            num_train_epochs=args.epoch,
            warmup_steps=100,  # 使用学习率预热
            weight_decay=0.01,  # L2正则化
            eval_steps=50,
            load_best_model_at_end=True,
        ),
    )
    trainer.train()
    
    # save_model(model, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, default="prajjwal1/bert-tiny")
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--accum", type=int, default=2)
    parser.add_argument("--output", type=str, default="output")
    args = parser.parse_args()
    main(args)
