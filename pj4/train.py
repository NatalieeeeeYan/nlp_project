import argparse

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


def main(args):
    dataset = load_dataset("SetFit/wnli")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    def preprocess_function(examples):
        return {
            **tokenizer(examples["text1"], examples["text2"]),
            "label": examples["label"],
        }

    dataset = dataset.map(preprocess_function, batched=True)
    print(tokenizer.decode(dataset["train"]["input_ids"][0]))

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        trust_remote_code=True,
    )
    print(sum(p.numel() for p in model.parameters()))

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
            logging_steps=1,
            learning_rate=args.lr,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            gradient_accumulation_steps=args.accum,
            num_train_epochs=args.epoch,
        ),
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="prajjwal1/bert-tiny")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--output", type=str, default="output")
    args = parser.parse_args()
    main(args)
