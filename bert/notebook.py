#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 13:34:26 2021

@author: qwang
"""

import numpy as np
from datasets import load_dataset, load_metric

from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

batch_size = 16

#%%
datasets = load_dataset("conll2003")
datasets
datasets["train"]
datasets["train"][0]

datasets["train"].features["ner_tags"]
tag_list = datasets["train"].features["ner_tags"].feature.names


#%%
def tokenize_and_align_tags(samples, label_all_tokens=True):
    tokenized_inputs = tokenizer(samples["tokens"], truncation=True, is_split_into_words=True)

    tags = []
    for i, tag in enumerate(samples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        tag_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically ignored in the loss function.
            if word_idx is None:
                tag_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                tag_ids.append(tag[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                tag_ids.append(tag[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        tags.append(tag_ids)

    tokenized_inputs["labels"] = tags
    return tokenized_inputs


print(tokenize_and_align_tags(datasets['train'][:3]))
tokenized_data = datasets.map(tokenize_and_align_tags, batched=True)

#%%
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(tag_list))
args = TrainingArguments(
    output_dir = 'exp',
    evaluation_strategy = "epoch",
    learning_rate = 2e-5,
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size = batch_size,
    num_train_epochs = 3,
    weight_decay = 0.01,
)

#%%
from transformers import DataCollatorForTokenClassification
data_collator = DataCollatorForTokenClassification(tokenizer)

#%%
metric = load_metric("seqeval")

def compute_metrics(p):
    batch_preds, batch_trues = p
    batch_preds = np.argmax(batch_preds, axis=2)

    preds_cut = [
        [tag_list[p] for (p, t) in zip(preds, trues) if t != -100]
        for preds, trues in zip(batch_preds, batch_trues)
    ]
    
    trues_cut = [
        [tag_list[t] for (p, t) in zip(preds, trues) if t != -100]
        for preds, trues in zip(batch_preds, batch_trues)
    ]

    results = metric.compute(predictions=preds_cut, references=trues_cut)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

#%%
trainer = Trainer(
    model,
    args,
    train_dataset = tokenized_data["train"].select(list(range(1000))),
    eval_dataset = tokenized_data["validation"].select(list(range(100))),
    data_collator = data_collator,
    tokenizer = tokenizer,
    compute_metrics = compute_metrics
)

trainer.train()
trainer.evaluate()


#%%
batch_preds, batch_trues, _ = trainer.predict(tokenized_data["validation"])
batch_preds = np.argmax(batch_preds, axis=2)
# Remove ignored index (special tokens)
preds_cut = [
    [tag_list[p] for (p, t) in zip(preds, trues) if t != -100]
    for preds, trues in zip(batch_preds, batch_trues)
]
trues_cut = [
    [tag_list[t] for (p, t) in zip(preds, trues) if t != -100]
    for preds, trues in zip(batch_preds, batch_trues)
]

results = metric.compute(predictions = preds_cut, references = trues_cut)
results
















































