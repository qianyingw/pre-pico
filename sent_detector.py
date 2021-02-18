#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:45:53 2021

@author: qwang
"""

import os
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup


os.chdir('/home/qwang/pre-pico')
NUM_EPOCHS = 10
SEED = 1234
PRE_WGTS = 'bert-base-uncased'
# PRE_WGTS = 'dmis-lab/biobert-v1.1'
# PRE_WGTS = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
# PRE_WGTS = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'

softmax = nn.Softmax(dim=1)

#%% Load data
dat = pd.read_csv("data/tsv/b1.csv", sep=',', engine="python")   
dat = dat.sample(frac=1, random_state=SEED)  # Shuffle
dat = dat.reset_index(drop=True)

sents = list(dat['sent'])
labels = list((dat['freq_ent'] > 0).astype(int))

# Split to train/valid/test
dlen = len(dat)
train_sents, train_labs = sents[: int(0.8*dlen)], labels[: int(0.8*dlen)] 
valid_sents, valid_labs = sents[int(0.8*dlen): int(0.9*dlen)], labels[int(0.8*dlen): int(0.9*dlen)] 
test_sents, test_labs = sents[int(0.9*dlen):], labels[int(0.9*dlen):] 


#%%
tokenizer = BertTokenizerFast.from_pretrained(PRE_WGTS)  

train_encs = tokenizer(train_sents, truncation=True, padding=True)
valid_encs = tokenizer(valid_sents, truncation=True, padding=True)
test_encs = tokenizer(test_sents, truncation=True, padding=True)

#%%
class PICOSentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = PICOSentDataset(train_encs, train_labs)
valid_dataset = PICOSentDataset(valid_encs, valid_labs)
test_dataset = PICOSentDataset(test_encs, test_labs)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

#%%
model = BertForSequenceClassification.from_pretrained(PRE_WGTS)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

# Slanted triangular Learning rate scheduler
total_steps = len(train_loader) * NUM_EPOCHS // 4
warm_steps = int(total_steps * 0.1)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_steps, num_training_steps=total_steps)

#%% Metric scores
def metrics_fn(preds, y, th=0.5):
    """ preds: torch tensor, [batch_size, output_dim]
        y: torch tensor, [batch_size]
    """   
    if torch.cuda.device_count() == 1:
        y_preds = (preds[:,1] > th).int().type(torch.LongTensor).cuda()
    else:
        y_preds = (preds[:,1] > th).int().type(torch.LongTensor)
    ones = torch.ones_like(y_preds)
    zeros = torch.zeros_like(y_preds)
    
    pos = torch.eq(y_preds, y).sum().item()
    tp = (torch.eq(y_preds, ones) & torch.eq(y, ones)).sum().item()
    tn = (torch.eq(y_preds, zeros) & torch.eq(y, zeros)).sum().item()
    fp = (torch.eq(y_preds, ones) & torch.eq(y, zeros)).sum().item()
    fn = (torch.eq(y_preds, zeros) & torch.eq(y, ones)).sum().item()
    
    assert pos == tp + tn
    acc = pos / y.shape[0]  # torch.FloatTensor([y.shape[0]])
    f1 = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn != 0) else 0
    rec = tp / (tp + fn) if (tp + fn != 0) else 0
    ppv = tp / (tp + fp) if (tp + fp != 0) else 0
    spc = tn / (tn + fp) if (tn + fp != 0) else 0
    
    return {'accuracy': acc, 'f1': f1, 'recall': rec, 'precision': ppv, 'specificity': spc}

#%%
def train_fn(model, data_loader, clip=0.1, accum_step=4, threshold=0.5):
    
    scores = {'loss': 0, 'accuracy': 0, 'f1': 0, 'recall': 0, 'precision': 0, 'specificity': 0}
    len_iter = len(data_loader)
    
    model.train()
    optimizer.zero_grad()
    
    with tqdm(total=len_iter) as progress_bar:      
        for j, batch in enumerate(data_loader):                      
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss, logits = outputs[0], outputs[1]
            preds = softmax(logits)      
            
            scores['loss'] += loss.item() 
            epoch_scores = metrics_fn(preds, labels, threshold)  # dictionary of 5 metric scores
            for key, value in epoch_scores.items():               
                scores[key] += value  
            
            loss = loss / accum_step  # average loss gradients bec they are accumulated by loss.backward()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            
            # Gradient accumulation    
            if (j+1) % accum_step == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()                       
            progress_bar.update(1)  
    
    for key, value in scores.items():
        scores[key] = value / len_iter       
    return scores
             

#%%
def valid_fn(model, data_loader, threshold=0.5):

    scores = {'loss': 0, 'accuracy': 0, 'f1': 0, 'recall': 0, 'precision': 0, 'specificity': 0}
    len_iter = len(data_loader)
    model.eval()

    with torch.no_grad():
        with tqdm(total=len_iter) as progress_bar:
            for batch in data_loader:
            
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss, logits = outputs[0], outputs[1]
                preds = softmax(logits)  
                
                scores['loss'] += loss.item() 
                epoch_scores = metrics_fn(preds, labels, threshold)  
                for key, value in epoch_scores.items():               
                    scores[key] += value       
                progress_bar.update(1)  # update progress bar   
                
    for key, value in scores.items():
        scores[key] = value / len_iter   
    return scores

#%%
for epoch in range(NUM_EPOCHS):   
    train_scores = train_fn(model, train_loader)
    valid_scores = valid_fn(model, valid_loader)       

    print("\n\nEpoch {}/{}...".format(epoch+1, NUM_EPOCHS))                       
    print('\n[Train] loss: {0:.3f} | acc: {1:.2f}% | f1: {2:.2f}% | rec: {3:.2f}% | prec: {4:.2f}% | spec: {5:.2f}%'.format(
            train_scores['loss'], train_scores['accuracy']*100, train_scores['f1']*100, train_scores['recall']*100, train_scores['precision']*100, train_scores['specificity']*100))
    print('[Valid] loss: {0:.3f} | acc: {1:.2f}% | f1: {2:.2f}% | rec: {3:.2f}% | prec: {4:.2f}% | spec: {5:.2f}%\n'.format(
        valid_scores['loss'], valid_scores['accuracy']*100, valid_scores['f1']*100, valid_scores['recall']*100, valid_scores['precision']*100, valid_scores['specificity']*100))