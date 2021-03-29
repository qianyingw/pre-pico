#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:45:53 2021

@author: qwang
"""

import os
import random
os.chdir('/home/qwang/pre-pico')
from tqdm import tqdm
import pandas as pd
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

import utils
metrics_fn = utils.metrics_fn

#%%
NUM_EPOCHS = 10
SEED = 1234
EXP_DIR = '/home/qwang/pre-pico/exp/sent/base'
PRE_WGTS = 'bert-base-uncased'
# PRE_WGTS = 'dmis-lab/biobert-v1.1'
# PRE_WGTS = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
# PRE_WGTS = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'

softmax = nn.Softmax(dim=1)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True     
torch.backends.cudnn.benchmark = False   # This makes things slower  

#%% Load data
dat = pd.read_csv("data/tsv/18mar_output/pico_18mar.csv", sep=',', engine="python")   
dat = dat.sample(frac=1, random_state=SEED)  # Shuffle
dat = dat.reset_index(drop=True)

sents = list(dat['sent'])
labels = list((dat['freq_ent'] > 0).astype(int))

# Split to train/valid/test
dlen = len(dat)
train_sents, train_labs = sents[: int(0.8*dlen)], labels[: int(0.8*dlen)] 
valid_sents, valid_labs = sents[int(0.8*dlen): int(0.9*dlen)], labels[int(0.8*dlen): int(0.9*dlen)] 
test_sents, test_labs = sents[int(0.9*dlen):], labels[int(0.9*dlen):] 

#%% Tokenization 
tokenizer = BertTokenizerFast.from_pretrained(PRE_WGTS)  
train_encs = tokenizer(train_sents, truncation=True, padding=True)
valid_encs = tokenizer(valid_sents, truncation=True, padding=True)
test_encs = tokenizer(test_sents, truncation=True, padding=True)

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
    
# Dataset and Loader
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
if os.path.exists(EXP_DIR) == False:
    os.makedirs(EXP_DIR)   
min_valid_loss = float('inf')
output_dict = {'prfs': {}}

for epoch in range(NUM_EPOCHS):   
    train_scores = train_fn(model, train_loader)
    valid_scores = valid_fn(model, valid_loader)       
    
    # Update output dictionary
    output_dict['prfs'][str('train_'+str(epoch+1))] = train_scores
    output_dict['prfs'][str('valid_'+str(epoch+1))] = valid_scores
    
    # Save scores 
    is_best = (valid_scores['loss'] < min_valid_loss)
    if is_best == True:   
        min_valid_loss = valid_scores['loss']
    
    # Save model
    utils.save_checkpoint({'epoch': epoch+1,
                           'state_dict': model.state_dict(),
                           'optim_Dict': optimizer.state_dict()},
                           is_best = is_best, checkdir = EXP_DIR)

    print("\n\nEpoch {}/{}...".format(epoch+1, NUM_EPOCHS))                       
    print('\n[Train] loss: {0:.3f} | acc: {1:.2f} | f1: {2:.2f} | rec: {3:.2f} | prec: {4:.2f} | spec: {5:.2f}'.format(
            train_scores['loss'], train_scores['accuracy']*100, train_scores['f1']*100, train_scores['recall']*100, train_scores['precision']*100, train_scores['specificity']*100))
    print('[Valid] loss: {0:.3f} | acc: {1:.2f} | f1: {2:.2f} | rec: {3:.2f} | prec: {4:.2f} | spec: {5:.2f}\n'.format(
        valid_scores['loss'], valid_scores['accuracy']*100, valid_scores['f1']*100, valid_scores['recall']*100, valid_scores['precision']*100, valid_scores['specificity']*100))
    

prfs_name = os.path.basename(EXP_DIR)+'_prfs.json'
prfs_path = os.path.join(EXP_DIR, prfs_name)
with open(prfs_path, 'w') as fout:
    json.dump(output_dict, fout, indent=4)