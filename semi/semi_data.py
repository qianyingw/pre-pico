#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 09:52:27 2021

@author: qwang
"""

import os
os.chdir('/home/qwang/pre-pico')
import json
import pandas as pd
import torch
import torch.nn as nn
import transformers
from transformers import BertTokenizerFast, BertForTokenClassification

from bert.bert_model import BERT_CRF, BERT_LSTM_CRF
from predict import sent_detect

softmax = nn.Softmax(dim=1)

#%% PICO model args
def load_model(mod, pre_wgts, pth_path, n_tags=13):
    ## Tokenization
    pre_wgts = 'bert-base-uncased' if pre_wgts == 'base' else pre_wgts
    pre_wgts = 'dmis-lab/biobert-v1.1' if pre_wgts == 'biobert' else pre_wgts
    pre_wgts = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract' if pre_wgts == 'pubmed-abs' else pre_wgts
    pre_wgts = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext' if pre_wgts == 'pubmed-full' else pre_wgts
    tokenizer = BertTokenizerFast.from_pretrained(pre_wgts, num_labels=n_tags) 
    
    ## Load model
    if mod == 'bert':
        model = BertForTokenClassification.from_pretrained(pre_wgts, num_labels=n_tags) 
    if mod == 'bert_crf':
        model = BERT_CRF.from_pretrained(pre_wgts, num_labels=n_tags)
    if mod == 'bert_lstm_crf':
        model = BERT_LSTM_CRF.from_pretrained(pre_wgts, num_labels=n_tags)
    # Load checkpoint
    checkpoint = torch.load(pth_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.cpu()
    model.eval()
    return tokenizer, model

#%% Calculate score across tokens
def pico_score(text, tokenizer, model): 
    # Tokenization
    inputs = tokenizer([text], is_split_into_words=True, return_offsets_mapping=True, padding=False, truncation=True, max_length=512)
    inputs = {key: torch.tensor(value) for key, value in inputs.items()} 
    
    ## Run model
    if isinstance(model, transformers.models.bert.modeling_bert.BertForTokenClassification):
        outputs = model(inputs['input_ids'].unsqueeze(0), inputs['attention_mask'].unsqueeze(0)) 
        logits = outputs[0].squeeze(0)[1:-1]  # [seq_len-2, n_tags], remove cls/sep token
    else:
        outs = model(inputs['input_ids'].unsqueeze(0), inputs['attention_mask'].unsqueeze(0), return_probs=True) 
        logits = outs[1].squeeze(0)  # [seq_len-2, n_tags]
    
    probs = softmax(logits)  # [seq_len-2, n_tags]
    probs_max = torch.max(probs, dim=1).values  # [seq_len-2], max probs of each record across tags
    score = torch.mean(probs_max).detach().numpy()  # average probs_max across all tokens 
    return score

#%% Generate PICO tags for tokens
def pico_tag(text, tokenizer, model, idx2tag): 
    # Tokenization
    inputs = tokenizer([text], is_split_into_words=True, return_offsets_mapping=True, padding=False, truncation=True, max_length=512)
    inputs = {key: torch.tensor(value) for key, value in inputs.items()} 
    
    ## Run model
    if isinstance(model, transformers.models.bert.modeling_bert.BertForTokenClassification):
        outputs = model(inputs['input_ids'].unsqueeze(0), inputs['attention_mask'].unsqueeze(0)) 
        logits = outputs[0].squeeze(0)[1:-1]  # [seq_len-2, n_tags], remove cls/sep token
    else:
        outs = model(inputs['input_ids'].unsqueeze(0), inputs['attention_mask'].unsqueeze(0), return_probs=True) 
        logits = outs[1].squeeze(0)  # [seq_len-2, n_tags]
    
    preds = torch.argmax(logits, dim=1)  # [seq_len-2]   
    preds = preds.numpy().tolist()  # len = seq_len-2
    tags = [idx2tag[str(idx)] for idx in preds]
    
    # Convert input_ids to tokens (contain ## pieces), then to original words
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][1:-1])
    tokens_pre, tags_pre = [], []
    for i, tok in enumerate(tokens):
        if tok.startswith("##"):
            if tokens_pre:
                tokens_pre[-1] = f"{tokens_pre[-1]}{tok[2:]}"
        else:
            tokens_pre.append(tok)
            tags_pre.append(tags[i])
    
    return tokens_pre, tags_pre

#%% Extract PICO text
df = pd.DataFrame(columns=['pmid', 'path', 'text'])
sent_pth = '/media/mynewdrive/pico/exp/sent/abs/best.pth.tar'
in_dir = '/media/mynewdrive/pico/pmc/tiab/b9.5k'  # b5, b6, b7, b8, b9, b95k (10,000 records)


for file in os.listdir(in_dir):   
    txt_path = os.path.join(in_dir, file)
    with open(txt_path, 'r', encoding='utf-8') as fin:
        text = fin.read()
    ## Extract pico text   
    text = sent_detect(text, sent_pth) 
    df = df.append({'pmid': file.split('-')[0], 'path': txt_path, 'text': text}, ignore_index=True)
    
df.to_csv('data/tsv/self_train_10k.csv', sep=',', index=False)

#%% Get score from two models
df = pd.read_csv("data/tsv/self_train.csv", sep=',', engine="python", encoding="utf-8")  
tokenizer1, model1 = load_model(mod='bert', pre_wgts='biobert', pth_path='/media/mynewdrive/pico/exp/bert/b0_bio/last.pth.tar')  # score1
tokenizer2, model2 = load_model(mod='bert_lstm_crf', pre_wgts='pubmed-full', pth_path='/media/mynewdrive/pico/exp/bert_lcrf/blc5_full/last.pth.tar')

for i, row in df.iterrows():
    if pd.isna(row['text']) == False:
        df.loc[i,'score1'] = pico_score(row['text'], tokenizer1, model1)
        df.loc[i,'score2'] = pico_score(row['text'], tokenizer2, model2)
    else:
        df.loc[i,'score2'] = 0
    print(i)
        
df.to_csv('data/tsv/self_train_scores_10k.csv', sep=',', index=False)       



#%% Generate conll tsv
df = pd.read_csv('data/tsv/self_train_scores_10k.csv', sep=',', engine='python', encoding='utf-8')  # 10,000
# Load idx2tag
with open('/home/qwang/pre-pico/app/b0_bio.json') as f:
    dat = json.load(f)    
idx2tag = dat['idx2tag']
# Load tokenizer, model
tokenizer, model = load_model(mod='bert', pre_wgts='biobert', pth_path='/media/mynewdrive/pico/exp/bert/b0_bio/last.pth.tar')      

tsv_dir = '/media/mynewdrive/pico/pmc/semi'
for i, row in df.iterrows():
    tokens, tags = pico_tag(row['text'], tokenizer, model, idx2tag)
    tsv_path = os.path.join(tsv_dir, row['pmid'] + '_conll.tsv')
    df_one_abs = pd.DataFrame({'token': tokens, 'tag': tags}
    df_one_abs.to_csv(tsv_path, sep='\t', header=False, index=False) 
