#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 12:49:15 2021

@author: qwang
"""


import os
os.chdir("/home/qwang/pre-pico")
import random
import json
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import BertTokenizerFast, BertForTokenClassification
from transformers import AdamW, get_linear_schedule_with_warmup

from bert import bert_utils, bert_fn

#%% Load args
with open('/home/qwang/pre-pico/app/b0_bio.json') as f:
    js = json.load(f) 
from argparse import Namespace
args = Namespace(**js['args'])

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True     
torch.backends.cudnn.benchmark = False

idx2tag = js['idx2tag']
tag2idx = {tag: idx for idx, tag in idx2tag.items()}

softmax = nn.Softmax(dim=1)
tokenizer = BertTokenizerFast.from_pretrained('dmis-lab/biobert-v1.1', num_labels=13) 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#%%
class SemiData():
    def __init__(self):   
        self.semi_path = '/home/qwang/pre-pico/data/semi_scores_9923.csv'
        self.gold_path = '/home/qwang/pre-pico/data/tsv/18mar_output/pico_18mar.json'
        
    
    def read_conll_tsv(self, tsv_path):
        ''' Read seqs/tags for one tsv file
            seqs[i] --> ['Leicestershire', '22', 'points', ',', ...], tags[i] --> ['B-ORG', 'O', 'O', ...]
        '''
        dat = pd.read_csv(tsv_path, sep='\t', header=None)
        dat = dat.dropna()
        seqs = list(dat[0])
        tags = list(dat[1])
        return seqs, tags
    
    
    def load_semi(self):
        ''' Load semi seqs/tags 
        '''
        df = pd.read_csv(self.semi_path, sep=',', engine='python', encoding='utf-8')  
        semi_seqs, semi_tags = [], []
        for _, row in df.iterrows():
            seqs, tags = self.read_conll_tsv(row['path'])
            semi_seqs.append(seqs)
            semi_tags.append(tags)
        self.semi_seqs = semi_seqs
        self.semi_tags = semi_tags
        
        
    def load_gold(self):
        ''' Load gold train/valid/test seqs/tags 
        '''
        json_path = self.gold_path
        self.gold_seqs, self.gold_tags = bert_utils.load_pico(json_path, group='train')
        self.gold_valid_seqs, self.gold_valid_tags = bert_utils.load_pico(json_path, group='valid')
        self.gold_test_seqs, self.gold_test_tags = bert_utils.load_pico(json_path, group='test')
        
        
    def load_model(self, pth_path='/media/mynewdrive/pico/exp/bert/b0_bio/last.pth.tar', n_tags=13): 
        # Load tokenizer, model
        model = BertForTokenClassification.from_pretrained('dmis-lab/biobert-v1.1', num_labels=13)
        # Load checkpoint
        checkpoint = torch.load(pth_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.cpu()
        model.eval()
        self.model = model


    def pico_score(self, text, tokenizer, model): 
        ''' Calculate score across tokens for one semi text
        '''
        # Tokenization
        inputs = tokenizer([text], is_split_into_words=True, return_offsets_mapping=True, padding=False, truncation=True, max_length=512)
        inputs = {key: torch.tensor(value) for key, value in inputs.items()} 
        # Run model
        outputs = model(inputs['input_ids'].unsqueeze(0), inputs['attention_mask'].unsqueeze(0)) 
        logits = outputs[0].squeeze(0)[1:-1]  # [seq_len-2, n_tags], remove cls/sep token
    
        probs = softmax(logits)  # [seq_len-2, n_tags]
        probs_max = torch.max(probs, dim=1).values  # [seq_len-2], max probs of each record across tags
        score = torch.mean(probs_max).detach().numpy()  # average probs_max across all tokens 
        return score
    
    
    def cand_splitter(self, cand_seqs, cand_tags, thres):
        ''' Compute PICO scores for candidates, split cand to incl/excl
        '''
        scores = []  
        with tqdm(total=len(cand_seqs)) as progress_bar:
            for seq in cand_seqs:
                text = ' '.join(seq)
                score = self.pico_score(text, self.tokenizer, self.model)
                scores.append(score)
                progress_bar.update(1)
                
        incl_seqs, incl_tags = [], []
        excl_seqs, excl_tags = [], []
        for seq, tag, score in zip(cand_seqs, cand_tags, scores):
            if score > thres:
                incl_seqs.append(seq)
                incl_tags.append(tag)
            else:
                excl_seqs.append(seq)
                excl_tags.append(tag)
                
        return incl_seqs, incl_tags, excl_seqs, excl_tags
    

#%%
st = SemiData()
st.load_semi()
st.load_gold()
st.load_model()

# Init
seqs, tags = st.gold_seqs, st.gold_tags
cand_seqs , cand_tags = st.semi_seqs, st.semi_tags

# Split
incl_seqs, incl_tags, excl_seqs, excl_tags = st.cand_splitter(cand_seqs, cand_tags, thres=0.99)

# Update        
cand_seqs, cand_tags = excl_seqs, excl_tags    
seqs = seqs + incl_seqs
tags = tags + incl_tags

#%% Loader for training
class SemiLoader():
    def __init__(self, seqs, tags):
        self.seqs = seqs
        self.tags = tags
    
    def loader(self, tag2idx, tokenizer, ratio=0.8):
        # Split enlarged data into train/valid
        train_size = int(len(self.seqs)*ratio)
        train_seqs, valid_seqs = self.seqs[:train_size], self.seqs[train_size:]
        train_tags, valid_tags = self.tags[:train_size], self.tags[train_size:]

        # Tokenize seqs & encoding tags (set tags for non-first sub tokens to -100)
        train_inputs = bert_utils.tokenize_encode(train_seqs, train_tags, tag2idx, tokenizer)
        valid_inputs = bert_utils.tokenize_encode(valid_seqs, valid_tags, tag2idx, tokenizer)
        
        # Torch Dataset
        train_dataset = bert_utils.EncodingDataset(train_inputs)
        valid_dataset = bert_utils.EncodingDataset(valid_inputs)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=bert_utils.PadDoc())
        valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True, collate_fn=bert_utils.PadDoc())
        
        return train_loader, valid_loader

#%% Model & Optimizer & Scheduler
class SelfTrain():
    def __init__(self, args, loader_len):   
        ''' Init model/optimizer/scheduler 
        '''
        model = BertForTokenClassification.from_pretrained('dmis-lab/biobert-v1.1', num_labels=13)
        model.to(torch.device('cpu'))
        optimizer = AdamW(model.parameters(), lr=args.lr)
        total_steps = loader_len * args.epochs // args.accum_step
        warm_steps = int(total_steps * args.warm_frac)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_steps, num_training_steps=total_steps)
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        
    def train(self, it, train_loader, valid_loader):
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        
        iter_dir = '/media/mynewdrive/pico/semi/iter_' + str(it)
        if os.path.exists(iter_dir) == False:
            os.makedirs(iter_dir)   
               
        prfs_dict = {'prfs': {}}
        min_valid_loss = float('inf')
        for epoch in range(args.epochs):  
        
            train_scores = bert_fn.train_fn(model, train_loader, idx2tag, optimizer, scheduler, tokenizer, args.clip, args.accum_step, device)
            valid_scores = bert_fn.valid_fn(model, valid_loader, idx2tag, tokenizer, device)
        
            # Update prfs dictionary
            prfs_dict['prfs'][str('train_'+str(epoch+1))] = train_scores
            prfs_dict['prfs'][str('valid_'+str(epoch+1))] = valid_scores
               
           # Save scores 
            is_best = (valid_scores['loss'] < min_valid_loss)
            if is_best == True:   
                min_valid_loss = valid_scores['loss']
            
            # Save model
            bert_utils.save_checkpoint({'epoch': epoch+1,
                                        'state_dict': model.state_dict(),
                                        'optim_Dict': optimizer.state_dict()},
                                         is_best = is_best, checkdir = iter_dir)
            
            print("\n\nEpoch {}/{}...".format(epoch+1, args.epochs))
            print('[Train] loss: {0:.3f} | f1: {1:.2f}% | prec: {2:.2f}% | rec: {3:.2f}%'.format(
                train_scores['loss'], train_scores['f1']*100, train_scores['prec']*100, train_scores['rec']*100))
            print('[Valid] loss: {0:.3f} | f1: {1:.2f}% | prec: {2:.2f}% | rec: {3:.2f}%\n'.format(
                valid_scores['loss'], valid_scores['f1']*100, valid_scores['prec']*100, valid_scores['rec']*100))
            
        # Write performance to json
        prfs_name = os.path.basename(iter_dir)+'_prfs.json'
        prfs_path = os.path.join(iter_dir, prfs_name)
        with open(prfs_path, 'w') as fout:
            json.dump(prfs_dict, fout, indent=4)

#%%
incl_seqs, incl_tags, excl_seqs, excl_tags = selector(cand_seqs, cand_tags, scores, thres=0.99)





