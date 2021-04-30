#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 12:49:15 2021

@author: qwang
"""


import os
os.chdir("/home/qwang/pre-pico/bert")
import random
import json
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import BertTokenizerFast, BertForTokenClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import logging
logging.set_verbosity_error()

import bert_utils, bert_fn

#%% Load args
with open('/home/qwang/pre-pico/app/b0_bio.json') as f:
    js = json.load(f) 
from argparse import Namespace
args = Namespace(**js['args'])
# args.epochs = 3

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True     
torch.backends.cudnn.benchmark = False

idx2tag = js['idx2tag']
idx2tag = {int(idx): tag for idx, tag in idx2tag.items()}
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
        print('\n\nsemi data loading finished (size {})\n\n'.format(len(semi_seqs)))
        
        
    def load_gold(self):
        ''' Load gold train/valid/test seqs/tags 
        '''
        json_path = self.gold_path
        self.gold_seqs, self.gold_tags = bert_utils.load_pico(json_path, group='train')
        self.gold_valid_seqs, self.gold_valid_tags = bert_utils.load_pico(json_path, group='valid')
        self.gold_test_seqs, self.gold_test_tags = bert_utils.load_pico(json_path, group='test')
        print('\n\ngold data loading finished (train size {})\n\n'.format(len(self.gold_seqs)))
        
        
    def load_model(self, pth_path='/media/mynewdrive/pico/exp/bert/b0_bio/last.pth.tar', n_tags=13): 
        # Load tokenizer, model
        model = BertForTokenClassification.from_pretrained('dmis-lab/biobert-v1.1', num_labels=13)
        # Load checkpoint
        checkpoint = torch.load(pth_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.cpu()
        model.eval()
        self.model = model
        print('\n\nModel loaded: {}\n\n'.format(pth_path))


    def pico_score(self, text, model): 
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
                score = self.pico_score(text, self.model)
                scores.append(score)
                progress_bar.update(1)
        print('\n\nPICO score computing finished (cand size {})\n\n'.format(len(cand_seqs)))
                
        incl_seqs, incl_tags = [], []
        excl_seqs, excl_tags = [], []
        for seq, tag, score in zip(cand_seqs, cand_tags, scores):
            if score > thres:
                incl_seqs.append(seq)
                incl_tags.append(tag)
            else:
                excl_seqs.append(seq)
                excl_tags.append(tag)
        print('\n\ncand split finished (incl {}, excl {})\n\n'.format(len(incl_seqs), len(excl_seqs)))        
        return incl_seqs, incl_tags, excl_seqs, excl_tags
    

#%% Loader for training
class SemiLoader():
    def __init__(self, seqs, tags):
        self.seqs = seqs
        self.tags = tags
    
    def loader(self, tag2idx, ratio=0.8):
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
class SemiTrainer():
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
        
        
    def train(self, it, train_loader, valid_loader, iter_dir):
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        
        if os.path.exists(iter_dir) == False:
            os.makedirs(iter_dir)   
               
        prfs_dict = {'iter': it, 'prfs': {}}
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
            
            print("\n\nIter {} - Epoch {}/{}...".format(it, epoch+1, args.epochs))
            if epoch == 19:
                print('[Train] loss: {0:.3f} | f1: {1:.2f}% | prec: {2:.2f}% | rec: {3:.2f}%'.format(
                    train_scores['loss'], train_scores['f1']*100, train_scores['prec']*100, train_scores['rec']*100))
                print('[Valid] loss: {0:.3f} | f1: {1:.2f}% | prec: {2:.2f}% | rec: {3:.2f}%\n'.format(
                    valid_scores['loss'], valid_scores['f1']*100, valid_scores['prec']*100, valid_scores['rec']*100))
        
        return prfs_dict
        

#%% Init
sd = SemiData()
sd.load_semi()  # semi_seqs/tags
sd.load_gold()  # gold_seqs/tags, gold_valid_seqs/tags, gold_test_seqs/tsgs
# Init model
sd.load_model()  # '/media/mynewdrive/pico/exp/bert/b0_bio/last.pth.tar'
# Initital train+valid set (gold train+valid)
seqs, tags = sd.gold_seqs, sd.gold_tags
# Initial candidates (all semi set)
cand_seqs, cand_tags = sd.semi_seqs, sd.semi_tags


#%% Self-train iterations
iters = 10
thres = 0.99
for it in range(1, iters+1):
    
    incl_seqs, incl_tags, excl_seqs, excl_tags = sd.cand_splitter(cand_seqs, cand_tags, thres)
    
    if len(incl_seqs) == 0:
        exit('No new records included')
    else:
        # Enlarged train set
        seqs = seqs + incl_seqs
        tags = tags + incl_tags
        
        # Loader for train
        sloader = SemiLoader(seqs, tags)
        train_loader, valid_loader = sloader.loader(tag2idx)
        
        # Train
        iter_dir = '/media/mynewdrive/pico/semi/iter_' + str(it)
        strainer = SemiTrainer(args, len(train_loader))
        prfs_dict = strainer.train(it, train_loader, valid_loader, iter_dir)
        
        # Output prfs
        prfs_dict['train_size'] = int(len(seqs)*0.8)
        prfs_name = os.path.basename(iter_dir)+'_prfs.json'
        prfs_path = os.path.join(iter_dir, prfs_name)
        with open(prfs_path, 'w') as fout:
            json.dump(prfs_dict, fout, indent=4)
        
        # Update cand        
        cand_seqs, cand_tags = excl_seqs, excl_tags 
        # Update model for PICO scores
        sd.load_model(pth_path = os.path.join(iter_dir, 'last.pth.tar'))
        

#%% Evaluation on valid/test set 
from seqeval.metrics import classification_report

def cls_report(data_loader, pth_path, device=torch.device('cpu')):
    model = BertForTokenClassification.from_pretrained('dmis-lab/biobert-v1.1', num_labels=13)
    # Load checkpoin
    checkpoint = torch.load(pth_path, map_location=device)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.cpu()
    model.eval()
    
    epoch_preds, epoch_trues = [], []
    for j, batch in enumerate(data_loader):                      
        
        input_ids = batch[0].to(device)  # [batch_size, seq_len]
        attn_mask = batch[1].to(device)  # [batch_size, seq_len]
        tags = batch[2].to(device)  # [batch_size, seq_len]
        true_lens = batch[3]  # [batch_size]
        
        outputs = model(input_ids, attention_mask = attn_mask, labels = tags)   
        logits =  outputs[1]  # [batch_size, seq_len, num_tags]
        preds = torch.argmax(logits, dim=2)  # [batch_size, seq_len]            
        # Append preds/trues with real seq_lens (before padding) to epoch_samaple_preds/trues
        for p, t, l in zip(preds, tags, true_lens):
            epoch_preds.append(p[1:l+1].tolist())  # p[:l].shape = l
            epoch_trues.append(t[1:l+1].tolist())  # t[:l].shape = l             
    
    # Convert epoch_idxs to epoch_tags
    # Remove ignored index (-100)
    epoch_preds_cut, epoch_trues_cut = [], []
    for preds, trues in zip(epoch_preds, epoch_trues):  # per sample
        preds_cut = [p for (p, t) in zip(preds, trues) if t != -100]
        trues_cut = [t for (p, t) in zip(preds, trues) if t != -100]               
        epoch_preds_cut.append(preds_cut)
        epoch_trues_cut.append(trues_cut)
    epoch_tag_preds = bert_utils.epoch_idx2tag(epoch_preds_cut, idx2tag)
    epoch_tag_trues = bert_utils.epoch_idx2tag(epoch_trues_cut, idx2tag)
    print(classification_report(epoch_tag_trues, epoch_tag_preds, output_dict=False, digits=4))    


# valid/test loader
valid_inputs = bert_utils.tokenize_encode(sd.gold_valid_seqs, sd.gold_valid_tags, tag2idx, tokenizer)
valid_dataset = bert_utils.EncodingDataset(valid_inputs)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True, collate_fn=bert_utils.PadDoc())

test_inputs = bert_utils.tokenize_encode(sd.gold_test_seqs, sd.gold_test_tags, tag2idx, tokenizer)
test_dataset = bert_utils.EncodingDataset(test_inputs)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, collate_fn=bert_utils.PadDoc())


pth_path = os.path.join('/media/mynewdrive/pico/semi/iter_1', 'last.pth.tar')
cls_report(valid_loader, pth_path)
cls_report(test_loader, pth_path)
