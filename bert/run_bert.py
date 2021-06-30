#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 17:02:50 2020

@author: qwang
"""


import os
os.chdir("/home/qwang/pre-pico/bert")
# import sys
# sys.path[0] = sys.path[0][:-5]

import random
import json
import torch
from torch.utils.data import DataLoader

from transformers import BertTokenizerFast, BertForTokenClassification
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
from transformers import AdamW, get_linear_schedule_with_warmup

from bert_args import get_args
import bert_utils
from bert_utils import tokenize_encode, EncodingDataset, PadDoc, plot_prfs

import bert_fn
import bert_crf_fn
from bert_model import BERT_CRF, BERT_LSTM_CRF, Distil_CRF

#%% Load data
args = get_args()
args.epochs = 20
args.lr = 1e-3
args.warm_frac = 0.1 # 0.1
args.exp_dir = "/media/mynewdrive/pico/exp/bert_crf/temp"
args.pre_wgts = 'pubmed-full'   # ['distil', 'bert', 'biobert', 'pubmed-full', 'pubmed-abs']
args.model = 'bert_crf'  # ['bert', 'bert_crf', 'bert_lstm_crf', 'distil', 'distil_crf']
args.save_model = True

with open('/media/mynewdrive/pico/exp/bert_crf/bc6_full/bc6_full_prfs.json') as f:
    js = json.load(f) 
from argparse import Namespace
args = Namespace(**js['args'])
idx2tag = js['idx2tag']
idx2tag = {int(idx): tag for idx, tag in idx2tag.items()}
tag2idx = {tag: idx for idx, tag in idx2tag.items()}

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True     
torch.backends.cudnn.benchmark = False   # This makes things slower  

# Load json file
json_path = os.path.join(args.data_dir, "tsv/18mar_output/pico_18mar.json")
# json_path = os.path.join(args.data_dir, "tsv/output/b1.json")
train_seqs, train_tags = bert_utils.load_pico(json_path, group='train')
valid_seqs, valid_tags = bert_utils.load_pico(json_path, group='valid')
test_seqs, test_tags = bert_utils.load_pico(json_path, group='test')

# train_data = bert_utils.load_conll(os.path.join(args.data_dir, "train.txt"))
# valid_data = bert_utils.load_conll(os.path.join(args.data_dir, "valid.txt"))
# test_data = bert_utils.load_conll(os.path.join(args.data_dir, "test.txt"))

# Unique tags
# all_tags = train_tags + valid_tags
# tag_set = set(t for tags in all_tags for t in tags)
# tag2idx = {tag: idx for idx, tag in enumerate(tag_set)}
# idx2tag = {idx: tag for tag, idx in tag2idx.items()}


#%% Encoding and DataLoader
n_tags = 13
# Define 'Fast' Tokenizer
if args.pre_wgts == 'distil':
    # tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')   
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased', num_labels=n_tags)   
elif args.pre_wgts == 'biobert':
    tokenizer = BertTokenizerFast.from_pretrained('dmis-lab/biobert-v1.1', num_labels=n_tags)  
elif args.pre_wgts == 'pubmed-full':
    tokenizer = BertTokenizerFast.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', num_labels=n_tags) 
elif args.pre_wgts == 'pubmed-abs':
    tokenizer = BertTokenizerFast.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract', num_labels=n_tags)       
else: # args.pre_wgts == 'bert-base'
    # tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', num_labels=n_tags)   
    
# Tokenize seqs & encoding tags (set tags for non-first sub tokens to -100)
train_inputs = tokenize_encode(train_seqs, train_tags, tag2idx, tokenizer)
valid_inputs = tokenize_encode(valid_seqs, valid_tags, tag2idx, tokenizer)
test_inputs = tokenize_encode(test_seqs, test_tags, tag2idx, tokenizer)

# Torch Dataset
train_dataset = EncodingDataset(train_inputs)
valid_dataset = EncodingDataset(valid_inputs)
test_dataset = EncodingDataset(test_inputs)

# temp = train_dataset[99]
# temp['tags']
# temp['input_ids']
# temp['attention_mask']
# temp['tags']

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=PadDoc())
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=PadDoc())
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=PadDoc())

# batch = next(iter(train_loader))
# input_ids_batch, attn_masks_batch, tags_batch, lens = batch   

#%% Model & Optimizer & Scheduler
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if args.pre_wgts == 'distil':
    pre_wgts = "distilbert-base-uncased"
elif args.pre_wgts == 'biobert':
    pre_wgts = "dmis-lab/biobert-v1.1"
elif args.pre_wgts == 'pubmed-full':
    pre_wgts = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
elif args.pre_wgts == 'pubmed-abs':
    pre_wgts = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"       
else: # args.pre_wgts == 'bert-base'
    pre_wgts = "bert-base-uncased" 


if args.model == 'bert':
    model = BertForTokenClassification.from_pretrained(pre_wgts, num_labels=n_tags) 
if args.model == 'bert_crf':
    model = BERT_CRF.from_pretrained(pre_wgts, num_labels=n_tags)
if args.model == 'bert_lstm_crf':
    model = BERT_LSTM_CRF.from_pretrained(pre_wgts, num_labels=n_tags)
if args.model == 'distil':
    model = DistilBertForTokenClassification.from_pretrained(pre_wgts, num_labels=n_tags)
if args.model == 'distil_crf':
    model = Distil_CRF.from_pretrained(pre_wgts, num_labels=n_tags)
    
model.to(device)
optimizer = AdamW(model.parameters(), lr=args.lr)

# Slanted triangular Learning rate scheduler
total_steps = len(train_loader) * args.epochs // args.accum_step
warm_steps = int(total_steps * args.warm_frac)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_steps, num_training_steps=total_steps)

#%% Train the model
if os.path.exists(args.exp_dir) == False:
    os.makedirs(args.exp_dir)   
       
# Create args and output dictionary (for json output)
output_dict = {'args': vars(args), 'prfs': {}}

# For early stopping
n_worse = 0
min_valid_loss = float('inf')

for epoch in range(args.epochs):  
    if args.model in ['distil_crf', 'bert_crf', 'bert_lstm_crf']:
        train_scores = bert_crf_fn.train_fn(model, train_loader, idx2tag, optimizer, scheduler, tokenizer, args.clip, args.accum_step, device)
        valid_scores = bert_crf_fn.valid_fn(model, valid_loader, idx2tag, tokenizer, device)
    if args.model in ['distil', 'bert']:
        train_scores = bert_fn.train_fn(model, train_loader, idx2tag, optimizer, scheduler, tokenizer, args.clip, args.accum_step, device)
        valid_scores = bert_fn.valid_fn(model, valid_loader, idx2tag, tokenizer, device)

    # Update output dictionary
    output_dict['prfs'][str('train_'+str(epoch+1))] = train_scores
    output_dict['prfs'][str('valid_'+str(epoch+1))] = valid_scores
       
    # Save scores 
    is_best = (valid_scores['loss'] < min_valid_loss)
    if is_best == True:   
        min_valid_loss = valid_scores['loss']
    
    # Save model
    if args.save_model == True:
        bert_utils.save_checkpoint({'epoch': epoch+1,
                                    'state_dict': model.state_dict(),
                                    'optim_Dict': optimizer.state_dict()},
                                     is_best = is_best, checkdir = args.exp_dir)
    
    print("\n\nEpoch {}/{}...".format(epoch+1, args.epochs))
    print('[Train] loss: {0:.3f} | acc: {1:.2f}% | f1: {2:.2f}% | prec: {3:.2f}% | rec: {4:.2f}%'.format(
        train_scores['loss'], train_scores['acc']*100, train_scores['f1']*100, train_scores['prec']*100, train_scores['rec']*100))
    print('[Valid] loss: {0:.3f} | acc: {1:.2f}% | f1: {2:.2f}% | prec: {3:.2f}% | rec: {4:.2f}%\n'.format(
        valid_scores['loss'], valid_scores['acc']*100, valid_scores['f1']*100, valid_scores['prec']*100, valid_scores['rec']*100))
    
    # Early stopping             
    # if valid_scores['loss']-min_valid_loss > 0: # args.stop_c1) and (max_valid_f1-valid_scores['f1'] > args.stop_c2):
    #     n_worse += 1
    # if n_worse == 5: # args.stop_p:
    #     print("Early stopping")
    #     break
        
# Write performance and args to json
prfs_name = os.path.basename(args.exp_dir)+'_prfs.json'
prfs_path = os.path.join(args.exp_dir, prfs_name)
output_dict['idx2tag'] = idx2tag
with open(prfs_path, 'w') as fout:
    json.dump(output_dict, fout, indent=4)
    
#%% Evaluation on valid/test set (classification report)
from seqeval.metrics import classification_report
from torchcrf import CRF
# crf = CRF(13, batch_first=True)

def cls_report(data_loader, pth_path, add_crf=True, device=torch.device('cpu')):
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
        word_ids = batch[4].to(device)  # [batch_size, seq_len]
        # print(true_lens)
        
        if add_crf == True:          
            preds_cut, probs_cut, mask_cut, log_likelihood = model(input_ids, attention_mask = attn_mask, labels = tags) 
            # preds_cut = model.crf.decode(probs_cut, mask=mask_cut)                      
            for sin_preds, sin_tags, sin_lens, sin_wids in zip(preds_cut, tags, true_lens, word_ids):
                sin_wids = sin_wids[1:sin_lens+1]
                sin_tags = sin_tags[1:sin_lens+1] # list of lists (1st/last tag is -100 so need to move one step)
                
                pre_wid = None
                sin_preds_new, sin_tags_new = [], []
                for p, t, wid in zip(sin_preds, sin_tags, sin_wids):
                    if wid != pre_wid:
                        sin_preds_new.append(p)
                        sin_tags_new.append(t.tolist())
                    pre_wid = wid
                epoch_preds.append(sin_preds_new)   # list of lists                 
                epoch_trues.append(sin_tags_new)  
        else:     
            outputs = model(input_ids, attention_mask = attn_mask, labels = tags)   
            logits =  outputs[1]  # [batch_size, seq_len, num_tags]
            preds = torch.argmax(logits, dim=2)  # [batch_size, seq_len]    
            
            for sin_preds, sin_tags, sin_lens, sin_wids in zip(preds, tags, true_lens, word_ids):
                # list of lists (1st/last tag is -100 so need to move one step)
                sin_wids = sin_wids[1:sin_lens+1]
                sin_tags = sin_tags[1:sin_lens+1] 
                sin_preds = sin_preds[1:sin_lens+1]
                
                pre_wid = None
                sin_preds_new, sin_tags_new = [], []
                for p, t, wid in zip(sin_preds, sin_tags, sin_wids):
                    if wid != pre_wid:
                        sin_preds_new.append(p.tolist())
                        sin_tags_new.append(t.tolist())
                    pre_wid = wid
                epoch_preds.append(sin_preds_new)   # list of lists                 
                epoch_trues.append(sin_tags_new)  
    
    # Convert epoch_idxs to epoch_tags
    epoch_tag_preds = bert_utils.epoch_idx2tag(epoch_preds, idx2tag)
    epoch_tag_trues = bert_utils.epoch_idx2tag(epoch_trues, idx2tag)
    print(classification_report(epoch_tag_trues, epoch_tag_preds, output_dict=False, digits=4))    

pth_path = os.path.join(args.exp_dir, 'best.pth.tar')
cls_report(valid_loader, pth_path, add_crf=True)
cls_report(test_loader, pth_path, add_crf=True)
pth_path = os.path.join(args.exp_dir, 'last.pth.tar')
cls_report(valid_loader, pth_path, add_crf=True)
cls_report(test_loader, pth_path, add_crf=True)
