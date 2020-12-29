#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 17:02:50 2020

@author: qwang
"""


import os
os.chdir("/home/qwang/bioner/conll")

import torch
from torch.utils.data import DataLoader

from transformers import BertTokenizerFast, BertForTokenClassification
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
from transformers import AdamW, get_linear_schedule_with_warmup

from arg_parser import get_args
import utils
from bert_fn import tokenize_encode, EncodingDataset, PadDoc, train_fn, valid_fn

#%% Load data
args = get_args()
train_data = utils.load_conll(os.path.join(args.data_dir, "train.txt"))
valid_data = utils.load_conll(os.path.join(args.data_dir, "valid.txt"))
test_data = utils.load_conll(os.path.join(args.data_dir, "test.txt"))

train_seqs, train_tags = train_data[0][:1000], train_data[1][:1000]
valid_seqs, valid_tags = valid_data[0][:100], valid_data[1][:100]

# Unique tags
all_tags = train_tags + valid_tags
tag_set = set(t for tags in all_tags for t in tags)
tag2idx = {tag: idx for idx, tag in enumerate(tag_set)}
idx2tag = {idx: tag for tag, idx in tag2idx.items()}

#%% Encoding and DataLoader
# Define 'Fast' Tokenizer
if args.pre_wgts == 'distil':
    # tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')   
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')   
elif args.pre_wgts == 'biobert':
    tokenizer = BertTokenizerFast.from_pretrained('dmis-lab/biobert-v1.1')  
elif args.pre_wgts == 'pubmed-full':
    tokenizer = BertTokenizerFast.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext') 
elif args.pre_wgts == 'pubmed-abs':
    tokenizer = BertTokenizerFast.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')       
else: # args.pre_wgts == 'bert-base'
    # tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizerFast.from_pretrained('/media/mynewdrive/rob/data/pre_wgts/bert_base')   
    
# Tokenize seqs & encoding tags (set tags for non-first sub tokens to -100)
train_inputs = tokenize_encode(train_seqs, train_tags, tag2idx, tokenizer)
valid_inputs = tokenize_encode(valid_seqs, valid_tags, tag2idx, tokenizer)

# Torch Dataset
train_dataset = EncodingDataset(train_inputs)
valid_dataset = EncodingDataset(valid_inputs)

temp = train_dataset[0]
temp['input_ids']
temp['attention_mask']
temp['tags']

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=PadDoc())
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=PadDoc())

batch = next(iter(train_loader))
input_ids_batch, attn_masks_batch, tags_batch, lens = batch   

#%% Model & Optimizer & Scheduler
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
n_tags = len(tag_set)

if args.pre_wgts == 'distil':
    model = DistilBertForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=n_tags)
elif args.pre_wgts == 'biobert':
    model = BertForTokenClassification.from_pretrained("dmis-lab/biobert-v1.1", num_labels=n_tags)
elif args.pre_wgts == 'pubmed-full':
    model = BertForTokenClassification.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', num_labels=n_tags) 
elif args.pre_wgts == 'pubmed-abs':
    model = BertForTokenClassification.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract', num_labels=n_tags)        
else: # args.pre_wgts == 'bert-base'
    model = BertForTokenClassification.from_pretrained('/media/mynewdrive/rob/data/pre_wgts/bert_base', num_labels=n_tags)   
    
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

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
max_valid_f1 = float('-inf')

for epoch in range(args.epochs):   
    train_scores = train_fn(model, train_loader, idx2tag, optimizer, scheduler, tokenizer, args.clip, args.accum_step, device)
    valid_scores = valid_fn(model, valid_loader, idx2tag, tokenizer, device)

    # Update output dictionary
    output_dict['prfs'][str('train_'+str(epoch+1))] = train_scores
    output_dict['prfs'][str('valid_'+str(epoch+1))] = valid_scores
       
    # Save scores
    # if valid_scores['loss'] < min_valid_loss:
    #     min_valid_loss = valid_scores['loss']    
    is_best = (valid_scores['f1'] > max_valid_f1)
    
    # if is_best == True:   
    #     max_valid_f1 = valid_scores['f1'] 
    #     utils.save_dict_to_json(valid_scores, os.path.join(args.exp_dir, 'best_val_scores.json'))
    
    # Save model
    # if args.save_model == True:
    #     utils.save_checkpoint({'epoch': epoch+1,
    #                            'state_dict': model.state_dict(),
    #                            'optim_Dict': optimizer.state_dict()},
    #                            is_best = is_best, checkdir = args.exp_dir)
    
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
# prfs_name = os.path.basename(args.exp_dir)+'_prfs.json'
# prfs_path = os.path.join(args.exp_dir, prfs_name)
# with open(prfs_path, 'w') as fout:
#     json.dump(output_dict, fout, indent=4)