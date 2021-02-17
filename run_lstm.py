#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:30:20 2020

@author: qwang
"""


import os
os.chdir("/home/qwang/bioner/conll")

import tensorflow as tf
# import tensorflow_addons as tfa
import numpy as np
import pandas as pd
from tqdm import tqdm

from arg_parser import get_args
from data_helper import DataBatch
import utils
from model import BiLSTM

#%% Load data
args = get_args()
train_data = utils.load_conll(os.path.join(args.data_dir, "train.txt"))
valid_data = utils.load_conll(os.path.join(args.data_dir, "valid.txt"))
test_data = utils.load_conll(os.path.join(args.data_dir, "test.txt"))

# idx_seq = 66
# idx_token_in_seq = 0
# train_seqs, train_tags = train_data[0], train_data[1]
# train_seqs[idx_seq]
# train_seqs[idx_seq][idx_token_in_seq]


#%%  
# Obtain batches
helper = DataBatch(train_data, valid_data, test_data)
train_batches, valid_batches, test_batches = helper.tf_pipe(args.seed, args.batch_size)

# Obtain tag2idx, idx2tag, word2idx, idx2word
tag2idx, idx2tag = helper.tag2idx, helper.idx2tag
word2idx, idx2word = helper.word2idx, helper.idx2word

# Load embedding
embed_mat = helper.load_embed(args.embed_path, args.embed_dim)

#%% Define model, optimizer, loss   
vocab_size = len(word2idx)
if args.max_vocab_size:
    vocab_size = args.max_vocab_size
                    
model = BiLSTM(vocab_size = vocab_size,
               embed_dim = args.embed_dim,
               hidden_dim = args.hidden_dim,
               num_tags = len(tag2idx),
               embed_matrix = embed_mat)
    
optimizer = tf.keras.optimizers.Adam(learning_rate = args.lr) 
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # Set "from_logits=True" may be more numerically stable

#%%
@tf.function(experimental_relax_shapes=True)  # Passing tensors with different shapes - need to relax shapes to avoid unnecessary retracing
def train_fn(model, optimizer, train_loss, batch_seq, batch_tag):
    with tf.GradientTape() as tape:
        logits, text_lens = model(batch_seq, training=True)  # [batch_size, seq_len, num_tags]
        batch_loss = loss_fn(batch_tag, logits)
        
    grads = tape.gradient(batch_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss.update_state(batch_loss)
    
    probs = tf.nn.softmax(logits, axis=2)  # [batch_size, seq_len, num_tags]
    preds = tf.argmax(probs, axis=2)  # [batch_size, seq_len]
    
    return preds, text_lens

@tf.function(experimental_relax_shapes=True)  # Relax argument shapes to avoid unnecessary retracing
def valid_fn(model, valid_loss, batch_seq, batch_tag):
    logits, text_lens = model(batch_seq)
    batch_loss = loss_fn(batch_tag, logits)
    valid_loss.update_state(batch_loss)
    
    probs = tf.nn.softmax(logits, axis=2)  # [batch_size, seq_len, num_tags]
    preds = tf.argmax(probs, axis=2)  # [batch_size, seq_len]   
      
    return preds, text_lens

#%% Run
train_loss = tf.keras.metrics.Mean(name='train_loss')
valid_loss = tf.keras.metrics.Mean(name='valid_loss')

n_train_batches = len(list(train_batches))
n_valid_batches = len(list(valid_batches))

for epoch in tf.range(1, args.epochs+1):
    
    ### Train ###
    epoch_preds, epoch_trues = [], []
    with tqdm(total = n_train_batches) as progress_bar:
        for batch_seq, batch_tag in train_batches:
            preds, text_lens = train_fn(model, optimizer, train_loss, batch_seq, batch_tag)  # [batch_size, seq_len]
            
            # Unpad preds/tags to the real lengths (for metrics)
            for pred, text_len in zip(preds, text_lens):  # logit: [seq_len, num_tags]   
                pred_cut = tf.make_ndarray(tf.make_tensor_proto(pred[:text_len])).tolist()   # convert tensor to list                
                epoch_preds.append(pred_cut)
                
            for tag, text_len in zip(batch_tag, text_lens):  # batch_tag: [seq_len, num_tags]   
                tag_cut = tf.make_ndarray(tf.make_tensor_proto(tag[:text_len])).tolist()   # convert tensor to list   
                epoch_trues.append(tag_cut)
                              
            progress_bar.update(1)
            
    # Convert epoch_idxs to epoch_tags
    epoch_tag_preds = utils.epoch_idx2tag(epoch_preds, idx2tag)
    epoch_tag_trues = utils.epoch_idx2tag(epoch_trues, idx2tag)
    # Calculate metrics for whole epoch
    train_scores = utils.scores(epoch_tag_trues, epoch_tag_preds)    
    
    ### Valid ###
    epoch_preds, epoch_trues = [], []
    with tqdm(total = n_valid_batches) as progress_bar:
        for batch_seq, batch_tag in valid_batches:
            preds, text_lens = valid_fn(model, valid_loss, batch_seq, batch_tag)
            
            # Unpad preds/tags to the real lengths (for metrics)
            for pred, text_len in zip(preds, text_lens):  # logit: [seq_len, num_tags]   
                pred_cut = tf.make_ndarray(tf.make_tensor_proto(pred[:text_len])).tolist()   # convert tensor to list                
                epoch_preds.append(pred_cut)
                
            for tag, text_len in zip(batch_tag, text_lens):  # batch_tag: [seq_len, num_tags]   
                tag_cut = tf.make_ndarray(tf.make_tensor_proto(tag[:text_len])).tolist()   # convert tensor to list   
                epoch_trues.append(tag_cut)
                
            progress_bar.update(1)
    
    # Convert epoch_idxs to epoch_tags
    epoch_tag_preds = utils.epoch_idx2tag(epoch_preds, idx2tag)
    epoch_tag_trues = utils.epoch_idx2tag(epoch_trues, idx2tag)
    # Calculate metrics for whole epoch
    valid_scores = utils.scores(epoch_tag_trues, epoch_tag_preds)
     
    
    tf.print(tf.strings.format('\n[Epoch{}|train] loss: {}, f1: {}, rec: {}, prec: {}, acc: {}', 
                               (epoch, train_loss.result(), 
                                train_scores['f1'], train_scores['rec'], 
                                train_scores['prec'], train_scores['acc'])))
    
    tf.print(tf.strings.format('[Epoch{}|valid] loss: {}, f1: {}, rec: {}, prec: {}, acc: {}\n', 
                               (epoch, valid_loss.result(), 
                                valid_scores['f1'], valid_scores['rec'], 
                                valid_scores['prec'], valid_scores['acc'])))
    
    train_loss.reset_states()
    valid_loss.reset_states()
    

# Save model weights    
model.save_weights(f"{args.exp_dir}/tf_model_wgts", save_format='tf')
# # Save model architecture and args   
# model.save(f"{args.exp_dir}/tf_model", save_format="tf")


#%% Evaluation on valid/test set (classification report)
from seqeval.metrics import classification_report
def cls_report(batches, wgt_file, cls_file):
    model.load_weights(os.path.join(args.exp_dir, wgt_file))
    
    epoch_preds, epoch_trues = [], []
    for batch_seq, batch_tag in batches:
        preds, text_lens = valid_fn(model, valid_loss, batch_seq, batch_tag)
        
        # Unpad preds/tags to the real lengths (for metrics)
        for pred, text_len in zip(preds, text_lens):  # logit: [seq_len, num_tags]   
            pred_cut = tf.make_ndarray(tf.make_tensor_proto(pred[:text_len])).tolist()   # convert tensor to list                
            epoch_preds.append(pred_cut)
            
        for tag, text_len in zip(batch_tag, text_lens):  # batch_tag: [seq_len, num_tags]   
            tag_cut = tf.make_ndarray(tf.make_tensor_proto(tag[:text_len])).tolist()   # convert tensor to list   
            epoch_trues.append(tag_cut)

    epoch_tag_preds = utils.epoch_idx2tag(epoch_preds, idx2tag)
    epoch_tag_trues = utils.epoch_idx2tag(epoch_trues, idx2tag)
    
    print(classification_report(epoch_tag_trues, epoch_tag_preds, output_dict=False))
    report = classification_report(epoch_tag_trues, epoch_tag_preds, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(os.path.join(args.exp_dir, cls_file), sep=',', header=True, index=True)

cls_report(valid_batches, wgt_file='tf_model_wgts', cls_file='cls_valid.csv')
cls_report(test_batches, wgt_file='tf_model_wgts', cls_file='cls_test.csv')

   
#%% Predict
from predict import pred_one_text
text = '''Talks over a post-Brexit trade agreement will resume later, after the UK and EU agreed to "go the extra mile" in search of a breakthrough.'''
seq_tagged = pred_one_text(text, word2idx, idx2tag, model)
for token, tag in seq_tagged:
    if tag != 'O':
        print(token, tag)