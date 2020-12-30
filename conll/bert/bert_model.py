#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 12:45:12 2020

@author: qwang
"""

from transformers import BertPreTrainedModel, BertConfig, BertModel
from transformers import DistilBertPreTrainedModel, DistilBertConfig, DistilBertModel

import torch
import torch.nn as nn
from torchcrf import CRF

#%%
class BERT_CRF(BertPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        
        self.config = config
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.fc = nn.Linear(config.hidden_size, config.num_labels)
        
        self.init_weights()
        self.crf = CRF(config.num_labels, batch_first=True)
        
     
    def forward(self, input_ids, attention_mask, labels=None):
               
        outputs = self.bert(input_ids, attention_mask)
        hidden_states = outputs[0]  # [batch_size, seq_len, hidden_size]
        
        out_dp = self.dropout(hidden_states)
        emission_probs = self.fc(out_dp)  # [batch_size, seq_len, num_tags]
        
        attention_mask = attention_mask.type(torch.uint8)
        
        probs_cut, mask_cut, tags_cut = [], [], []
        for prob, mask, tag in zip(emission_probs, attention_mask, labels):
            # prob: [seq_len, num_tags]
            # mask/tag: [seq_len]
            probs_cut.append(prob[tag!=-100, :])
            mask_cut.append(mask[tag!=-100])
            tags_cut.append(tag[tag!=-100])
        
        probs_cut = torch.stack(probs_cut) 
        mask_cut = torch.stack(mask_cut) 
        tags_cut = torch.stack(tags_cut) 
                     
              
        if labels is None:
            preds = self.crf.decode(probs_cut, mask=mask_cut)  # assign mask for 'unpad'
            return preds  # preds: list of list containing best tag seqs for each batch
        else:
            log_likelihood = self.crf(probs_cut, tags_cut, mask=mask_cut)  # [batch_size]
            preds = self.crf.decode(probs_cut, mask=mask_cut)  # assign mask for 'unpad'
            return preds, log_likelihood

#%%
class Distil_CRF(DistilBertPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        
        self.config = config
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.hidden_size, config.num_labels)
        
        self.init_weights()
        self.crf = CRF(config.num_labels, batch_first=True)
        
     
    def forward(self, input_ids, attention_mask, labels=None):
        
        # Calculate true text_lens
        text_lens = torch.sum(attention_mask, dim=1)  # [batch_size]
        
        outputs = self.distilbert(input_ids, attention_mask)
        hidden_states = outputs[0]  # [batch_size, seq_len, hidden_size] 
        out_dp = self.dropout(hidden_states)
        emission_probs = self.fc(out_dp)  # [batch_size, seq_len, num_tags]
        
        attention_mask = attention_mask.type(torch.uint8)
        probs_cut, mask_cut, tags_cut = [], [], []
        for prob, mask, tag in zip(emission_probs, attention_mask, labels):
            # prob: [seq_len, num_tags]
            # mask/tag: [seq_len]
            probs_cut.append(prob[tag!=-100, :])
            mask_cut.append(mask[tag!=-100])
            tags_cut.append(tag[tag!=-100])
        
        probs_cut = torch.stack(probs_cut) 
        mask_cut = torch.stack(mask_cut) 
        tags_cut = torch.stack(tags_cut) 
                     
              
        if labels is None:
            preds = self.crf.decode(probs_cut, mask=mask_cut)  # assign mask for 'unpad'
            return preds  # preds: list of list containing best tag seqs for each batch
        else:
            log_likelihood = self.crf(probs_cut, tags_cut, mask=mask_cut)  # [batch_size]
            preds = self.crf.decode(probs_cut, mask=mask_cut)  # assign mask for 'unpad'
            return preds, log_likelihood
