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
import torch.nn.functional as F

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
    
        if labels is None:  # prediction only. no padding
            # Remove cls/sep (tagged as -100 by tokenize_encode)
            probs_cut = emission_probs[:, 1:-1, :]  # [batch_size, seq_len-2, num_tags]
            mask_cut = attention_mask[:, 1:-1]   # [batch_size, seq_len-2]         
            preds = self.crf.decode(probs_cut, mask=mask_cut)  # assign mask for 'unpad'
            return preds  # preds: list of list containing best tag seqs for each batch
        else:
            probs_cut, mask_cut, tags_cut = [], [], []
            for prob, mask, tag in zip(emission_probs, attention_mask, labels):
                # prob: [seq_len, num_tags]
                # mask/tag: [seq_len]
                probs_cut.append(prob[tag!=-100, :])
                mask_cut.append(mask[tag!=-100])
                tags_cut.append(tag[tag!=-100])
            probs_cut = torch.stack(probs_cut)  # [batch_size, seq_len-2, num_tags]
            mask_cut = torch.stack(mask_cut)   # [batch_size, seq_len-2]
            tags_cut = torch.stack(tags_cut)  # [batch_size, seq_len-2]
                
            # log_likelihood = self.crf(F.softmax(probs_cut, dim=2), tags_cut, mask=mask_cut, reduction='mean')  
            log_likelihood = self.crf(F.log_softmax(probs_cut, dim=2), tags_cut, mask=mask_cut, reduction='mean')  
            preds = self.crf.decode(probs_cut, mask=mask_cut)  # assign mask for 'unpad'
            return preds, probs_cut, mask_cut, log_likelihood

#%%
class BERT_LSTM_CRF(BertPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        
        self.config = config
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.lstm = nn.LSTM(input_size = config.hidden_size, hidden_size = config.hidden_size,
                            num_layers = 1, dropout = config.hidden_dropout_prob, 
                            batch_first = True, bidirectional = False)
        
        self.fc = nn.Linear(config.hidden_size, config.num_labels)
        
        self.init_weights()
        self.crf = CRF(config.num_labels, batch_first=True)
        
     
    def forward(self, input_ids, attention_mask, labels=None):
               
        outputs = self.bert(input_ids, attention_mask)
        hidden_states = outputs[0]  # [batch_size, seq_len, hidden_size]
        
        out_dp = self.dropout(hidden_states)
        out_lstm, (h_n, c_n) = self.lstm(out_dp)    # [batch_size, seq_len, hidden_size]         
        emission_probs = self.fc(out_lstm)  # [batch_size, seq_len, num_tags]
        
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
            return preds, mask_cut, log_likelihood
        
        
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
        # text_lens = torch.sum(attention_mask, dim=1)  # [batch_size]
        
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
            return preds, mask_cut, log_likelihood
