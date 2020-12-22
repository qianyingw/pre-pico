#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:25:23 2020

@author: qwang
"""

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa

#%%
class BiLSTM(tf.keras.Model):
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_tags, embed_matrix):
        super(BiLSTM, self).__init__()
        
        self.embed = layers.Embedding(input_dim = vocab_size, output_dim = embed_dim, trainable=True,
                                      embeddings_initializer = tf.keras.initializers.Constant(embed_matrix))                                      
        
        self.bilstm = layers.Bidirectional(layers.LSTM(units = hidden_dim, return_sequences=True, return_state=False))  
        self.fc = layers.Dense(units = num_tags)
        # self.fc = layers.dense(units = num_tags, activation='relu')
        # self.softmax = layers.Activation('softmax')   
    
    def call(self, text):
        '''
            text: [batch_size, seq_len]
        '''
        out_embed = self.embed(text)  # [batch_size, seq_len, embed_dim]
        out_lstm = self.bilstm(out_embed)  # [batch_size, seq_len, hidden_dim*2]
        probs = self.fc(out_lstm)  # [batch_size, seq_len, num_tags]
        # probs = self.softmax(probs)  # [batch_size, seq_len, num_tags]
        
        return probs

#%% 
class CRF(tf.keras.Model):
    
    def __init__(self, vocab_size, embed_dim, num_tags, embed_matrix):
        super(CRF, self).__init__()
        
        self.embed = layers.Embedding(input_dim = vocab_size, output_dim = embed_dim, trainable=True,
                                      embeddings_initializer = tf.keras.initializers.Constant(embed_matrix))
        
        self.dropout = layers.Dropout(rate = 0.5)
        self.fc = layers.Dense(units = num_tags)
        
        self.trans_pars = tf.Variable(tf.random.uniform(shape = (num_tags, num_tags)))

    def call(self, text, tags=None, training=None):
        '''
            text: [batch_size, seq_len]
            tags: [batch_size, seq_len]
        '''
        
        non_zeros = tf.cast(tf.math.not_equal(text, 0), dtype=tf.int32)  # [batch_size, seq_len]
        text_lens = tf.math.reduce_sum(non_zeros, axis=1)  # [batch_size]
        
        out_embed = self.embed(text)  # [batch_sizem seq_len, embed_dim]
        out_dp = self.dropout(out_embed, training)  # [batch_sizem seq_len, embed_dim]
        probs = self.fc(out_dp)  # [batch_size, seq_len, num_tags]
        
    
        if tags is None:
            return probs, text_lens
        else:
            log_likelihood, self.trans_pars = tfa.text.crf_log_likelihood(inputs = probs,  # [batch_size, seq_len, num_tags] 
                                                                          tag_indices = tags,  # [batch_size, seq_len]
                                                                          sequence_lengths = text_lens,  # [batch_size]
                                                                          transition_params = self.trans_pars)  # [num_tags, num_tags]
        
            return probs, text_lens, log_likelihood  # log_likelihood: [batch_size]
        
        
#%% 
class BiLSTM_CRF(tf.keras.Model):
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_tags, embed_matrix):
        super(BiLSTM_CRF, self).__init__()
        
        self.embed = layers.Embedding(input_dim = vocab_size, output_dim = embed_dim, trainable=True,
                                      embeddings_initializer = tf.keras.initializers.Constant(embed_matrix))
        
        self.dropout = layers.Dropout(rate = 0.5)
        self.bilstm = layers.Bidirectional(layers.LSTM(units = hidden_dim, return_sequences=True, return_state=False))
        self.fc = layers.Dense(units = num_tags)
        
        self.trans_pars = tf.Variable(tf.random.uniform(shape = (num_tags, num_tags)))

    def call(self, text, tags=None, training=None):
        '''
            text: [batch_size, seq_len]
            tags: [batch_size, seq_len]
        '''
        
        non_zeros = tf.cast(tf.math.not_equal(text, 0), dtype=tf.int32)  # [batch_size, seq_len]
        text_lens = tf.math.reduce_sum(non_zeros, axis=1)  # [batch_size]
        
        out_embed = self.embed(text)  # [batch_sizem seq_len, embed_dim]
        out_dp = self.dropout(out_embed, training)  # [batch_sizem seq_len, embed_dim]
        out_lstm = self.bilstm(out_dp)  # [batch_size, seq_len, hidden_dim*2]
        probs = self.fc(out_lstm)  # [batch_size, seq_len, num_tags]
        
    
        if tags is None:
            return probs, text_lens
        else:
            log_likelihood, self.trans_pars = tfa.text.crf_log_likelihood(inputs = probs,  # [batch_size, seq_len, num_tags] 
                                                                          tag_indices = tags,  # [batch_size, seq_len]
                                                                          sequence_lengths = text_lens,  # [batch_size]
                                                                          transition_params = self.trans_pars)  # [num_tags, num_tags]
        
            return probs, text_lens, log_likelihood  # log_likelihood: [batch_size]

