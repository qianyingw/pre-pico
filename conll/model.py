#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:25:23 2020

@author: qwang
"""

import tensorflow as tf
from tensorflow.keras import layers

#%%
class NerBiLSTM(tf.keras.Model):
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, embed_matrix):
        super(NerBiLSTM, self).__init__()
        
        self.embed = layers.Embedding(input_dim = vocab_size, output_dim = embed_dim, trainable=True,
                                      embeddings_initializer=tf.keras.initializers.Constant(embed_matrix))                                      
        
        self.bilstm = layers.Bidirectional(layers.LSTM(units = hidden_dim, return_sequences=True, return_state=False))  
        self.fc = layers.Dense(units = output_dim)
        # self.fc = layers.dense(units = output_dim, activation='relu')
        # self.softmax = layers.Activation('softmax')   
    
    def call(self, text):
        '''
            text: [batch_size, seq_len]
        '''
        out_embed = self.embed(text)  # [batch_size, seq_len, embed_dim]
        out_lstm = self.bilstm(out_embed)  # [batch_size, seq_len, hidden_dim*2]
        probs = self.fc(out_lstm)  # [batch_size, seq_len, output_dim]
        # probs = self.softmax(probs)  # [batch_size, seq_len, output_dim]
        
        return probs