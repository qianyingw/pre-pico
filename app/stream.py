#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 19:18:10 2021

@author: qwang
"""

import json
import pubmed_parser
import pred_fn
from pred_fn import sent_detect, pred_one_bert, tup2dict

import streamlit as st
# streamlit run /home/qwang/pre-pico/app/stream.py

#%%
mod = 'bert'
pre_wgts = 'biobert'

sent_pth = '/home/qwang/pre-pico/app/sent_abs.pth.tar'
prfs_path = '/home/qwang/pre-pico/app/b0_bio.json'
pth_path = '/home/qwang/pre-pico/app/b0_bio_last.pth.tar' 

#%% App
st.header('PICO recognition for in vivo abstract')
pmid = st.text_input('Input one PMID: ', 23326526)


try:
    xml = pubmed_parser.parse_xml_web(pmid)
    title = xml['title']
    text = xml['abstract']
    if text == "":
        st.text("No abstract available!")
    else:      
        st.write(title)
        st.write(text)
        
        ## Extract pico text   
        text = sent_detect(text, sent_pth)
        # Load idx2tag
        with open(prfs_path) as f:
            dat = json.load(f)    
        idx2tag = dat['idx2tag']
        # Extract pico phrases              
        tup = pred_one_bert(text, mod, pre_wgts, pth_path, idx2tag)
        res = tup2dict(tup)
        
        st.write("""### PICO:  ###
                 """) 
        if res['Species']:
            st.write("""Species: """, res['Species'])
        if res['Strain']:
            st.write("""Strain: """, res['Strain'])
        if res['Induction']:
            st.write("""Induction: """, res['Induction'])
        if res['Intervention']:
            st.write("""Intervention: """, res['Intervention'])
        if res['Comparator']:
            st.write("""Comparator: """, res['Comparator'])
        if res['Outcome']:
            st.write("""Outcome: """, res['Outcome'])
        
except:
    st.text("PMID not applicable!") 