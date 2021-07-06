#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 19:18:10 2021

@author: qwang
"""

import pubmed_parser
from pred_fn import sent_detect, pico_extract, tup2dict

import streamlit as st
# streamlit run /home/qwang/pre-pico/app/stream.py

#%%
sent_pth = '/home/qwang/pre-pico/app/sent_abs.pth.tar'
pth_path = '/home/qwang/pre-pico/app/full.pth.tar' 

tag2idx = {'O': 0, 
		   'B-Species': 1, 
		   'I-Species': 2, 
		   'B-Strain': 3, 
		   'I-Strain': 4, 
		   'B-Induction': 5, 
		   'I-Induction': 6, 
		   'B-Intervention': 7, 
		   'I-Intervention': 8, 
		   'B-Comparator': 9, 
		   'I-Comparator': 10, 
		   'B-Outcome': 11, 
		   'I-Outcome': 12}
idx2tag = {idx: tag for tag, idx in tag2idx.items()}

#%% App
st.header('PICO extraction for in vivo abstract')
# pmid = st.text_input('Input one PMID: ', 23326526)
pmid = st.text_input('Input one PMID: ', 27231887)


try:
    xml = pubmed_parser.parse_xml_web(pmid)
    title = xml['title']
    text = xml['abstract']
    if text == "":
        st.text("No abstract available!")
    else:      
        # st.write(title)
        # st.write(text)
        ## Extract pico text   
        text = sent_detect(text, sent_pth)
        # Extract pico phrases              
        tup, _, _ = pico_extract(text, pth_path, idx2tag)
        res = tup2dict(tup)
        
        st.write("""### Extracted PICO text  ###
                 """) 
        st.write(text)
        
        keys = [k for k, v in res.items()]
        
        st.write("""### Extracted PICO phrases  ###
                 """) 
        if 'Species' in keys:
            st.write("""Species: """, res['Species'])
        if 'Strain' in keys:
            st.write("""Strain: """, res['Strain'])
        if 'Induction' in keys:
            st.write("""Induction: """, res['Induction'])
        if 'Intervention' in keys:
            st.write("""Intervention: """, res['Intervention'])
        if 'Comparator' in keys:
            st.write("""Comparator: """, res['Comparator'])
        if 'Outcome' in keys:
            st.write("""Outcome: """, res['Outcome'])
        
except:
    st.text("PMID not applicable!") 