#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Get-Topic-Model" data-toc-modified-id="Get-Topic-Model-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Get Topic Model</a></span></li><li><span><a href="#Main" data-toc-modified-id="Main-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Main</a></span></li></ul></div>


import sys


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import matplotlib.colors as colors
#from tqdm import tqdm
import time
import timeit
from tqdm import tqdm
import math
import random
import os
import sys

import io
import pickle
import unidecode
from itertools import chain
from collections import defaultdict

from numpy import linalg as LA


import pyLDAvis.gensim_models
#pyLDAvis.enable_notebook()# Visualise inside a notebook
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore
from gensim.models import CoherenceModel

import gc


# # Get Topic Model
# 

# In[2]:


class dataset:
    def __init__(self,field = 'APS_big',minyr = ''):
        self.field = field
        self.read_dict('doc_phrases_dict')
        
        
    def read_dict(self,name):
        if not hasattr(self,name):
            if os.path.isfile('../bin/Fields/'+self.field+'/'+name+'.pickle'):
                print('Reading In ' + name)
                with open('../bin/Fields/'+self.field+'/'+name+'.pickle', 'rb') as f:
                    dummy = pickle.load(f)
                setattr(self,name,dummy)

            else:
                print(name + ' not found in files. Nothing Read')
        else:
            print(name+' already read in')
            
    def write_dict(self,dict_,name):
        print('Writing ' + name + ' to File')
        with open('../bin/Fields/'+self.field + '/' +name+'.pickle', 'wb') as f:
            pickle.dump(dict_,f)
        print('Finished Writing')
        


# In[3]:


def prepare_corpus(cls,word_thresh = 300):
    dp_keys = list(cls.doc_phrases_dict.keys())
    #[:5000]
    print(len(dp_keys))
    doc_phrases_dict = {key:cls.doc_phrases_dict[key] for key in dp_keys}

    indices = []
    null_docs = []
    midnull_docs = []
    doc_keys = list(doc_phrases_dict.keys())
    for i,(doc,phrases) in enumerate(doc_phrases_dict.items()):
        if phrases == []:
            null_docs.append(doc)
        elif len(phrases) <= 10:
            null_docs.append(doc)       
    for doc in null_docs:
        del doc_phrases_dict[doc]


    cls.doc_phrases_ = list(doc_phrases_dict.values())
    cls.doc_phrases_dict = doc_phrases_dict

    cls.read_dict('corpus')
    #cls.read_dict('dictionary')
    if not hasattr(cls,'corpus'):
        start = timeit.default_timer()
        print("Generating Dictionary and Corpus for Topic Modeling")
        cls.dictionary = Dictionary(cls.doc_phrases_)
        cls.dictionary.filter_extremes(no_below=word_thresh)
        cls.corpus = [cls.dictionary.doc2bow(doc) for doc in cls.doc_phrases_]
        cls.write_dict(cls.corpus,'corpus')
        #dic = corpora.Dictionary(corpus)
        cls.dictionary.save('../bin/Fields/'+cls.field + '/dictionary')
        #cls.write_dict(cls.dictionary,'dictionary')
        stop = timeit.default_timer()
        print(f'Time for Generating Corpus and Dictionary: {round(stop-start,4)}\n\n')
    else:
        cls.dictionary = Dictionary.load('../bin/Fields/'+cls.field + '/dictionary')
        
    return cls

def topic_model(cls):
    cls.read_dict('umass_dict')
    cls.read_dict('cv_dict')
    cls.read_dict('beta_matrix_dict')
    if not hasattr(cls,'umass_dict'):
        cls.cv_dict = defaultdict(list)
        cls.umass_dict = defaultdict(list)
        cls.beta_matrix_dict = defaultdict(list)

    set_num_topics = np.linspace(20,100,17)

    print('Generating Topic Models and getting coherence scores for given Number of Topics\n')
    for num_topics in set_num_topics:
        if (num_topics not in cls.cv_dict.keys()) or (len(cls.cv_dict[num_topics]) != 5):
            print(f'Num Topics: {num_topics}')
            for i in range(5):
                if i >= len(cls.cv_dict[num_topics]):
                    print("Ensemble Seed: "+str(i))
                    start = timeit.default_timer()
                    lda_model = LdaMulticore(corpus=cls.corpus, id2word=cls.dictionary, iterations=50, num_topics=num_topics, workers = 10, passes=10,random_state=i)
                    c_v = CoherenceModel(model=lda_model, texts = cls.doc_phrases_, corpus=cls.corpus, dictionary=cls.dictionary, coherence='c_v')
                    umass = CoherenceModel(model=lda_model, texts = cls.doc_phrases_, corpus=cls.corpus, dictionary=cls.dictionary, coherence='u_mass')

                    cls.cv_dict[num_topics] += [c_v.get_coherence()]
                    cls.umass_dict[num_topics] += [umass.get_coherence()]
                    cls.beta_matrix_dict[num_topics] += [lda_model.get_topics()]
                    cls.lda_model = lda_model
                    del lda_model
                    del c_v
                    del umass

                    cls.write_dict(cls.umass_dict,'umass_dict')
                    cls.write_dict(cls.cv_dict,'cv_dict')
                    cls.write_dict(cls.beta_matrix_dict,'beta_matrix_dict')

                    stop = timeit.default_timer()
                    print(f'Time for Generating: {round((stop-start)/60,4)} min\n')
                    print('#####################################################\n\n')
                    break
            break
            
def topic_model_k(self,k,rng):
    print('Generating Topic Models and getting coherence scores for given Number of Topics\n')
    print("Ensemble Seed: "+str(rng))
    self.num_topics = k
    start = timeit.default_timer()
    lda_model = LdaMulticore(corpus=self.corpus, id2word=self.dictionary, iterations=50, num_topics=k, workers = 10, passes=10,random_state=rng)
    c_v = CoherenceModel(model=lda_model, texts = self.doc_phrases_, corpus=self.corpus, dictionary=self.dictionary, coherence='c_v')
    umass = CoherenceModel(model=lda_model, texts = self.doc_phrases_, corpus=self.corpus, dictionary=self.dictionary, coherence='u_mass')


    self.lda_model = lda_model
    stop = timeit.default_timer()
    print(f'Time for Generating: {round((stop-start)/60,4)} min\n')
    print('#####################################################\n\n')
    
    return self

def get_embeddings(self):
    self.doc_embeddings = defaultdict(list)
    i = 0

    for doc in tqdm(self.doc_phrases_dict.keys()):
        embed = self.lda_model.get_document_topics(self.corpus[i])

        dummy = [0 for k in range(self.num_topics)]
        for j,top in enumerate(list(zip(*embed))[0]):
            dummy[top] = embed[j][1]

        dummy = dummy/np.sum(dummy)
        self.doc_embeddings[doc] = dummy
        i+=1
    return self

    
        
        


# # Main

# In[ ]:


if __name__ == '__main__':
    # Test w APS_big
    print('BEGINNING LDA: FINDING COHERENCE SCORE')
    print('sys.argv[1] = dataset = '+sys.argv[1])
    start = timeit.default_timer()
    aps = dataset(sys.argv[1])
    aps = prepare_corpus(aps)
    stop = timeit.default_timer()
    print(f'Time for Reading Files & Generating Corpus & Dictionary: {round((stop-start)/60,4)} min\n')
    topic_model(aps)
    sys.exit(0)


# In[ ]:




