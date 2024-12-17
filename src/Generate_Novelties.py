#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib.colors as colors
import timeit
import math
import random
#from scipy.optimize import fsolve
from mpmath import *
from datetime import datetime, timedelta
from scipy.optimize import curve_fit, fsolve
from scipy.stats import binned_statistic as bs
import scipy.stats as scs


import re
import csv 
from time import sleep
from tqdm import tqdm
import time
from ast import literal_eval
import seaborn as sns

import os
import sys
import io
import pickle
import unidecode
from itertools import chain
from collections import defaultdict
import itertools

import gensim_LDA_script as lda

import pyLDAvis.gensim_models
pyLDAvis.enable_notebook()# Visualise inside a notebook
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore
from gensim.models import CoherenceModel



############################### YEARLY AUTHOR PUBS AND CITS ##########################################

def get_multiple_pubs_cits(self,noss = False,ext = False):
    
    if noss == False:
        if hasattr(self,'auid_yr_pubs_cits'):
            print('Object already has auid_yr_pubs_cits')
            return self
        else:
            self.read_dict('auid_yr_pubs_cits')
            if hasattr(self,'auid_yr_pubs_cits'):
                return self
    else:
        if hasattr(self,'auid_yr_pubs_cits_noss'):
            print('Object already has auid_yr_pubs_cits_noss')
            return self
        else:
            self.read_dict('auid_yr_pubs_cits_noss')
            if hasattr(self,'auid_yr_pubs_cits_noss'):
                return self
            
    print('Generating auid_yr_pubs_cits dictionary')
    dummy = defaultdict(dict)
    for auid in tqdm(list(self.authorid_docs_dict.keys()),miniters = 1000,mininterval = .5):
        dummy[auid] = get_pubs_cits_auid(self,auid,noss)
        dummy[auid]['t_0'] = list(np.arange(1,len(list(dummy[auid].keys())),1))
    print('DONE')
    if noss == True:
        self.auid_yr_pubs_cits_noss = dummy
        self.write_dict(dummy,'auid_yr_pubs_cits_noss',ext)
    else:
        self.auid_yr_pubs_cits = dummy
        self.write_dict(dummy,'auid_yr_pubs_cits',ext)

    return self


def get_pubs_cits_auid(self,auid,noss = False):
    if noss == False:
        dummy_pubs = self.authorid_docs_dict[auid]
    else:
        dummy_pubs = [doc for doc in self.authorid_docs_dict[auid] if self.is_ss_paper[doc]==False]

    if dummy_pubs == []:
        #self.auid_yr_pubs_cits[auid] = defaultdict(list)
        return defaultdict(list)
    years_ = [self.doc_year_dict[x] for x in dummy_pubs]
    years_ = [x for x in years_ if x >= 1919]
    dummy_dict = defaultdict(list)
    if len(years_) == 0:
        dummy_dict[0] = [float('nan'),float('nan'),float('nan'),float('nan')]
        return dummy_dict
    minyr = min(years_)
    prev_pub = []
    total_cits = list(chain.from_iterable([self.citations_dict[p] for p in dummy_pubs]))
    for year in np.arange(minyr,2025,1):
        delP = [p for p in dummy_pubs if self.doc_year_dict[p] == year]
        prev_pub += delP
        delCit = [p for p in total_cits if self.doc_year_dict[p] == year]
        total_cits = [x for x in total_cits if x not in delCit]
        dummy_dict[year].append(len(delP))
        dummy_dict[year].append(len(delCit))
        if year == minyr:
            dummy_dict[year].append(len(delP))
            dummy_dict[year].append(len(delCit))
        else:
            #print(dummy_dict[year-1])
            dummy_dict[year].append(len(delP) + dummy_dict[year-1][2])
            dummy_dict[year].append(len(delCit) + dummy_dict[year-1][3])

    return dummy_dict


############################### DISRUPTION INDEX ##########################################

def CD_compute(self,refs,cits,cits_of_refs,refs_of_cits_dict):
    n_t = len(cits_of_refs)
    if n_t == 0:
        return float('nan')
    f_it = [1 if ppr in cits else 0 for ppr in cits_of_refs]
    refset = set(refs)
    #not set(a).isdisjoint(b)
    b_it = [1 if not set(refs_of_cits_dict[ppr]).isdisjoint(refset) else 0 for ppr in cits_of_refs]
    #b_it = [1 if len(set(refs_of_cits_dict[ppr]).intersection(set(refs))) > 0 else 0 for ppr in cits_of_refs]

    CDt = 1/n_t * np.sum([-2*f_it[j]*b_it[j] + f_it[j] for j in range(len(cits_of_refs))])
    return CDt

def CD(self, paper_id,t=10):
    # t - Number of years after publication of doi
    # n_t - Number of Forward Citations (Cite doi in and doi's references)
    # f_it - 1 if paper cites focal patent doi, 0 otherwise
    # b_it - 1 if paper cites a reference of focal patent doi, 0 otherwise

    yr = self.doc_year_dict[paper_id] + t

    if len(self.citations_dict[paper_id]) == 0:
        return float('nan')

    cits = [x for x in self.citations_dict[paper_id] if ((self.doc_year_dict[x] <= yr) and (self.doc_year_dict[x] > yr - t))]
    if len(cits) == 0:
        return float('nan')

    i = list(chain.from_iterable([self.citations_dict[ref] for ref in self.references_dict[paper_id]]))
    i = [x for x in i if ((self.doc_year_dict[x] <= yr) and (self.doc_year_dict[x] > yr - t))]
    i = list(set(i + cits))
    #i = [ppr for ppr in i if ((self.doc_year_dict[ppr] <= yr) and (self.doc_year_dict[ppr] > yr - t))]
    #i = [ppr for ppr in i if self.doc_year_dict[ppr] > yr - t]
    cit_dois_within_t_years = i

    return CD_compute(self,self.references_dict[paper_id],self.citations_dict[paper_id],cit_dois_within_t_years,self.references_dict)

def get_all_disruption_index(self,t=10,ext = False):

    if hasattr(self,'doc_disrupt'):
        print('Object already has Disruption Index')
        return 0
    elif ('Disruption Index' in self.DDBB.columns):
        print('Disruption Index already in DDBB. Generating Doi-Novelty Dictionary')
        ids = self.docs
        disrupt = self.DDBB['Disruption Index'].to_list()
        self.doc_disrupt = defaultdict(float)
        for i,pap in enumerate(ids):
            self.doc_disrupt[doi] = disrupt[i]
        return 0

    self.doc_disrupt = defaultdict(self.create_nan_defaultdict)
    print('Getting All Disruption Indices')
    #for doc in tqdm(list(self.citations_dict.keys()),desc = 'Computing Disruption Indices...',miniters = 1000,mininterval = .5):
    for doc in tqdm(list(self.docs),desc = 'Computing Disruption Indices...',miniters = 1000,mininterval = .5):
        self.doc_disrupt[doc] = CD(self,doc,t)
    print('Finished Generating All Disruption Indices')
    self.write_dict(self.doc_disrupt,'doc_disrupt',ext)
    
    return self


##############################################################################################################

################## INNOVATION ##########################################################        

def generate_phrase_docs_dict(self):
    print('Generating phrase_docs_dict')
    self.phrase_docs_dict = defaultdict(list)
    for doc in tqdm(self.docs):
        phrases = self.doc_phrases_dict[doc]
        for phr in set(phrases):
            self.phrase_docs_dict[phr].append(doc)
'''
def read_in_phrase_link_params(self):
    if not hasattr(self,'phrase_docs_dict'):
        self.generate_phrase_docs_dict(s)

    if os.path.isfile('../bin/Fields/'+self.field+'/PMI_dict.pickle'):
        print('Reading PMI Dict')
        with open('../bin/Fields/'+self.field+'/words_completed.pickle', 'rb') as f:
            self.words_completed = pickle.load(f)
        with open('../bin/Fields/'+self.field+'/PMI_dict.pickle', 'rb') as f:
            self.PMI_dict = pickle.load(f)
    else:
        self.words_completed = 0
        self.PMI_dict = dict()
'''

def get_all_innov_dicts(self,ext=False):
    self = generate_distal_novelty_dicts(self,PMI = False)
    self = generate_distal_novelty_dicts(self,PMI = True)
    self.write_dict(self.distal_novelty_dict,'distal_novelty_dict',ext)
    self.write_dict(self.impact_distal_novelty_dict,'impact_distal_novelty_dict',ext)
        
    self.phrase_links = list(self.distal_novelty_dict.keys()) 
    self = get_all_innov(self)
    
    return self


def generate_distal_novelty_dicts(self,PMI):
    # Makes Dictionary of Phrases 
    # Each dictionary value is a dictionary of 'dois' which are paper ids and 'year' which gives publication year
    # self.distal_novelty_dict contains the appearances of these phrases during their first year being seen
    # self.doc_impactinnov contains the apperances of these phrases after their first year of being seen
    #PMI = True implies the phrases links are phrases that appear in the same document, but are not necessarily next to each other
    #PMI = False runs the algorithm on the phrases from the PLDA
    if hasattr(self,'distal_novelty_dict'):
        print('Novelty Dict already Computed')
        return self
    if not hasattr(self,'distal_novelty_dict'):
        self.read_dict('distal_novelty_dict')
        if hasattr(self,'distal_novelty_dict'):
            print('Novelty Dict already Computed')
            self.read_dict('impact_distal_novelty_dict')
            return self
    if not hasattr(self,'phrase_docs_dict'):
        print('Generating phrase_docs_dict')
        self.phrase_docs_dict = defaultdict(list)
        if not hasattr(self,'doc_phrases_dict'):
            self.read_dict('doc_phrases_dict')
        #for doc,phrases in tqdm(self.doc_phrases_dict.items()):
        for doc in tqdm(self.docs):
            for phr in set(self.doc_phrases_dict[doc]):
                self.phrase_docs_dict[phr].append(doc)
        self.all_multiword_phrases = []
        for phr in tqdm(self.phrase_docs_dict.keys()):
            if ' ' in phr:
                self.all_multiword_phrases.append(phr)
        if not hasattr(self,'PMI_dict'):
            self.read_dict('PMI_dict')

    print('Making PMI_df for computing')
    PMI_df = pd.DataFrame.from_dict(data = self.PMI_dict, orient = 'index').sort_values(by = 0)
    PMI_df = PMI_df[PMI_df[0] > 2]
    if not hasattr(self,'distal_novelty_dict'):
        self.distal_novelty_dict = dict()
    if not hasattr(self, 'impact_distal_novelty_dict'):
        self.impact_distal_novelty_dict = dict()

    if PMI == True:
        dum_phrases = PMI_df.index.to_list()
    else:
        dum_phrases = self.all_multiword_phrases

    start_phrase = len(self.distal_novelty_dict)
    print('Generating Phrase-Novelty Dicts: PMI =',PMI)
    #i = 0
    for phr in tqdm(dum_phrases):
        minyr = 0
        if PMI == True:
            phr1 = phr.split(',')[0]
            phr2 = phr.split(',')[1]
            docs = list(set(self.phrase_docs_dict[phr1]).intersection(set(self.phrase_docs_dict[phr2])))
            years = [self.doc_year_dict[doc] for doc in docs if self.doc_year_dict[doc] != 0]
            if len(years) != 0:
                minyr = min(years)

        else:    
            docs = self.phrase_docs_dict[phr]
            years = [self.doc_year_dict[doc] for doc in docs if self.doc_year_dict[doc] != 0]
            if len(years) != 0:
                minyr = min(years)
        if (minyr >= 1970):
            dummy2 = [doc for doc in docs if self.doc_year_dict[doc] == minyr]
            dummy3 = [doc for doc in docs if self.doc_year_dict[doc] > minyr]
            yr3 = [self.doc_year_dict[doc] for doc in dummy3]
            self.distal_novelty_dict[phr] = {'docs' : dummy2, 'year' : minyr}
            if len(dummy3) == 0:
                self.impact_distal_novelty_dict[phr] = {'docs' : [], 'year' : -1}
            else:
                self.impact_distal_novelty_dict[phr] = {'docs' : dummy3, 'year' : yr3}
    print('Finished w/ Phrase-Novelty Dicts')
    self.phrase_links = list(self.distal_novelty_dict.keys())
    
    return self
    


def get_all_innov(self,ext = False):

    #print('Writing all Distances between References for all Documents to a dictionary "doc_ref_dists"')

    if hasattr(self,'doc_innov'):
        print('Object already has Innovation')
        return 0
    if not hasattr(self,'PMI_dict'):
        print('No PMI_dict read in')
        self.read_dict('PMI_dict')
        if not hasattr(self,'PMI_dict'):
            print('No Pointwise Mutual Information Dict found. Compute before finding Innovation Scores')
            print('Loading Dictionary with NaNs')
            for doc in tqdm(list(set(self.docs)),miniters = 1000,mininterval = .5):
                self.doc_innov[doc] = float('nan')
                self.doc_impactinnov[doc] = float('nan')

    #docs_completed = len(list(self.doc_ref_dists.keys()))  
    start = timeit.default_timer()
    print('Generating doc_innov & doc_impactinnov_dict')
    self.doc_innov = defaultdict(int)

    docs = list(chain.from_iterable([self.distal_novelty_dict[phr]['docs'] for phr in self.distal_novelty_dict.keys()]))
    for doc in tqdm(list(set(docs)),miniters = 1000,mininterval = .5):
        self.doc_innov[doc] = docs.count(doc)

    self.write_dict(self.doc_innov,'doc_innov',ext)

    self.doc_impactinnov = defaultdict(int)
    for phr in tqdm(self.impact_distal_novelty_dict.keys(),miniters = 1000,mininterval = .5):
        for doc in self.distal_novelty_dict[phr]['docs']:
            self.doc_impactinnov[doc] += len(self.impact_distal_novelty_dict[phr]['docs'])


    self.write_dict(self.doc_impactinnov,'doc_impactinnov',ext)
    return self


############################### ENTROPY #####################################  

def doc_shan(self,doc):
    return shan(self,self.doc_embeddings_dict[doc])

def shan(self,vec):
    vec = [x for x in vec if x != 0]
    return (-1)*np.sum(np.multiply(vec,np.log2(vec)))

def get_all_shannon_entropy(self):

    '''
    WRITING TO FILE TAKES MUCH LONGER THAN COMUPTATION. DO NOT WRITE
    '''
    #print('Writing all Distances between References for all Documents to a dictionary "doc_ref_dists"')
    if hasattr(self,'doc_shanent'):
        print('Object already has Shannon Entropy')
        return self
    

    self.doc_shanent =  defaultdict(self.create_nan_defaultdict)
    print('Getting All Shannon Entropy')
    for doc in tqdm(self.doc_embeddings_dict.keys(),desc = 'Shannon Entropy...',miniters = 1000,mininterval = .5):
        self.doc_shanent[doc] = doc_shan(self,doc)
    print('Finished Generating All LDA Shannon Entropy')
    
    return self

############################### CONEPT ENTROPY #####################################  
def generate_concept_embeddings(self):
    self.doc_concept_embeddings = defaultdict(list)
    all_concepts = []
    level = 2
    for doc,concepts in tqdm(self.doc_concepts_dict.items(),total = len(self.doc_concepts_dict), desc = 'Getting Concept Embeddings'):
        for con in concepts:
            if con['level'] == level:
                all_concepts.append(con['concept'])
                self.doc_concept_embeddings[doc].append(con['score'])
        if len(self.doc_concept_embeddings[doc])!=0:
            self.doc_concept_embeddings[doc] = np.array(self.doc_concept_embeddings[doc])/np.sum(self.doc_concept_embeddings[doc])
    return self

def doc_shan_concepts(self,doc):
    return shan(self,self.doc_concept_embeddings[doc])

def shan(self,vec):
    vec = [x for x in vec if x != 0]
    return (-1)*np.sum(np.multiply(vec,np.log2(vec)))

def get_all_shannon_entropy_concepts(self):

    '''
    WRITING TO FILE TAKES MUCH LONGER THAN COMUPTATION. DO NOT WRITE
    '''
    if hasattr(self,'doc_shanent'):
        print('Object already has Shannon Entropy')
        return self
    if not hasattr(self,'doc_concept_embeddings'):
        self = generate_concept_embeddings(self)

    self.doc_shanent =  defaultdict(self.create_nan_defaultdict)
    print('Getting All Shannon Entropy')
    for doc in tqdm(self.doc_concept_embeddings.keys(),desc = 'Shannon Entropy...',miniters = 1000,mininterval = .5):
        self.doc_shanent[doc] = doc_shan_concepts(self,doc)
    print('Finished Generating All Concept Shannon Entropy')
    
    return self

################################################

def get_shannon_entropy_dict(self,concepts = True, word_thresh = 50,num_topics = 20,seed_ = 1,ext = False):
    if concepts == True:
        self = get_all_shannon_entropy_concepts(self)
    else:
        start = timeit.default_timer()
        test_lda = lda.dataset(self.field)
        test_lda = lda.prepare_corpus(test_lda,word_thresh)
        num_t = num_topics
        random_number_seed = seed_
        test_lda = lda.topic_model_k(test_lda,num_t,random_number_seed)
        test_lda = lda.get_embeddings(test_lda)
        test_lda.write_dict(test_lda.doc_embeddings,'doc_embeddings_dict',ext)
        stop = timeit.default_timer()
        print(stop-start,'(s)')
    return self
        
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################



############################### READ AND WRITE ##########################################        


def read_embeddings(self):
    if not hasattr(self,'embeddings'):
        if os.path.isfile('../bin/Fields/'+self.field+'/doc_embeddings_dict.pickle'):
            print(f'Reading In Embeddings for {self.field}')
            with open('../bin/Fields/'+self.field+'/doc_embeddings_dict.pickle', 'rb') as f:
                dummy = pickle.load(f)
            setattr(self,'doc_embeddings_dict',dummy)
        elif os.path.isfile('../datasets/Embeddings/Dict_papers_embedings_'+self.field+'.pkl'):
            print(f'Reading In Embeddings for {self.field}')
            with open('../datasets/Embeddings/Dict_papers_embedings_'+self.field+'.pkl', 'rb') as f:
                dummy = pickle.load(f)
            setattr(self,'doc_embeddings_dict',dummy)

        else:
            print('Embeddings not found in files. Nothing Read')
    else:
        print('Embeddings already read in')


class Field_Analysis:
    def __init__(self,FD_obj,minyr = ''):
        self.field = FD_obj.field
        self.DDBB = FD_obj.DDBB
        self.doc_authorids_dict = FD_obj.doc_authorids_dict
        self.authorid_docs_dict = FD_obj.authorid_docs_dict
        self.doc_year_dict = FD_obj.doc_year_dict
        self.year_docs_dict = FD_obj.year_docs_dict
        self.references_dict = FD_obj.references_dict
        self.citations_dict = FD_obj.citations_dict
        self.doc_embeddings_dict = FD_obj.doc_embeddings_dict
        self.doc_phrases_dict = FD_obj.doc_phrases_dict
        
        #self.read_dict('doc_phrases_dict')
        #self.generate_h_index_df()
        self.get_superstars()
        
        self.read_dict('collabauth')
        self.read_dict('no_collabauth')
        self.read_dict('insp_auth')
        
        other_dicts = ['auid_yr_pubs_cits','auid_yr_pubs_cits_noss','doc_disrupt','doc_innov','doc_impactinnov',
                      'distal_novelty_dict','impact_distal_novelty_dict','collabauth','no_collabauth','insp_auth']
        for d in other_dicts:
            if hasattr(self,'doc_innov') and (d in ['distal_novelty_dict','impact_distal_novelty_dict']):
                print(d,'not read because doc-innovation dicts already computed')
                continue
            elif hasattr(FD_obj,d):
                print('Setting',d)
                setattr(self,d,getattr(FD_obj,d))
            else:
                print(d,'does not exist')
                
        self.get_proper_dois()
        self.generate_authorid_startyr_dict()
        
        self.get_all_shannon_entropy()
        #self.get_all_disruption_index()
        

        self.read_dict('authorid_novelty_stats')
        
        #self.generate_h_index_df()
        #self.get_superstars()
        
        #self.generate_paperstats_df()
        #self.get_authorstats_df()
    ###############################################################
            
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
        
        
    ###########################################################################################    
    ###########################################################################################    
    ########################### PREPARE DATA FOR ANALYSIS #####################################   
    ###########################################################################################    
    ###########################################################################################    
    ###########################################################################################
    
    def get_proper_dois(self):
        print('Removed all Papers with more than 25 authors')
        
        self.docs = list(self.doc_embeddings_dict.keys())
        docs = []
        for doc in self.docs:
            if len(self.doc_authorids_dict[doc]) < 25:
                docs += [doc]
                
        self.docs = docs
        self.DDBB = self.DDBB[self.DDBB['id'].isin(docs)]

    ########### Get Dictionary of Yearly Publication and Citation Statistics of Authorids ##############

    def get_multiple_pubs_cits(self,noss = False):
        if noss == False:
            if os.path.isfile('../bin/Fields/'+self.field+'/auid_yr_pubs_cits.pickle'):
                self.read_dict('auid_yr_pubs_cits')
                return 0
        else:
            if os.path.isfile('../bin/Fields/'+self.field+'/auid_yr_pubs_cits_noss.pickle'):
                self.read_dict('auid_yr_pubs_cits_noss')
                return 0
        print('Generating auid_yr_pubs_cits dictionary')
        dummy = defaultdict(dict)
        #self.auid_yr_pubs_cits = defaultdict()
        for auid in tqdm(list(self.authorid_docs_dict.keys()),miniters = 1000,mininterval = .5):
            dummy[auid] = self.get_pubs_cits_auid(auid,noss)
            dummy[auid]['t_0'] = list(np.arange(1,len(list(dummy[auid].keys())),1))
        print('DONE')
        if noss == True:
            self.auid_yr_pubs_cits_noss = dummy
            self.write_dict(dummy,'auid_yr_pubs_cits_noss')
        else:
            self.auid_yr_pubs_cits = dummy
            self.write_dict(dummy,'auid_yr_pubs_cits')


    def get_pubs_cits_auid(self,auid,noss = False):
        # auid : year : [delta_Pub,delta_cit,cum_pub,cum_cit]
        if noss == False:
            dummy_pubs = self.authorid_docs_dict[auid]
        else:
            dummy_pubs = [doc for doc in self.authorid_docs_dict[auid] if self.is_ss_paper[doc]==False]
            
        if dummy_pubs == []:
            #self.auid_yr_pubs_cits[auid] = defaultdict(list)
            return defaultdict(list)
        years_ = [self.doc_year_dict[x] for x in dummy_pubs]
        years_ = [x for x in years_ if x >= 1919]
        dummy_dict = defaultdict(list)
        if len(years_) == 0:
            dummy_dict[0] = [float('nan'),float('nan'),float('nan'),float('nan')]
            return dummy_dict
        minyr = min(years_)
        prev_pub = []
        total_cits = list(chain.from_iterable([self.citations_dict[p] for p in dummy_pubs]))
        for year in np.arange(minyr,2025,1):
            delP = [p for p in dummy_pubs if self.doc_year_dict[p] == year]
            prev_pub += delP
            delCit = [p for p in total_cits if self.doc_year_dict[p] == year]
            total_cits = [x for x in total_cits if x not in delCit]
            dummy_dict[year].append(len(delP))
            dummy_dict[year].append(len(delCit))
            if year == minyr:
                dummy_dict[year].append(len(delP))
                dummy_dict[year].append(len(delCit))
            else:
                #print(dummy_dict[year-1])
                dummy_dict[year].append(len(delP) + dummy_dict[year-1][2])
                dummy_dict[year].append(len(delCit) + dummy_dict[year-1][3])

        return dummy_dict
    '''
    def set_norm_DTC(self):
        start = timeit.default_timer()
        if hasattr(self,'normDTC'):
            if len(self.normDTC) < len(list(self.doc_embeddings_dict.values())):
                print('Normalization started but not yet completed')
                print(str(len(self.normDTC)) + ' out of ' + str(len(self.doc_embeddings_dict)) + ' remaining')
                self.normDTC = defaultdict(list)
            else:
                print('Normalization Already Completed')
                return self.normDTC
        else:
            print('Beginning l2 Normalizaiton of Document Embeddings') 
            self.normDTC = defaultdict(list)

        for doc in tqdm(self.docs_abs, desc = "Getting norm DTC"):
            self.normDTC[doc] = self.doc_embeddings_dict[doc]/ np.linalg.norm(self.doc_embeddings_dict[doc])
        stop = timeit.default_timer()
        print('Completed Normalization of DTC:',(stop-start),'(s)')
    '''    
    
    def generate_authorid_startyr_dict(self):
        beginyr = []
        print('Getting authorid_startyr_dict')
        if hasattr(self,'auid_yr_pubs_cits'):
            self.authorid_startyr_dict = defaultdict()
            for auth in tqdm(list(self.auid_yr_pubs_cits.keys())):
                dummy = list(self.auid_yr_pubs_cits[auth].keys())[0]
                if (dummy == 't_0') | (dummy == 0):
                    doop = float('nan')
                else:
                    doop = int(dummy)
                beginyr.append(doop)
                self.authorid_startyr_dict[auth] = doop
                
    def set_working_attributes(self):
        #self.generate_doc_authorid_year_dicts()
        #self.generate_reference_citation_dicts()
        self.get_multiple_pubs_cits()
        #self.set_norm_DTC()
        
        
    ###########################################################################################    
    ###########################################################################################    
    ############################### NOVELTY MEASURES ##########################################   
    ###########################################################################################    
    ###########################################################################################    
    ###########################################################################################
    
    def create_nan_defaultdict(self):
        return float('nan')
    
    
    ############################### DISRUPTION INDEX ##########################################

    def CD_compute(self,refs,cits,cits_of_refs,refs_of_cits_dict):
        n_t = len(cits_of_refs)
        if n_t == 0:
            return float('nan')
        if len(refs) == 0:
            return float('nan')
    
        f_it = [1 if ppr in cits else 0 for ppr in cits_of_refs]
        refset = set(refs)
        #not set(a).isdisjoint(b)
        b_it = [1 if not set(refs_of_cits_dict[ppr]).isdisjoint(refset) else 0 for ppr in cits_of_refs]
        #b_it = [1 if len(set(refs_of_cits_dict[ppr]).intersection(set(refs))) > 0 else 0 for ppr in cits_of_refs]

        CDt = 1/n_t * np.sum([-2*f_it[j]*b_it[j] + f_it[j] for j in range(len(cits_of_refs))])
        return CDt

    def CD(self, paper_id,t=10):
        # t - Number of years after publication of doi
        # n_t - Number of Forward Citations (Cite doi in and doi's references)
        # f_it - 1 if paper cites focal patent doi, 0 otherwise
        # b_it - 1 if paper cites a reference of focal patent doi, 0 otherwise

        yr = self.doc_year_dict[paper_id] + t

        if len(self.citations_dict[paper_id]) == 0:
            return float('nan')

        cits = [x for x in self.citations_dict[paper_id] if ((self.doc_year_dict[x] <= yr) and (self.doc_year_dict[x] > yr - t))]
        if len(cits) == 0:
            return float('nan')
        if len(self.references_dict[paper_id]) == 0:
            return float('nan')

        i = list(chain.from_iterable([self.citations_dict[ref] for ref in self.references_dict[paper_id]]))
        i = [x for x in i if ((self.doc_year_dict[x] <= yr) and (self.doc_year_dict[x] > yr - t))]
        i = list(set(i + cits))
        #i = [ppr for ppr in i if ((self.doc_year_dict[ppr] <= yr) and (self.doc_year_dict[ppr] > yr - t))]
        #i = [ppr for ppr in i if self.doc_year_dict[ppr] > yr - t]
        cit_dois_within_t_years = i

        return self.CD_compute(self.references_dict[paper_id],self.citations_dict[paper_id],cit_dois_within_t_years,self.references_dict)

    def get_all_disruption_index(self,t=10,ext = False):

        if hasattr(self,'doc_disrupt'):
            print('Object already has Disruption Index')
            return 0
        elif ('Disruption Index' in self.DDBB.columns):
            print('Disruption Index already in DDBB. Generating Doi-Novelty Dictionary')
            ids = self.docs
            disrupt = self.DDBB['Disruption Index'].to_list()
            self.doc_disrupt = defaultdict(float)
            for i,pap in enumerate(ids):
                self.doc_disrupt[doi] = disrupt[i]
            return 0

        self.doc_disrupt = defaultdict(self.create_nan_defaultdict)
        print('Getting All Disruption Indices')
        #for doc in tqdm(list(self.citations_dict.keys()),desc = 'Computing Disruption Indices...',miniters = 1000,mininterval = .5):
        for doc in tqdm(list(self.docs),desc = 'Computing Disruption Indices...',miniters = 1000,mininterval = .5):
            self.doc_disrupt[doc] = self.CD(doc,t)
        print('Finished Generating All Disruption Indices')
        self.write_dict(self.doc_disrupt,'doc_disrupt',ext)
        
############################### INFORMATION DIVERSITY #####################################        
    '''    
    def reference_diversity(self,doi):
        #print(doi)
        refs = self.references_abs_dict[doi]
        vecs = [self.normDTC[doi] for doi in refs if self.normDTC[doi] != []]
        if len(vecs) <= 1:
            return float('nan')
        M = np.mean(vecs,axis = 0)
        return np.sum([(1-np.dot(d,M))**2 for d in vecs])/len(vecs)
    
    def get_all_reference_diversity(self):
        
        if hasattr(self,'doc_refdiv'):
            print('Object already has Reference Diversity')
            return 0
        elif 'Reference Diversity' in self.DDBB.columns:
            print('Reference Diversity already in DDBB. Generating Doi-Novelty Dictionary')
            dois = self.DDBB['doi'].to_list()
            infdiv = self.DDBB['Information Diversity'].to_list()
            self.doc_refdiv = defaultdict(float)
            for i,doi in enumerate(dois):
                self.doc_refdiv[doi] = infdiv[i]
            return 0

        if not hasattr(self,'normDTC'):
            self.set_norm_DTC()

        self.doc_refdiv =  defaultdict(self.create_nan_defaultdict)
        print('Getting All Information Diversity')
        for doi in tqdm(list(self.references_abs_dict.keys()),desc = 'Reference Diversity...',miniters = 1000,mininterval = .5):
            self.doc_refdiv[doi] = self.reference_diversity(doi)
        print('Finished Generating All Reference Diversity')
        #self.write_dict(self.doc_refdiv,'doc_refdiv')
    
############################### CITATION DIVERSITY ##################################### 
        
    def citation_diversity(self,doi):
        #print(doi)
        refs = self.citations_abs_dict[doi]
        vecs = [self.normDTC[doi] for doi in refs if self.normDTC[doi] != []]
        if len(vecs) <= 1:
            return float('nan')
        M = np.mean(vecs,axis = 0)
        return np.sum([(1-np.dot(d,M))**2 for d in vecs])/len(vecs)
    
    def get_all_citation_diversity(self):

        #print('Writing all Distances between References for all Documents to a dictionary "doc_ref_dists"')
        if hasattr(self,'doc_citdiv'):
            print('Object already has Citation Diversity')
            return 0
        elif 'Citation Diversity' in self.DDBB.columns:
            print('Citation Diversity already in DDBB. Generating Doi-Novelty Dictionary')
            dois = self.DDBB['doi'].to_list()
            citdiv = self.DDBB['Citation Diversity'].to_list()
            self.doc_citdiv = defaultdict(float)
            for i,doi in enumerate(dois):
                self.doc_citdiv[doi] = citdiv[i]
            return 0
        if not hasattr(self,'normDTC'):
            self.set_norm_DTC()

        self.doc_citdiv =  defaultdict(self.create_nan_defaultdict)
        print('Getting All Citation Similarity')
        for doi in tqdm(list(self.references_abs_dict.keys()),desc = 'Citation Diversity...',miniters = 1000,mininterval = .5):
            self.doc_citdiv[doi] = self.citation_diversity(doi)
        print('Finished Generating All Citation Diversity')
        #self.write_dict(self.doc_citdiv,'doc_citdiv')
    '''    
############################### ENTROPY #####################################  

    def doc_shan(self,doc):
        return self.shan(self.doc_embeddings_dict[doc])
    
    def shan(self,vec):
        vec = [x for x in vec if x != 0]
        return (-1)*np.sum(np.multiply(vec,np.log2(vec)))
    
    def get_all_shannon_entropy(self):

        '''
        WRITING TO FILE TAKES MUCH LONGER THAN COMUPTATION. DO NOT WRITE
        '''
        #print('Writing all Distances between References for all Documents to a dictionary "doc_ref_dists"')
        if hasattr(self,'doc_shanent'):
            print('Object already has Shannon Entropy')
            return 0
        elif 'Shannon Entropy' in self.DDBB.columns:
            print('Shannon Entropy already in DDBB. Generating Doi-Novelty Dictionary')
            dois = self.DDBB['doc'].to_list()
            shanent = self.DDBB['Shannon Entropy'].to_list()
            self.doc_shanent = defaultdict(float)
            for i,doc in enumerate(docs):
                self.doc_shanent[doc] = shanent[i]
            return 0

        self.doc_shanent =  defaultdict(self.create_nan_defaultdict)
        print('Getting All Shannon Entropy')
        for doc in tqdm(self.docs,desc = 'Shannon Entropy...',miniters = 1000,mininterval = .5):
            self.doc_shanent[doc] = self.doc_shan(doc)
        print('Finished Generating All Shannon Entropy')
        #self.write_dict(self.doc_shanent,'doc_shanent')

    ##################################

    def binaryEnt(self,x):
        return -1*(x*np.log2(x) + (1-x)*np.log2(1-x))

    #def Fano(Pi_max, N, S):
    #    return np.log2(N-1)-S+Pi_max*np.log2((1/Pi_max - 1)*(1/(N-1))) - np.log2(1-Pi_max)

    def Fano(self,Pi_max, N, S):
        return (1-Pi_max)*np.log2(N-1)-S+self.binaryEnt(1-Pi_max)

    def CalcPi(self,N,S, thresh = .9):
        if math.isnan(S) or (N == 1) or (np.log2(N) < S):
            return float('nan')
        else:
            if (S <= 1):
                return fsolve(self.Fano,.9,(N,S))[0]
            else:
                return fsolve(self.Fano,.5,(N,S))[0]

    ################## INNOVATION ##########################################################        
            
    def generate_phrase_docs_dict(self):
        print('Generating phrase_docs_dict')
        self.phrase_docs_dict = defaultdict(list)
        for doc in tqdm(self.docs):
            phrases = self.doc_phrases_dict[doc]
            for phr in set(phrases):
                self.phrase_docs_dict[phr].append(doc)

    def read_in_phrase_link_params(self):
        if not hasattr(self,'phrase_docs_dict'):
            self.generate_phrase_docs_dict(s)

        if os.path.isfile('../bin/Fields/'+self.field+'/PMI_dict.pickle'):
            print('Reading PMI Dict')
            with open('../bin/Fields/'+self.field+'/words_completed.pickle', 'rb') as f:
                self.words_completed = pickle.load(f)
            with open('../bin/Fields/'+self.field+'/PMI_dict.pickle', 'rb') as f:
                self.PMI_dict = pickle.load(f)
        else:
            self.words_completed = 0
            self.PMI_dict = dict()

    def get_all_innov_dicts(self):
        if not hasattr(self,'distal_novelty_dict'):
            self.generate_distal_novelty_dicts(PMI = False)
            self.generate_distal_novelty_dicts(PMI = True)
            self.write_dict(self.distal_novelty_dict,'distal_novelty_dict')
            self.write_dict(self.impact_distal_novelty_dict,'impact_distal_novelty_dict')
        self.phrase_links = list(self.distal_novelty_dict.keys()) 
        self.get_all_innov()
        #self.get_author_innov_dicts()


    def generate_distal_novelty_dicts(self,PMI):
        # Makes Dictionary of Phrases 
        # Each dictionary value is a dictionary of 'dois' which are paper ids and 'year' which gives publication year
        # self.distal_novelty_dict contains the appearances of these phrases during their first year being seen
        # self.doc_impactinnov contains the apperances of these phrases after their first year of being seen
        #PMI = True implies the phrases links are phrases that appear in the same document, but are not necessarily next to each other
        #PMI = False runs the algorithm on the phrases from the PLDA
        if os.path.isfile('../bin/Fields/'+self.field+'/distal_novelty_dict.pickle'):
            print('Novelty Dict already Computed')
            return 0
            
        if not hasattr(self,'phrase_docs_dict'):
            print('Generating phrase_docs_dict')
            self.phrase_docs_dict = defaultdict(list)
            if not hasattr(self,'doc_phrases_dict'):
                self.read_dict('doc_phrases_dict')
            #for doc,phrases in tqdm(self.doc_phrases_dict.items()):
            for doc in tqdm(self.docs):
                for phr in set(self.doc_phrases_dict[doc]):
                    self.phrase_docs_dict[phr].append(doc)
            self.all_multiword_phrases = []
            for phr in tqdm(self.phrase_docs_dict.keys()):
                if ' ' in phr:
                    self.all_multiword_phrases.append(phr)
            if not hasattr(self,'PMI_dict'):
                self.read_dict('PMI_dict')

        print('Making PMI_df for computing')
        PMI_df = pd.DataFrame.from_dict(data = self.PMI_dict, orient = 'index').sort_values(by = 0)
        PMI_df = PMI_df[PMI_df[0] > 2]
        if not hasattr(self,'distal_novelty_dict'):
            self.distal_novelty_dict = dict()
        if not hasattr(self, 'impact_distal_novelty_dict'):
            self.impact_distal_novelty_dict = dict()

        if PMI == True:
            dum_phrases = PMI_df.index.to_list()

        else:
            dum_phrases = self.all_multiword_phrases

        start_phrase = len(self.distal_novelty_dict)
        print('Generating Phrase-Novelty Dicts: PMI =',PMI)
        #i = 0
        for phr in tqdm(dum_phrases):
            minyr = 0
            if PMI == True:
                phr1 = phr.split(',')[0]
                phr2 = phr.split(',')[1]
                #dummy_ind = list(set(self.phrase_docs_dict[phr1]).intersection(set(self.phrase_docs_dict[phr2])))
                docs = list(set(self.phrase_docs_dict[phr1]).intersection(set(self.phrase_docs_dict[phr2])))
                years = [self.doc_year_dict[doc] for doc in docs if self.doc_year_dict[doc] != 0]
                if len(years) != 0:
                    minyr = min(years)

                #dummy = self.paperstats_df[self.paperstats_df['id'].isin(dummy_ind) & self.paperstats_df['year'].notna()]
            else:    
                docs = self.phrase_docs_dict[phr]
                years = [self.doc_year_dict[doc] for doc in docs if self.doc_year_dict[doc] != 0]
                if len(years) != 0:
                    
                    
                    minyr = min(years)
                #dummy = self.paperstats_df[self.paperstats_df['id'].isin(self.phrase_docs_dict[phr]) & self.paperstats_df['year'].notna()]
            #if (dummy.shape[0] != 0) and (dummy['year'].min() >= 1970):
            if (minyr >= 1970):
                #dummy2 = dummy[dummy['year']==dummy['year'].min()]
                #dummy3 = dummy[(dummy['year']>dummy['year'].min())]
                dummy2 = [doc for doc in docs if self.doc_year_dict[doc] == minyr]
                dummy3 = [doc for doc in docs if self.doc_year_dict[doc] > minyr]
                yr3 = [self.doc_year_dict[doc] for doc in dummy3]
                self.distal_novelty_dict[phr] = {'docs' : dummy2, 'year' : minyr}
                if len(dummy3) == 0:
                    self.impact_distal_novelty_dict[phr] = {'docs' : [], 'year' : -1}
                else:
                    self.impact_distal_novelty_dict[phr] = {'docs' : dummy3, 'year' : yr3}
        print('Finished w/ Phrase-Novelty Dicts')
        self.phrase_links = list(self.distal_novelty_dict.keys())    


    def get_all_innov(self):

        #print('Writing all Distances between References for all Documents to a dictionary "doc_ref_dists"')

        if hasattr(self,'doc_innov'):
            print('Object already has Innovation')
            return 0
        if not hasattr(self,'PMI_dict'):
            print('No PMI_dict read in')
            self.read_dict('PMI_dict')
            if not hasattr(self,'PMI_dict'):
                print('No Pointwise Mutual Information Dict found. Compute before finding Innovation Scores')
                print('Loading Dictionary with NaNs')
                for doc in tqdm(list(set(self.docs)),miniters = 1000,mininterval = .5):
                    self.doc_innov[doc] = float('nan')
                    self.doc_impactinnov[doc] = float('nan')

        #docs_completed = len(list(self.doc_ref_dists.keys()))  
        start = timeit.default_timer()
        print('Generating doc_innov & doc_impactinnov_dict')
        self.doc_innov = defaultdict(int)

        docs = list(chain.from_iterable([self.distal_novelty_dict[phr]['docs'] for phr in self.distal_novelty_dict.keys()]))
        for doc in tqdm(list(set(docs)),miniters = 1000,mininterval = .5):
            self.doc_innov[doc] = docs.count(doc)
            #self.doc_innov[doc] = len([x for x in docs if x == doc])

        self.write_dict(self.doc_innov,'doc_innov')
        
        self.doc_impactinnov = defaultdict(int)
        for phr in tqdm(self.impact_distal_novelty_dict.keys(),miniters = 1000,mininterval = .5):
            for doc in self.distal_novelty_dict[phr]['docs']:
                self.doc_impactinnov[doc] += len(self.impact_distal_novelty_dict[phr]['docs'])
                
        
        self.write_dict(self.doc_impactinnov,'doc_impactinnov')


            
    ###########################################################################################    
    ###########################################################################################    
    ############################## CONCATENATE ANALYSIS TO DF #################################   
    ###########################################################################################    
    ###########################################################################################    
    ###########################################################################################  
    
    def generate_paperstats_df(self):
        print('Appending all Novelty scores to paperstats_df')
        self.paperstats_df = self.DDBB[self.DDBB['id'].isin(self.docs)][['id','year']]
        #self.paperstats_df['Has Abstract'] = False
        #self.paperstats_df.loc[self.paperstats_df['id'].isin(self.docs_abs),'Has Abstract'] = True
        docs = self.paperstats_df['id'].to_list()
        
        shanent = []
        citdiv = []
        refdiv = []
        disrupt = []
        innov = []
        cits = []
        refs = []
        cits_abs = []
        refs_abs = []
        if not hasattr(self,'doc_innov'):
            self.doc_innov = defaultdict(float)
        for doc in tqdm(docs):
            shanent.append(self.doc_shanent[doc])
            #citdiv.append(self.doc_citdiv[doc])
            #refdiv.append(self.doc_refdiv[doc])
            disrupt.append(self.doc_disrupt[doc])
            innov.append(self.doc_innov[doc])
            cits.append(len(self.citations_dict[doc]))
            refs.append(len(self.references_dict[doc]))
            #cits_abs.append(len(self.citations_abs_dict[doc]))
            #refs_abs.append(len(self.references_abs_dict[doc]))
        self.paperstats_df['Shannon Entropy'] = shanent
        #self.paperstats_df['Citation Diversity'] = citdiv
        #self.paperstats_df['Reference Diversity'] = refdiv
        self.paperstats_df['Disruption Index'] = disrupt
        self.paperstats_df['Innovation'] = innov
        self.paperstats_df['Citations'] = cits
        self.paperstats_df['References'] = refs
        #self.paperstats_df['Citations w/ abstracts'] = cits_abs
        #self.paperstats_df['References w/ abstracts'] = refs_abs
        cit_bins = np.array(list(np.linspace(0,15,16)) + list(np.logspace(np.log10(17),3.6,30)))
        self.paperstats_df['Citations (binned)'] = [cit_bins[np.argmax(cit_bins >= x)] for x in self.paperstats_df['Citations'].to_list()]
        
    def generate_all_novelty_scores(self):
        
        #self.get_all_reference_diversity()
        self.get_all_shannon_entropy()
        self.get_all_innov_dicts()
        #self.get_all_citation_diversity()
        self.get_all_disruption_index()
        
        
        
        self.generate_paperstats_df()
        
    def get_author_novelty_stats(self):
        if hasattr(self,'authorid_novelty_stats'):
            print('Object already has authorid_novelty_stats')
            return 0
        self.authorid_novelty_stats = defaultdict()
        for auth,docs in tqdm(self.authorid_docs_dict.items()):
            dummy = defaultdict()
            dummy_noss = defaultdict()
            docs_noss = [doc for doc in docs if self.is_ss_paper[doc] == False]
            dummy['Shannon Entropy'] = np.nanmean([self.doc_shanent[doc] for doc in docs])
            dummy['Shannon Entropy No SS'] = np.nanmean([self.doc_shanent[doc] for doc in docs_noss])

            #dummy['Reference Diversity'] = np.nanmean([self.doc_refdiv[doc] for doc in docs])
            #dummy['Reference Diversity No SS'] = np.nanmean([self.doc_refdiv[doc] for doc in docs_noss])

            #dummy['Citation Diversity'] = np.nanmean([self.doc_citdiv[doc] for doc in docs])
            #dummy['Citation Diversity No SS'] = np.nanmean([self.doc_citdiv[doc] for doc in docs_noss])

            dummy['Average Disruption Index'] = np.nanmean([self.doc_disrupt[doc] for doc in docs])
            dummy['Average Disruption Index No SS'] = np.nanmean([self.doc_disrupt[doc] for doc in docs_noss])
            
            dummy['Average Innovation'] = np.nanmean([self.doc_innov[doc] for doc in docs])
            dummy['Average Innovation No SS'] = np.nanmean([self.doc_innov[doc] for doc in docs_noss])
            
            if len(docs_noss) == 0: 
                dummy['Max Disruption Index No SS'] = float('nan')
                dummy['Total Innovation No SS'] = 0
            else:
                dummy['Max Disruption Index No SS'] = np.nanmax([self.doc_disrupt[doc] for doc in docs_noss])
                dummy['Total Innovation No SS'] = np.nansum([self.doc_innov[doc] for doc in docs_noss])
                
            dummy['Max Disruption Index'] = np.nanmax([self.doc_disrupt[doc] for doc in docs])
            dummy['Total Innovation'] = np.nansum([self.doc_innov[doc] for doc in docs])

            self.authorid_novelty_stats[auth] = dummy
        self.write_dict(self.authorid_novelty_stats,'authorid_novelty_stats')
        
    # Gets Ratio of # of times an author references/is cited by another author to number of authors referenced/that cite the given author
    #non_superstars = self.AJM[~self.AJM['author_ids'].isin(self.superstars)].groupby('author_ids').head(1)['author_ids'].to_list()
    def get_authorstats_df(self):
        sspapers = self.sspapers
        non_superstars = self.non_superstars

        #authorid_cit_rec_novelty = []
        authorstats_df = []
        author_sets = [self.superstars,self.non_superstars]
        j = 0
        for author_set in author_sets:


            i = 0
            labels = ['Shannon Entropy','Average Disruption Index','Max Disruption Index','Average Innovation',
                      'Total Innovation','Publication Count','Citation Count']
            authstats = [[] for x in range(len(labels))]
            authstats_noss = [[] for x in range(len(labels))]
            ratio_ss_reference = []
            ratio_ss_reference_noss = []
            auth_num_papers = []
            auth_num_refdiv = []
            auth_num_citdiv = []
            auth_num_papers_noss = []
            hindex = []
            for author in tqdm(author_set):

                pprs_noss = [doc for doc in self.authorid_docs_dict[author] if self.is_ss_paper[doc] == False]

                for i,label in enumerate(labels):
                    if i < len(labels)-2:
                        authstats[i] += [self.authorid_novelty_stats[author][label]]
                        authstats_noss[i] += [self.authorid_novelty_stats[author][label + ' No SS']]
                    else:
                        if len(self.auid_yr_pubs_cits[author][2024]) == 0:
                            authstats[i] += [float('nan')]
                            authstats_noss[i] += [float('nan')]
                        else:
                            authstats[i] += [self.auid_yr_pubs_cits[author][2024][i-(len(labels)-4)]]
                            if j == 0:
                                authstats_noss[i] += [float('nan')]
                            else:
                                if len(self.auid_yr_pubs_cits_noss[author][2024]) == 0:
                                    authstats_noss[i] += [float('nan')]
                                else:
                                    authstats_noss[i] += [self.auid_yr_pubs_cits_noss[author][2024][i-(len(labels)-4)]] 
                auth_num_papers.append(len(self.authorid_docs_dict[author]))
                auth_num_papers_noss.append(len(pprs_noss))

                doc_refs = [self.references_dict[doc] for doc in self.authorid_docs_dict[author]]

                doc_refs_noss = [self.references_dict[doc] for doc in pprs_noss]
                ratio_ss_reference_counts = [1 if any(self.is_ss_paper[ref] for ref in refs) else 0 for refs in doc_refs]
                ratio_ss_reference_counts_noss = [1 if any(self.is_ss_paper[ref] for ref in refs) else 0 for refs in doc_refs_noss]
                ratio_ss_reference += [np.sum(ratio_ss_reference_counts)/len(ratio_ss_reference_counts)] 
                ratio_ss_reference_noss += [np.sum(ratio_ss_reference_counts_noss)/len(ratio_ss_reference_counts_noss)] 
                hindex.append(self.h_index_dict[author])

                i+=1
            if j == 0:
                 authstats_noss = [float('nan') for l in labels]
            j+=1
            print('FINISHED')
            
            bin_width = 2
            df_data = {}
            
            dummydict = dict()
            for k,label in enumerate(labels):
                dummydict['Author '+label] = authstats[k]
                if k < len(labels)-2:
                    dummydict['Author '+label + ' (binned)'] = [round(x,2)-(((round(x,2)*100)%(bin_width/2))/100) for x in authstats[k]]
                dummydict['Author '+label + ' (without superstar papers)'] = authstats_noss[k]

            #df_data = df_data.update(dummydict)
            
            dict1 = {'Author' : author_set, 'Ratio of SS References (non-binned)': ratio_ss_reference,
                    'Ratio of SS References (non-binned)': ratio_ss_reference,
                    'Num Papers' : auth_num_papers,'Num Papers (without superstar papers)' : auth_num_papers_noss,
                    'h-index':hindex}
            

            dict1.update(dummydict)
            authorstats_df.append(pd.DataFrame.from_dict(dict1))

            

            #a = np.arange(.0,1.02,.02)

            #width = 10
            #dum = [round(width/100*int(x*100/width),2) for x in ratio_ss_reference]
            #dum = [(np.ceil(x*100) - np.ceil(x*100)%2)/100 for x in ratio_ss_reference]
            #dum = [(np.ceil(x*100) + np.ceil(x*100)%2)/100 for x in ratio_ss_reference]
            #dum = [a[np.argmax(a >= x)]-.01 if x < 1 else 1 for x in ratio_ss_reference] 
            #dum = [x if x != -.01 else 0 for x in dum]
            #round(5/100*int(i*100/5),2)


        authorstats_df = pd.concat(authorstats_df)
        authorstats_df['Group'] = 'Non-Superstar'
        authorstats_df.loc[authorstats_df['Author'].isin(self.superstars),'Group'] = 'Superstar'

        bins = [0,.2,.4,.6,.8,1]
        for i,bin_ in enumerate(bins[:-1]):
            if i == 0:
                authors = authorstats_df[(authorstats_df['Ratio of SS References (non-binned)'] >= bin_) & (authorstats_df['Ratio of SS References (non-binned)'] <= bins[i+1])]['Author'].to_list()
            else:
                authors = authorstats_df[(authorstats_df['Ratio of SS References (non-binned)'] > bin_)  & (authorstats_df['Ratio of SS References (non-binned)'] <= bins[i+1])]['Author'].to_list()
            authorstats_df.loc[authorstats_df['Author'].isin(authors),'Ratio of SS References'] = bin_ + .1


        
        self.authorstats_df = authorstats_df.reset_index()
        
        authors = self.authorstats_df[self.authorstats_df['Author'].notna()]['Author'].to_list()
        
        self.generate_authorid_startyr_dict()
        
        beginyr = []
        for auth in tqdm(self.authorstats_df['Author'].to_list()):
            beginyr.append(self.authorid_startyr_dict[auth])  
        
        self.authorstats_df['Year of First Pub.'] = beginyr
        
    ###########################################################################################    
    ###########################################################################################    
    ################################### SUPERSTARS ############################################   
    ###########################################################################################    
    ###########################################################################################    
    ###########################################################################################

    
    ##################################### H-INDEX #############################################
        
    def generate_h_index_df(self):
        h_index_dict = dict()

        print('Generating author to h_index dictionary')
        if not hasattr(self,'citations_dict'):
            self.read_dict('citations_dict')
        for author in tqdm(self.authorid_docs_dict.keys()):
            j = 1
            dummm = [len(self.citations_dict[doc]) for doc in self.authorid_docs_dict[author] if doc in self.citations_dict.keys()]
            while True:
                dummm2 = [x for x in dummm if x >= j]
                if j > len(dummm2):
                    h_index_dict[author] = j-1
                    break
                j+=1
        print('Finished')
        self.h_index_dict = h_index_dict

    def get_superstars(self,cutoff = .99):
        self.generate_h_index_df()
        print('Finding List of Superstars')
        h_df = pd.DataFrame(data = {'Author':self.h_index_dict.keys(),'h-index':self.h_index_dict.values()})
        h_df = h_df.sort_values(by = 'h-index')
        h_df['percentile'] = 1/h_df.shape[0]
        h_df['percentile'] = h_df['percentile'].cumsum()
        self.h_df = h_df
        h_cutoff = h_df[h_df['percentile']>= cutoff].head(1)['h-index'].to_list()[0]
        self.superstars = h_df[h_df['h-index']>= h_cutoff]['Author'].to_list()
        self.non_superstars = h_df[h_df['h-index'] < h_cutoff]['Author'].to_list()
        print(f'Total Superstars: {len(self.superstars)}')
        
        self.sspapers = []
        self.is_ss_paper = defaultdict(bool)
        for superstar in self.superstars:
            self.sspapers += self.authorid_docs_dict[superstar]
            for doc in self.authorid_docs_dict[superstar]:
                self.is_ss_paper[doc] = True
        self.sspapers = list(set(self.sspapers))
     
    ################################## COLLAB AUTHORS #########################################
    
    def get_collab_authors(self):

        #valid_authors = self.authorstats_df[(self.authorstats_df['Year of First Pub.'].notna()) & (self.authorstats_df['Year of First Pub.'] <= 2015)]['Author'].to_list()
        #beginyr_high = self.authorstats_df[(self.authorstats_df['Year of First Pub.'].notna()) & (self.authorstats_df['Year of First Pub.'] <= 2015)]['Year of First Pub.'].to_list()
        
        valid_authors = [a for a in self.authorid_docs_dict.keys() if (self.authorid_startyr_dict[a] <= 2015)]
        beginyr_high = [self.authorid_startyr_dict[a] for a in valid_authors]

        pprs_w_ss = []
        perc_ss_collab = []
        collabauth = []
        no_collabauth = []
        insp_auth = []

        for a in tqdm(valid_authors,miniters = 1000,mininterval = 1):
            #minyr = min([self.doc_year_dict[doi] for doi in self.authorid_docs_dict[a]])
            minyr = self.authorid_startyr_dict[a]
            earlypprs = [doi for doi in self.authorid_docs_dict[a] if (self.doc_year_dict[doi] <= minyr + 5)]
            dummypprs = [x for x in earlypprs if x in self.sspapers]
            #dummypprs = [x for x in self.authorid_docs_dict[a] if x in self.sspapers]
            pprs_w_ss.append(dummypprs)
            #dummyperc = len(dummypprs)/len(self.authorid_docs_dict[a])
            dummyperc = len(dummypprs)/len(earlypprs)
            perc_ss_collab.append(dummyperc)
            if (dummyperc > .5) & (len(earlypprs)>2):
                collabauth.append(a)
            if (dummyperc == 0):
                no_collabauth.append(a)
            else:
                insp_auth.append(a)
                
        self.collabauth = collabauth
        self.no_collabauth = no_collabauth
        self.insp_auth = insp_auth

        self.write_dict(collabauth,'collabauth')
        self.write_dict(no_collabauth,'no_collabauth')
        self.write_dict(insp_auth,'insp_auth')
        
    def num_papers_year_t(self,author,t=10):
        #print(author)
        minyr = list(self.auid_yr_pubs_cits[author].keys())[0]
        if minyr + t > 2024:
            t = 2024 - minyr
            #print(t)
        return self.auid_yr_pubs_cits[author][minyr + t][2]

    def num_papers_year_t_group(self,authors, t = 10):
        #return [num_papers_year_t(self,auth,t=10) for auth in authors]
        return [self.num_papers_year_t(auth,t=10) for auth in authors]

    #################################################################################################

    def num_papers_year_t_group_binned(self,authors, t = 10):
        return [(int(self.num_papers_year_t(auth,t=10)/2)*2) for auth in authors]
        #return [(int(num_papers_year_t(self,auth,t=10)/2)*2) for auth in authors]

    #################################################################################################

    def get_numpubs_firstyear_auth_dict(self,authors,t=10):
        # Makes dictionary of authors with x number of publications within t-years
        # as well as a dicitonary of number of authors per bin
        # The bin widths are on even intervals (0,2,4,6,8,etc)

        numpubs_auth = defaultdict(list)
        for i,auth in enumerate(authors):
            numpubs_auth[int(num_papers_year_t(auth,t)/2)*2].append(auth)

        numpubs_counts = defaultdict(int)
        for k in numpubs_auth.keys():
            numpubs_counts[k] = len(numpubs_auth[k])

        numpubs_auth_fin = defaultdict(dict)

        for k,v in numpubs_auth.items():
            year_dict = defaultdict(list)
            for auth in v:
                year_dict[list(self.auid_yr_pubs_cits[auth].keys())[0]].append(auth)
            numpubs_auth_fin[k] = year_dict

        return numpubs_auth_fin,numpubs_counts

    def get_matched_auth_groups_yr(self,collabs,noncollabs,t=10):
        numpubs_collab_auth,numpubs_collab_counts = self.get_numpubs_firstyear_auth_dict(self.collabauth,t=10)
        numpubs_noncollab_auth,numpubs_noncollab_counts = self.get_numpubs_firstyear_auth_dict(self.no_collabauth,t=10)
        print('Author NumPubs Dicts Constructed')
        rand_noncollab_auth = []
        rand_collab_auth = []
        #        if len(v) > len(numpubs_noncollab_auth[k]):
    #            for i in range(len(v) - len(numpubs_noncollab_auth[k])):
    #                randind = random.randint(0,len(numpubs_collab_auth[k])-1)
    #                numpubs_collab_auth[k].pop(randind)
    #        rand_collab_auth += numpubs_collab_auth[k]
        for k,v in numpubs_collab_auth.items():
            for y,a in numpubs_collab_auth[k].items():
                if len(a) > len(numpubs_noncollab_auth[k][y]):
                    for i in range(len(a) - len(numpubs_noncollab_auth[k][y])):
                        randind = random.randint(0,len(numpubs_collab_auth[k][y])-1)
                        numpubs_collab_auth[k][y].pop(randind)
                    rand_collab_auth += numpubs_collab_auth[k][y] 
                    for j in range(len(numpubs_collab_auth[k][y])):
                        randind = random.randint(0,len(numpubs_noncollab_auth[k][y])-1)
                        rand_noncollab_auth.append(numpubs_noncollab_auth[k][y][randind])
                        numpubs_noncollab_auth[k][y].pop(randind)

        self.collabauthyr_2 = rand_collab_auth
        self.no_collabauthyr_2 = rand_noncollab_auth


    #################################################################################################

    def get_numpubs_auth_dict(self,authors,t=10):
        # Makes dictionary of authors with x number of publications within t-years
        # as well as a dicitonary of number of authors per bin
        # The bin widths are on even intervals (0,2,4,6,8,etc)

        numpubs_auth = defaultdict(list)
        for auth in tqdm(authors, desc = 'numpubs_auth_dict'):
            numpubs_auth[int(self.num_papers_year_t(auth,t)/2)*2].append(auth)
            #numpubs_auth[int(num_papers_year_t(self,auth,t)/2)*2].append(auth)

        numpubs_counts = defaultdict(int)
        for k in tqdm(numpubs_auth.keys(),desc = 'numpubs_auth_dict'):
            numpubs_counts[k] = len(numpubs_auth[k])

        return numpubs_auth,numpubs_counts


    def get_matched_auth_groups(self,t=10):
        print('Getting Num Pubs for Collaborators')
        numpubs_collab_auth,numpubs_collab_counts = self.get_numpubs_auth_dict(self.collabauth,t=10)
        #numpubs_collab_auth,numpubs_collab_counts = get_numpubs_auth_dict(self,self.collabauth,t=10)
        print('Getting Num Pubs for Non-Collaborators')
        numpubs_noncollab_auth,numpubs_noncollab_counts = self.get_numpubs_auth_dict(self.no_collabauth,t=10)
        #numpubs_noncollab_auth,numpubs_noncollab_counts = get_numpubs_auth_dict(self,self.no_collabauth,t=10)
        print('Author NumPubs Dicts Constructed')
        rand_noncollab_auth = []
        rand_collab_auth = []
        for k,v in tqdm(numpubs_collab_auth.items()):
            if len(v) > len(numpubs_noncollab_auth[k]):
                for i in range(len(v) - len(numpubs_noncollab_auth[k])):
                    randind = random.randint(0,len(numpubs_collab_auth[k])-1)
                    numpubs_collab_auth[k].pop(randind)
            rand_collab_auth += numpubs_collab_auth[k]
            #print('Num Collaborators with,',k,'pubs:',len(rand_collab_auth[k]))
            for j in range(len(numpubs_collab_auth[k])):
                randind = random.randint(0,len(numpubs_noncollab_auth[k])-1)
                rand_noncollab_auth.append(numpubs_noncollab_auth[k][randind])
                numpubs_noncollab_auth[k].pop(randind)        
            #print('Num Collaborators with,',k,'pubs:',len(rand_collab_auth[k]))
            #print('Num Non-Collaborators with,',k,'pubs:',len(rand_noncollab_auth[k]),'\n')

        self.collabauth_2 = rand_collab_auth
        self.no_collabauth_2 = rand_noncollab_auth
    


        

