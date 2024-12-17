#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib.colors as colors
import timeit
import math
import random
import pickle
import os

from collections import defaultdict
from time import sleep
from tqdm import tqdm
import time
from itertools import chain
from ast import literal_eval

import Generate_Novelties as nov


def test_func():
    print("Hello!")

class read_corpus:
    def __init__(self,field = 'aps_old_check',aps_cits = True, ext = ''):
        self.field = field
        self.ext = ext
        if len(ext)>0:
            if ext[0] != '/':
                self.ext = '/' + ext
                print('File extension for modified parameters:',ext)
        #print('Reading in all raw data')
        
        if not os.path.isdir("../bin/Fields/"+field):
            print('Creating directory for',field)
            os.mkdir("../bin/Fields/"+field)
            if (self.ext != ''):
                if not os.path.isdir("../bin/Fields/"+field+self.ext):
                    os.mkdir("../bin/Fields/"+field+self.ext)
        # CONTAINS ALL APPROPRIATE LARGE-DATA ATTRIBUTES TO BE 
        # SET FOR INDIVIDUAL CLASS OBJECTS

        if os.path.isfile("../dataset/Fields/"+field+"/metadata.pickle"):
            #self.read_dict('metadata')
            self.metadata_to_pd()
            
        self.get_cit_ref_dicts()
            
        if not hasattr(self,'citations_dict'):
            print('Corpus does not have citations_dict. Get Citations dict before computing author stats.')
        else:
            self.get_doc_auid_dicts()
            self.get_superstars()
        
        other_dicts = ['auid_yr_pubs_cits','auid_yr_pubs_cits_noss','doc_concept_embeddings','doc_disrupt','doc_innov','doc_impactinnov',
                      'distal_novelty_dict','impact_distal_novelty_dict',
                       'collabauths_2','noncollabauths_2','collabauth','no_collabauth','insp_auth']
        print('\nREADING ALL POTENTIAL DICTIONARIES\n#########\n#########')
        for d in other_dicts:
            if hasattr(self,'doc_innov') and (d in ['distal_novelty_dict','impact_distal_novelty_dict']):
                print(d,'not read because doc-innovation dicts already computed')
                continue
            else:
                self.read_dict(d)
                
        other_csvs = ['groupstats_df','Fig4_Author_Novelty_Per_Year','Fig4_Author_Novelty_Per_Year_noss']
        csv_names = ['groupstats_df','df_ss','df_noss']
        for i,c in enumerate(other_csvs):
            if self.ext != '':
                if os.path.isfile("../bin/Fields/"+field+self.ext+"/"+c+".csv"):
                    print('Reading in,',self.ext+'_'+c)
                    dummy = pd.read_csv("../bin/Fields/"+field+self.ext+"/"+c+".csv")
                    setattr(self, csv_names[i],dummy)
                elif os.path.isfile("../bin/Fields/"+field+"/"+c+".csv"):
                    print('Reading in,',c)
                    dummy = pd.read_csv("../bin/Fields/"+field+"/"+c+".csv")
                    setattr(self, csv_names[i],dummy)
                else:
                    print(c,'not found.')
            elif os.path.isfile("../bin/Fields/"+field+"/"+c+".csv"):
                print('Reading in,',c)
                dummy = pd.read_csv("../bin/Fields/"+field+"/"+c+".csv")
                setattr(self, csv_names[i],dummy)
            else:
                print(c,'not found.')
             
    ################################################################################
            
    def read_dict(self,name):
        ext_print = ''
        if len(self.ext) > 0:
            ext_print = self.ext + '/'
        if not hasattr(self,name):
            if os.path.isfile('../bin/Fields/'+self.field+self.ext+'/'+name+'.pickle'):
                print('Reading In ' + ext_print + name)
                with open('../bin/Fields/'+self.field+self.ext+'/'+name+'.pickle', 'rb') as f:
                    dummy = pickle.load(f)
                setattr(self,name,dummy)
            elif os.path.isfile('../bin/Fields/'+self.field+'/'+name+'.pickle'):
                print('Reading In ' + name)
                with open('../bin/Fields/'+self.field+'/'+name+'.pickle', 'rb') as f:
                    dummy = pickle.load(f)
                setattr(self,name,dummy)
            else:
                print(name + ' not found in files. Nothing Read')
        else:
            print(name+' already read in')

    def write_dict(self,dict_,name, ext = False):
        filename = '../bin/Fields/'+self.field + '/' +name+'.pickle'
        if (ext == True):
            filename = '../bin/Fields/'+self.field+self.ext+'/'+name+'.pickle'
        print('Writing ' + name +' to File')
        with open(filename, 'wb') as f:
            pickle.dump(dict_,f)
        print('Finished Writing')
        
    def read_different_dict(self,name,field):
        if not hasattr(self,name):
            if os.path.isfile('../bin/Fields/'+field+'/'+name+'.pickle'):
                print('Reading In ' + name)
                with open('../bin/Fields/'+field+'/'+name+'.pickle', 'rb') as f:
                    dummy = pickle.load(f)
                setattr(self,name,dummy)

            else:
                print(name + ' not found in files. Nothing Read')
        else:
            print(name+' already read in')

    def write_different_dict(self,dict_,name,field):
        print('Writing ' + name +' to File')
        with open('../bin/Fields/'+ field + '/' + name +'.pickle', 'wb') as f:
            pickle.dump(dict_,f)
        print('Finished Writing')
        
    def create_nan_defaultdict(self):
        return float('nan')
    
    ################################################################################
    ################################################################################
    ################################################################################
    
    def metadata_to_pd(self):
        print('Constructing DataFrame from Metadata')
        print('Reading In metadata')
        with open('../dataset/Fields/'+self.field+'/metadata.pickle', 'rb') as f:
            dummy = pickle.load(f)
        setattr(self,'metadata',dummy)
        
        self.doc_concepts_dict = defaultdict(list)
        self.doc_lowest_level_dict = defaultdict(int)
        
        self.doc_authorids_dict = defaultdict(list)
        self.authorid_docs_dict = defaultdict(list)
        self.year_docs_dict = defaultdict(list)
        self.doc_year_dict = defaultdict(self.create_nan_defaultdict)
        self.doi_doc_dict = defaultdict(str)
        ids = []
        dois = []
        years = []
        authors = []
        abstracts = []
        refs = []
        for work,data in tqdm(self.metadata.items(),desc= "Converting metadata to DataFrame..."):
            ids += [work]
            dois += [data['doi']]
            self.doi_doc_dict[data['doi']] = work
            years += [data['year']]
            authors += [data['authors']]
            if 'abstract' in data.keys():
                abstracts += [data['abstract']]
            else:
                abstracts += [None]
            auths = [x['id'] for x in data['authors']]
            self.doc_year_dict[work] = data['year']
            self.year_docs_dict[data['year']] += [work]
            self.doc_authorids_dict[work] = auths
            for auth in auths:
                self.authorid_docs_dict[auth] += [work]
            if 'concepts' in data.keys():
                if len(data['concepts']) > 0:
                    self.doc_concepts_dict[work] = list(chain.from_iterable([[{'concept': c, 'level':int(k[3:]),'score':s} for c,s in v.items()] for k,v in data['concepts'].items()]))
                    self.doc_lowest_level_dict[work] = max([x['level'] for x in self.doc_concepts_dict[work]]+[0])

        self.DDBB = pd.DataFrame(data = {'id':ids,'doi':dois,'year':years,'author':authors,'paperAbstract':abstracts})

        self.docs = self.DDBB['id'].to_list()
        

        del self.metadata
        
    ################################################################################
    
    def get_cit_ref_dicts(self):
        
        if self.ext != '':
            if os.path.isfile("../bin/Fields/"+self.field+"/"+self.ext+"/citation_graph.csv"):
                print('Reading in Citation Graph')
                cit_graph = pd.read_csv("../bin/Fields/"+self.field+"/"+self.ext+"/citation_graph.csv")
            elif not os.path.isfile("../dataset/Fields/"+self.field+"/citation_graph.csv"):
                print('No Citation Graph Found')
            else:
                print('Reading in Citation Graph')
                cit_graph = pd.read_csv("../dataset/Fields/"+self.field+"/citation_graph.csv")
        elif self.ext == '':
            if not os.path.isfile("../dataset/Fields/"+self.field+"/citation_graph.csv"):
                print('No Citation Graph Found')
            else:
                print('Reading in Citation Graph')
                cit_graph = pd.read_csv("../dataset/Fields/"+self.field+"/citation_graph.csv")

        ced_p = []
        cing_p = []
        cited_papers = cit_graph['cited_doc'].to_list()
        citing_papers = cit_graph['citing_doc'].to_list()
        self.citations_dict = defaultdict(list)
        self.references_dict = defaultdict(list)

        for i in tqdm(range(len(cited_papers)),desc = 'Getting Citation and Reference Dicts'):
            self.citations_dict[cited_papers[i]].append(citing_papers[i])
            self.references_dict[citing_papers[i]].append(cited_papers[i])
        #self.docs = list(set(docs))
        del cit_graph
        
            
    ################################################################################                
            
    def get_doc_auid_dicts(self):
        print('Getting all authorid doc dicts')
        docs = []
        for doc in tqdm(self.docs, desc = 'Removing Papers with greater than 25 authors'):
            if len(self.doc_authorids_dict[doc]) <= 25:
                docs.append(doc)
        self.docs = docs
        #self.read_dict('doc_authorids')

        doc_authorids_dict = defaultdict(list)
        authorid_docs_dict = defaultdict(list)
        for doc in tqdm(self.docs,total = len(self.docs),desc = 'Getting doc_authorid_dicts from cleaned document set'):
            auths = self.doc_authorids_dict[doc]
            doc_authorids_dict[doc] = auths
            for a in auths:
                authorid_docs_dict[a] += [doc]
        self.doc_authorids_dict = doc_authorids_dict
        self.authorid_docs_dict = authorid_docs_dict
                
    ##################################################################################
    
    def generate_h_index_df(self):
        h_index_dict = dict()

        print('Generating author to h_index dictionary')
        if not hasattr(self,'citations_dict'):
            print('No Citation Dictionary')
            return None
            #self.read_dict('citations_dict')
        for author in tqdm(self.authorid_docs_dict.keys(), desc = 'Getting h-index dict'):
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
        
    ###############################################################################################
    
    def get_multiple_pubs_cits(self,noss = False,ext = False):
        self = nov.get_multiple_pubs_cits(self,noss,ext)
        
    def get_all_disruption_index(self,t=10,ext = False):
        self = nov.get_all_disruption_index(self,t,ext)
        
    def get_all_innov_dicts(self,ext=False):
        self = nov.get_all_innov_dicts(self,ext)
        
    def get_all_shanent(self,concepts = True, word_thresh = 50,num_topics = 20,seed_ = 1,ext = False):
        self = nov.get_shannon_entropy_dict(self,concepts, word_thresh,num_topics,seed_,ext)
    
    
    
    