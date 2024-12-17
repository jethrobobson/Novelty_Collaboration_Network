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

'''
emal.generate_paperstats_df()
emal.get_author_novelty_stats()
emal.get_authorstats_df()

emal.get_collab_authors()
emal.get_matched_auth_groups()
'''

###########################################################################################    
###########################################################################################    
############################## CONCATENATE ANALYSIS TO DF #################################   
###########################################################################################    
###########################################################################################    
###########################################################################################  

############################### PAPERSTATS_DF ##########################################

def generate_paperstats_df(self):
    print('Appending all Novelty scores to paperstats_df')
    self.paperstats_df = self.DDBB[self.DDBB['id'].isin(self.docs)][['id','year']]
    docs = self.paperstats_df['id'].to_list()
    
    lbls_0 = ['doc_shanent','doc_innov','doc_disrupt']
    df_lbls_0 = ['Shannon Entropy','Innovation','Disruption Index']
    dicts_0 = [self.doc_shanent,self.doc_innov,self.doc_disrupt]
    lbls,df_lbls,dicts = [],[],[]
    for i,lbl in enumerate(lbls_0):
        if hasattr(self,lbl):
            lbls.append(lbl)
            df_lbls.append(df_lbls_0[i])
            dicts.append(dicts_0[i])
            
    nov_dicts = [[] for x in dicts]
    cits = []
    refs = []

    for doc in tqdm(docs):
        for i,dct in enumerate(dicts):
            nov_dicts[i].append(dct[doc])
        cits.append(len(self.citations_dict[doc]))
        refs.append(len(self.references_dict[doc]))
    for i,dct in enumerate(nov_dicts):
        self.paperstats_df[df_lbls[i]] = dct
    self.paperstats_df['Citations'] = cits
    self.paperstats_df['References'] = refs
    cit_bins = np.array(list(np.linspace(0,15,16)) + list(np.logspace(np.log10(17),np.log10(max(cits)+1),30)))
    self.paperstats_df['Citations (binned)'] = [cit_bins[np.argmax(cit_bins >= x)] for x in self.paperstats_df['Citations'].to_list()]
    
    return self


def get_author_novelty_stats(self):
    if hasattr(self,'authorid_novelty_stats'):
        print('Object already has authorid_novelty_stats')
        return 0
    self.authorid_novelty_stats = defaultdict()
    lbls_0 = ['doc_shanent','doc_innov','doc_disrupt']
    df_lbls_0 = ['Shannon Entropy','Innovation','Disruption Index']
    dicts_0 = [self.doc_shanent,self.doc_innov,self.doc_disrupt]
    lbls,df_lbls,dicts = [],[],[]
    for i,lbl in enumerate(lbls_0):
        if hasattr(self,lbl):
            lbls.append(lbl)
            df_lbls.append(df_lbls_0[i])
            dicts.append(dicts_0[i])
            
    for auth,docs in tqdm(self.authorid_docs_dict.items()):
        dummy = defaultdict()
        dummy_noss = defaultdict()
        docs_noss = [doc for doc in docs if self.is_ss_paper[doc] == False]
        
        for j,lbl in enumerate(df_lbls_0):
            dummy['Average '+ lbl] = np.nanmean([dicts[j][doc] for doc in docs])
            dummy['Average '+ lbl + ' No SS'] = np.nanmean([dicts[j][doc] for doc in docs_noss])
            
            if lbl == 'Innovation':
                dummy['Total Innovation'] = np.nansum([self.doc_innov[doc] for doc in docs])
                if len(docs_noss) == 0: 
                    dummy['Total Innovation No SS'] = 0
                else:
                    dummy['Total Innovation No SS'] = np.nansum([self.doc_innov[doc] for doc in docs_noss])
            elif lbl == 'Disruption Index':
                dummy['Max Disruption Index'] = np.nanmax([self.doc_disrupt[doc] for doc in docs])
                if len(docs_noss) == 0: 
                    dummy['Max Disruption Index No SS'] = float('nan')
                else:
                    dummy['Max Disruption Index No SS'] = np.nanmax([self.doc_disrupt[doc] for doc in docs_noss])

        self.authorid_novelty_stats[auth] = dummy
    self.write_dict(self.authorid_novelty_stats,'authorid_novelty_stats')
    return self

# Gets Ratio of # of times an author references/is cited by another author to number of authors referenced/that cite the given author
#non_superstars = self.AJM[~self.AJM['author_ids'].isin(self.superstars)].groupby('author_ids').head(1)['author_ids'].to_list()

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
    return self
                
                
def get_authorstats_df(self):
    sspapers = self.sspapers
    non_superstars = self.non_superstars

    #authorid_cit_rec_novelty = []
    authorstats_df = []
    author_sets = [self.superstars,self.non_superstars]
    j = 0
    for author_set in author_sets:


        i = 0
        labels = ['Average Shannon Entropy','Average Disruption Index','Max Disruption Index','Average Innovation',
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

    self = generate_authorid_startyr_dict(self)

    beginyr = []
    for auth in tqdm(self.authorstats_df['Author'].to_list()):
        beginyr.append(self.authorid_startyr_dict[auth])  

    self.authorstats_df['Year of First Pub.'] = beginyr
    
    return self

################################## COLLAB AUTHORS #########################################
    
def get_collab_auths(self):
    if hasattr(self,'collabauth'):
        print('Collaborator groups already found')
        return self
    self.is_ss_dict = defaultdict(bool)
    for auth in self.authorid_docs_dict:
        self.is_ss_dict[auth] = False
    for auth in self.superstars:
        self.is_ss_dict[auth] = True
        
    authors = self.authorstats_df[self.authorstats_df['Author'].notna()]['Author'].to_list()
    beginyr = []
    authorid_startyr_dict = defaultdict()
    for auth in self.authorstats_df['Author'].to_list():
        dummy = list(self.auid_yr_pubs_cits[auth].keys())[0]
        if (dummy == 't_0') | (dummy == 0):
            doop = float('nan')
        else:
            doop = int(dummy)
        beginyr.append(doop)
        authorid_startyr_dict[auth] = doop
    self.authorstats_df['Year of First Pub.'] = beginyr


    valid_authors = self.authorstats_df[(self.authorstats_df['Year of First Pub.'].notna()) & (self.authorstats_df['Year of First Pub.'] <= 2015)]['Author'].to_list()
    beginyr_high = self.authorstats_df[(self.authorstats_df['Year of First Pub.'].notna()) & (self.authorstats_df['Year of First Pub.'] <= 2015)]['Year of First Pub.'].to_list()

    pprs_w_ss = []
    perc_ss_collab = []
    collabauth = []
    no_collabauth = []
    insp_auth = []

    for a in tqdm(valid_authors,desc = 'Get Frequent Collaborators and Non-collaborators',miniters = 1000,mininterval = 1):
        #minyr = min([self.doc_year_dict[doi] for doi in self.authorid_docs_dict[a]])
        minyr = authorid_startyr_dict[a]
        earlypprs = [doi for doi in self.authorid_docs_dict[a] if (self.doc_year_dict[doi] <= minyr + 5)]
        dummypprs = [x for x in earlypprs if x in self.sspapers]
        if self.is_ss_dict[a] == True:
            dummypprs = [x for x in dummypprs if len([y for y in self.doc_authorids_dict[x] if self.is_ss_dict[y] == True]) > 1]
        #dummypprs = [x for x in self.authorid_docs_dict[a] if x in self.sspapers]
        #pprs_w_ss.append(dummypprs)
        #dummyperc = len(dummypprs)/len(self.authorid_docs_dict[a])
        dummyperc = len(dummypprs)/len(earlypprs)
        #perc_ss_collab.append(dummyperc)
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

    return self

def num_papers_year_t(self,author,t=10):
    #print(author)
    minyr = list(self.auid_yr_pubs_cits[author].keys())[0]
    if minyr + t > 2020:
        t = 2020 - minyr
        #print(t)
    return self.auid_yr_pubs_cits[author][minyr + t][2]

def num_papers_year_t_group(self,authors, t = 10):
    return [num_papers_year_t(self,auth,t=10) for auth in authors]

######################################################################################

#################################################################################################

def get_numpubs_auth_dict(self,authors,t=10):
    # Makes dictionary of authors with x number of publications within t-years
    # as well as a dicitonary of number of authors per bin
    # The bin widths are on even intervals (0,2,4,6,8,etc)
    
    numpubs_auth = defaultdict(list)
    for auth in tqdm(authors, desc = 'numpubs_auth_dict'):
        numpubs_auth[int(num_papers_year_t(self,auth,t)/2)*2].append(auth)

    numpubs_numauths = defaultdict(int)
    for k in tqdm(numpubs_auth.keys(),desc = 'numpubs_numauths_dict'):
        numpubs_numauths[k] = len(numpubs_auth[k])
        
    return numpubs_auth,numpubs_numauths


def get_matched_auth_groups(self,collabs,noncollabs,t=10):
    '''
    Matching an early collaborator with an arbitrary non-collaborators of 
    equal publication rate, to create a case group of collaborators with 
    an equal sized control group of non-collaborators
    '''
    if hasattr(self,'collabauths_2'):
        print('Matched Collaborator groups already found')
        return self
    print('Getting Num Pubs for Collaborators')
    numpubs_collab_auth,numpubs_collab_counts = get_numpubs_auth_dict(self,collabs,t=10)
    print('Getting Num Pubs for Non-Collaborators')
    numpubs_noncollab_auth,numpubs_noncollab_counts = get_numpubs_auth_dict(self,noncollabs,t=10)
    print('Author NumPubs Dicts Constructed')
    rand_noncollab_auth = []
    rand_collab_auth = []
    for k,v in tqdm(numpubs_collab_auth.items(), desc = 'Getting equal size non/collab groups'):
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
    self.collabauths_2 = rand_collab_auth
    self.noncollabauths_2 = rand_noncollab_auth
    
    self.write_dict(self.collabauths_2,'collabauths_2')
    self.write_dict(self.noncollabauths_2,'noncollabauths_2')
    
    return self

#########################################################################################################################

def get_au_yr_all_nov_dict(self,noss = False):
    labels = ['Shannon Entropy','Disruption Index','Innovation']
    dicts = [self.doc_shanent,self.doc_disrupt,self.doc_innov]
    auths = self.authorstats_df[(self.authorstats_df['Year of First Pub.'] <= 2015)]['Author'].to_list()
    
    au_yr_novdict_noss = defaultdict(dict)
    #novdict_noss = []

    num_iterations = 10
    start1 = timeit.default_timer()
    start2 = timeit.default_timer()
    for a in tqdm(auths, desc= 'Getting Yearly Novelty Dicts for all Novelties'):
        au_yr_novdict_noss[a] = get_single_au_yr_all_nov_dict(self,a,noss)

    return au_yr_novdict_noss

def get_single_au_yr_all_nov_dict(self,a,noss = False):
    labels = ['Shannon Entropy','Disruption Index','Innovation']
    dicts = [self.doc_shanent,self.doc_disrupt,self.doc_innov]
    
    dummy = defaultdict(lambda: defaultdict(list))
    remaining_papers = self.authorid_docs_dict[a]
    #total_papers = remaining_papers
    if noss == True:
        if a not in self.superstars:
            remaining_papers = [x for x in remaining_papers if self.is_ss_paper[x]==False]
    if len(remaining_papers) != 0:
        yrs = list(np.arange(min([self.doc_year_dict[doc] for doc in remaining_papers]),2025,1))
        for j,year in enumerate(yrs):
            dummypprs = [doc for doc in remaining_papers if self.doc_year_dict[doc] == year]
            remaining_papers = list(set(remaining_papers) - set(dummypprs))
            for i,label in enumerate(labels):
                dummynov = np.nanmean([dicts[i][doc] for doc in dummypprs])
                k = 0
                if label == 'Innovation':
                    if j == 0:
                        dummy[label][year] = [dummynov,dummynov]
                    else:
                        dummy[label][year] = [dummynov,np.nansum([dummynov,dummy[label][year-1][1]])]
                elif label == 'Disruption Index':
                    if j == 0:
                        dummy[label][year] = [dummynov,dummynov]
                    else:
                        dummy[label][year] = [dummynov,np.nanmax([dummynov,dummy[label][year-1][1]])]
                else:
                    dummy[label][year] = dummynov
            #novdict_noss.append(dummy[year])
    return dummy

def get_auth_allnov_per_year_df_fig4(self,lbls,novdict,noss = False):
    pubs = []
    cits = []
    totalpubs = []
    totalcits = []
    t0 = []
    auth = []
    years = []
    distnov_noss = []
    cum_distnov_noss = []
    inf_div_noss = []
    shanent_noss = []
    cit_div_noss = []

    if noss == True:
        aypc_dict = self.auid_yr_pubs_cits_noss
    else:
        aypc_dict = self.auid_yr_pubs_cits

    auths = self.authorstats_df[(self.authorstats_df['Year of First Pub.'] <= 2015)]['Author'].to_list()

    nov_dicts = [[] for x in range(len(lbls))]
    if 'Innovation' in lbls:
        nov_dicts.append([])
    if 'Disruption Index' in lbls:
        nov_dicts.append([])
    start1 = timeit.default_timer()
    start2 = timeit.default_timer()
    num_iterations = 3
    for a in tqdm(auths, desc = 'Getting Yearly Author Novelty Dataframe'):
        #if len(self.auid_yr_pubs_cits_noss[a]['t_0']) != 0:
        if len(aypc_dict[a]['t_0']) != 0:
            dummystats = list(aypc_dict[a].values())[:-1]
            years += (list(aypc_dict[a].keys())[:-1])
            pubs += [x[0] for x in dummystats]
            cits += [x[1] for x in dummystats]
            totalpubs += ([x[2] for x in dummystats])
            totalcits += ([x[3] for x in dummystats])
            t0 += [0] + list(aypc_dict[a].values())[-1]
            #if len(list(self.auid_yr_pubs_cits_noss[a].values())[-1])+1 != len(list(self.auid_yr_pubs_cits_noss[a].keys())[:-1]):
            #if len(list(au_yr_distnov_noss[a].values())) != len(dummystats):
            #    print(a)
            #    break
            auth += [a for x in dummystats]

            #dummystats = list(au_yr_distnov_noss[a].values())
            l = 0
            for k,lbl in enumerate(lbls):
                if (lbl != 'Innovation') and (lbl != 'Disruption Index'):
                    nov_dicts[l] += list(novdict[a][lbl].values())

                else:
                    nov_dicts[l] += [x[0] for x in novdict[a][lbl].values()]
                    l+=1
                    nov_dicts[l] += [x[1] for x in novdict[a][lbl].values()]
                l+=1
            '''
            if len(nov_dicts[0]) < len(years):
                print(a)
                print(len(years))
                print(len(nov_dicts[0]))
                break


            distnov_noss += [x[0] for x in dummystats]
            cum_distnov_noss += [x[1] for x in dummystats]
            inf_div_noss += list(au_yr_inf_div_noss[a].values())
            cit_div_noss += list(au_yr_cit_div_noss[a].values())
            shanent_noss += list(au_yr_shanent_noss[a].values())
                '''
    dummydict = dict()
    l = 0
    for k,label in enumerate(lbls):
        if label == 'Innovation':
            dummydict[label] = nov_dicts[l]
            l+=1
            dummydict['Total '+label] = nov_dicts[l]
        elif label == 'Disruption Index':
            dummydict[label] = nov_dicts[l]
            l+=1
            dummydict['Max '+label] = nov_dicts[l]
        else:
            dummydict[label] = nov_dicts[l]
        l+=1

    dict1 = {'Author':auth,'Year':years,'t0':t0,
                                'Yearly Publications':pubs,'Yearly Citations':cits,
                                'Total Publications':totalpubs,'Total Citations':totalcits}
    #return pd.DataFrame.from_dict(dict1)

    dict1.update(dummydict)
    fig4_df = pd.DataFrame.from_dict(dict1)

    lst1 = np.array(fig4_df['Total Publications'].to_list())
    lst2 = np.array(fig4_df['Total Citations'].to_list())
    citperpub = lst2/lst1
    fig4_df['Citations per Publication'] = list(citperpub)
    
    return fig4_df

###############################################################################

def get_all_collabauth_analysis(self,ext = False):
    ext_lbl = ''
    if ext == True:
        ext_lbl = self.ext
    
    self = get_collab_auths(self)
    self = get_matched_auth_groups(self,self.collabauth,self.no_collabauth)

    fig4_nov_dict = get_au_yr_all_nov_dict(self)
    labels = ['Shannon Entropy','Disruption Index','Innovation']
    self.df_ss = get_auth_allnov_per_year_df_fig4(self,labels,fig4_nov_dict)

    fig4_nov_dict_noss = get_au_yr_all_nov_dict(self,True)
    labels = ['Shannon Entropy','Disruption Index','Innovation']
    self.df_noss = get_auth_allnov_per_year_df_fig4(self,labels,fig4_nov_dict_noss,True)
    
    print('Saving df_ss')
    self.df_ss.to_csv('../bin/Fields/'+self.field+ext_lbl+'/Fig4_Author_Novelty_Per_Year.csv')
    print('Saving df_noss')
    self.df_noss.to_csv('../bin/Fields/'+self.field+ext_lbl+'/Fig4_Author_Novelty_Per_Year_noss.csv')
    
    return self
    

###########################################################################################    
###########################################################################################    
############################## GROUP STATS FUNCTIONS #### #################################   
###########################################################################################    
###########################################################################################    
###########################################################################################  

############################### PAPERSTATS_DF ##########################################

def set_norm_DTC(self, concepts = True):
    if concepts == True:
        dict_ = self.doc_concept_embeddings
    else:
        dict_ = self.doc_embeddings_dict
    start = timeit.default_timer()
    if hasattr(self,'normDTC'):
        if len(self.normDTC) < len(list(dict_.values())):
            print('Normalization started but not yet completed')
            print(str(len(self.normDTC)) + ' out of ' + str(len(dict_)) + ' remaining')
            self.normDTC = defaultdict(list)
        else:
            print('Normalization Already Completed')
            return self
    else:
        print('Beginning l2 Normalizaiton of Document Embeddings') 
        self.normDTC = defaultdict(list)

    for doc in tqdm(self.docs, desc = "Getting norm DTC"):
        self.normDTC[doc] = dict_[doc]/ np.linalg.norm(dict_[doc])
    stop = timeit.default_timer()
    print('Completed Normalization of DTC:',(stop-start),'(s)')
    return self


#################### SIMILARITY CALCS ###############################################################


def sqrt_SoS(self,vec):
    return np.sqrt(np.sum(np.multiply(vec,vec)))

def doc_sim(self, doi_1,doi_2):
    if not hasattr(self,'normDTC'):
        print('Beginning Normalizaiton of Document_Topic_Count') 
        self.set_norm_DTC()
    if (len(self.normDTC[doi_1]) == 0) or (len(self.normDTC[doi_2]) == 0):
        return float('nan')

    result = cos_sim(self,self.normDTC[doi_1],self.normDTC[doi_2])
    if result <= 10**(-8):
        return 1.01*10**(-8)
    elif result >= .9999999:
        return 1
    return result

def cos_sim(self,vec1,vec2):
    return np.dot(vec1,vec2)

def get_all_sim(self,dois):
    
    dummy_dois = [x for x in dois if len(self.doc_embeddings_dict[x]) > 0]
    if len(dummy_dois) >= 100:
        all_sim = []
        freq = []
        tot = len(dummy_dois)*(len(dummy_dois)-1)/2
        for k in range(len(dummy_dois)):
            dummy_sim = []
            for l in range(len(dummy_dois)):
                if k < l:
                    dummy_sim += [doc_sim(self,dummy_dois[k],dummy_dois[l])]
            all_sim.append(np.nanmean(dummy_sim))
            freq.append(len(dummy_dois)-(k+1))
        return list(np.array(all_sim)*np.array(freq)*len(all_sim)/tot)
    else:
        dummy_sim = []
        for k in range(len(dummy_dois)):
            for l in range(len(dummy_dois)):
                if k < l:
                    dummy_sim += [doc_sim(self,dummy_dois[k],dummy_dois[l])]
    return dummy_sim

def create_nan_defaultdict():
        return float('nan')
    
def get_all_doi_citsim(self):

    self.read_dict('doc_citsim')
    if (not hasattr(self,'doc_citsim')):
        self.doc_citsim = defaultdict(self.create_nan_defaultdict)
    if (len(self.doc_embeddings_dict) != len(self.doc_citsim)):
        dois = self.docs
        start1 = timeit.default_timer()
        start2 = timeit.default_timer()
        df_citsim = []
        print('Getting All Citation Similarity')
        #i = 0
        for doi in tqdm(list(self.docs)):
            if (doi not in self.doc_citsim.keys()):
                if len(self.citations_dict[doi])>1:
                    self.doc_citsim[doi] = np.nanmean(get_all_sim(self,self.citations_dict[doi]))
                else:
                    self.doc_citsim[doi] = float('nan')
                    
        stop = timeit.default_timer()
        print('Finished Generating All Citation Similarities:',(stop-start1),'(s)')
        self.write_dict(self.doc_citsim,'doc_citsim')
    return self

def generate_doc_yr_cit_dict(self):
    self.doc_yr_cit_dict = defaultdict(dict)
    start = timeit.default_timer()
    for doi in tqdm(self.docs, desc = 'Generating doc_yr_cit_dict...'):
        doop = [len([x for x in self.citations_dict[doi] if self.doc_year_dict[x]==year]) for year in np.arange(self.doc_year_dict[doi],2021,1)]
        #d = defaultdict(create_nan_defaultdict,dict(zip(np.arange(self.doc_year_dict[doi],2021,1),doop)))
        self.doc_yr_cit_dict[doi] = defaultdict(create_nan_defaultdict,dict(zip(np.arange(self.doc_year_dict[doi],2021,1),doop)))
    stop = timeit.default_timer()
    print("doc_yr_cit_dict generated:",(stop-start),"(s)")
    return self

#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################

def get_ss_cit_network_2(self):

    print('Creates superstar network with cited and citing dois as well as the document_index '
          ,'where they are located in abstracts_df for LDA')
    
    superstars = self.superstars
    i = 0
    start1 = timeit.default_timer()
    start2 = timeit.default_timer()
    ss_cit_network = []
    num_time_checks = 5
    for superstar in tqdm(superstars,desc ='Collecting SS Citation DF data...'):
        papers = self.authorid_docs_dict[superstar]

        citingpapers = []
        citedpapers = []
        for doc in papers:
            citingpapers += self.citations_dict[doc]
            citedpapers += [doc for x in self.citations_dict[doc]]

        ss_cit_network += [pd.DataFrame(data = {'superstar':superstar,
                                                       'superstar_doi':citedpapers,'citing_doc':citingpapers})]

    self.ss_cit_network = pd.concat(ss_cit_network).reset_index()
    print('DONE!')

    # GET ALL AUTHORS OF CITING PAPERS
    print('Get All Authors of Citing Papers')
    self.ss_cit_network['citing authors'] = float('nan')
    self.ss_cit_network = self.ss_cit_network.sort_values(by = 'citing_doc')
    citing_doi_counts = self.ss_cit_network.groupby('citing_doc').count()['superstar'].to_list()
    authorid_list = []
    i = 0
    for citing_paper in tqdm(self.ss_cit_network.groupby('citing_doc').head(1)['citing_doc'].to_list(),desc = 'getting authorid list'):
        if len(self.doc_authorids_dict[citing_paper]) == 0 :
            authorid_list += [(float('nan')) for k in range(citing_doi_counts[i])]
        else:
            authorid_list+= [self.doc_authorids_dict[citing_paper] for k in range(citing_doi_counts[i])]
        i+=1

    self.ss_cit_network['citing authors'] = authorid_list
    self.ss_cit_network.sort_values(by = ['superstar','superstar_doi'])
    self.ss_cit_network = self.ss_cit_network[self.ss_cit_network['citing authors'].notna()]

    self.citauth_ss_citdocs_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    self.citauth_ss_citdocs_dict_noss = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    ss = self.ss_cit_network['superstar'].to_list()
    citdocs = self.ss_cit_network['citing_doc'].to_list()
    ssdocs = self.ss_cit_network['superstar_doi'].to_list()
    citauths = self.ss_cit_network['citing authors'].to_list()
        
        

    i = 0
    for auths in tqdm(citauths, desc = 'Generating citauth_ss_citdocs_dict'):
        if self.is_ss_paper[citdocs[i]] == False:
            for auth in auths:
                self.citauth_ss_citdocs_dict_noss[auth][ss[i]]['citing_doc'] += [citdocs[i]] 
                self.citauth_ss_citdocs_dict_noss[auth][ss[i]]['ss_doc'] += [ssdocs[i]]
            
        for auth in auths:
            self.citauth_ss_citdocs_dict[auth][ss[i]]['citing_doc'] += [citdocs[i]] 
            self.citauth_ss_citdocs_dict[auth][ss[i]]['ss_doc'] += [ssdocs[i]] 
        i+=1

    del self.ss_cit_network
    del authorid_list
    del citing_doi_counts 

    # Form Ranked Inspirator Network

    #self.ss_cit_network.loc[self.ss_cit_network['citing authors'].notna(),'citing authors'] = list(map(literal_eval,self.ss_cit_network[self.ss_cit_network['citing authors'].notna()]['citing authors'].to_list()))
    #self.ss_cit_network = self.ss_cit_network[self.ss_cit_network['citing authors'].notna()].explode('citing authors')
    
    if os.path.isfile("../bin/Fields/"+self.field+"/ranked_superstar_citation_network.csv"):
        if not hasattr(self,'ranked_ss_cit_network'):
            print("Reading ranked_superstar_citation_network")
            self.ranked_ss_cit_network = pd.read_csv("../bin/Fields/"+self.field+"/ranked_superstar_citation_network.csv")
    else:

        i = 0
        start1 = timeit.default_timer()
        start2 = timeit.default_timer()

        print('FORM RANKED SUPERSTAR NETWORK')
        ss_inspirator_cits = defaultdict()
        i = 0
        dfs = []
        for ss in tqdm(self.superstars):
            inspirator_cits = defaultdict(list)
            for doc in self.authorid_docs_dict[ss]:
                for cit in self.citations_dict[doc]:  
                    for auth in self.doc_authorids_dict[cit]:
                        inspirator_cits[auth] += [cit]
            ss_inspirator_cits[ss] = inspirator_cits
            num_cits = [len(list(set(inspirator_cits[auth]))) for auth in list(inspirator_cits.keys())]
            prop_cits = [num_cits[j]/len(self.authorid_docs_dict[auth]) for j,auth in enumerate(list(inspirator_cits.keys()))]
            dfs += [pd.DataFrame(data = {'superstar':ss, 'inspirators': list(inspirator_cits.keys()), 
                                                'num_citations': num_cits,
                                        'proportional_citations':prop_cits})]
            i+=1
        del self.citauth_ss_citdocs_dict
        self.ranked_ss_cit_network = pd.concat(dfs).reset_index().drop(columns = 'index')
        self.ranked_ss_cit_network = self.ranked_ss_cit_network[~self.ranked_ss_cit_network['inspirators'].isin(self.superstars)]
        print('Finished Forming Ranked SS Network')

    stop = timeit.default_timer()
    print('Total Time:',stop-start1,'(s)')
    return self

######################### COMPARING INFORMATION of Inspirator Groups ####################################################

def get_ss_insp_dfs_groups_noss(self,_superstar_,group_perc = 'inspirators',end_bounds = np.arange(0,1.001,.2)):
    '''
    Gets list of dataframes of partitioned inspired-groups of a superstar
    '''

    superstar = _superstar_
    labels = [str("%.2f" % (bound)) + '-' + str("%.2f" % end_bounds[i+1]) for i,bound in enumerate(end_bounds[:-1])]
    d = []
    papers = []
    dummy_ranked = self.ranked_ss_cit_network[(self.ranked_ss_cit_network['superstar'] == superstar)]
    for i,bound in enumerate(end_bounds):
        if i != len(end_bounds)-1:

            inspirators = dummy_ranked[(dummy_ranked['percentile_'+group_perc] > bound) & (dummy_ranked['percentile_'+group_perc] <= (end_bounds[i+1]))]['inspirators'].to_list()
            
            cited_year = []
            citing_year = []
            citing_doi = []
            cited_doi = []
            superstar_doi = []
            t_0 = []
            t_0_citing = []

            for auth in inspirators:
                dummy_ss = self.citauth_ss_citdocs_dict_noss[auth][superstar]['ss_doc']
                dummy_citing = self.citauth_ss_citdocs_dict_noss[auth][superstar]['citing_doc']
                citing_doi += dummy_citing
                cited_doi += dummy_ss

                dummy_cited_yr = [self.doc_year_dict[x] for x in dummy_ss]
                dummy_citing_yr = [self.doc_year_dict[x] for x in dummy_citing]

                cited_year += dummy_cited_yr
                citing_year += dummy_citing_yr

                t_0 += [x - self.authorid_startyr_dict[superstar] for x in dummy_cited_yr]
                t_0_citing += [x - self.authorid_startyr_dict[superstar] for x in dummy_citing_yr]


            dummy_cit = pd.DataFrame(data = {'superstar' : superstar, 'superstar_doi' : cited_doi, 'citing_doc':citing_doi,
                                            'cited_year' : cited_year, 'citing_year':citing_year})
            #,'t_0':t_0,'t_0_citing':t_0_citing})
            d.append(dummy_cit.groupby('citing_doc').head(1))

    return d

def cit_distr_Inspirators_groups_all(self,_superstar_,group_perc = 'inspirators',end_bounds = np.arange(0,1.001,.2),_sw_ = 1,ranking = False):
    cols = ['Shannon Entropy','Innovation','Disruption Index','Citation Similarity']


    sw = _sw_

    #inf_avg_sw = [[],[]]
    inf_avg_sw = [[] for x in range(len(end_bounds)-1)]
    doc_sim_ = [[] for x in range(len(end_bounds)-1)]
    insp_cit_sim = [[] for x in range(len(end_bounds)-1)]
    num_papers = [[] for x in range(len(end_bounds)-1)]
    prev_papers = [[] for x in range(len(end_bounds)-1)]
    dummypprs = [[] for x in range(len(end_bounds)-1)]
    perc_cit_top_sspapers = [[] for x in range(len(end_bounds)-1)]
    inspired_cit = [defaultdict(list) for x in range(len(end_bounds)-1)]


    superstar = _superstar_

    dummy = get_ss_insp_dfs_groups_noss(self,superstar,group_perc,end_bounds)
    start = timeit.default_timer()
    years = []
    for d in dummy:
        if d.shape[0]!=0:
            first_year = self.authorid_startyr_dict[superstar]
            years = np.arange(0,(2025-first_year),1)
            break
    stop = timeit.default_timer()
    #print('Sec1:',stop-start,'(s)')
    ###############################################
    
    lbls = ['Shannon Entropy','Disruption Index','Innovation']#,'Citation Similarity']
    dicts = [self.doc_shanent,self.doc_disrupt,self.doc_innov]#,self.doc_citsim]
    dois = self.docs
    start = timeit.default_timer()
    
    t0cit_citdoi_dict = []
    for k,group in enumerate(dummy):
        dummy2 = defaultdict(list)
        citdoi = dummy[k]['citing_doc'].to_list()
        dum_t0 = [self.doc_year_dict[x] - self.authorid_startyr_dict[superstar] for x in citdoi]
        for i,t0cit in enumerate(dum_t0):
            dummy2[t0cit].append(citdoi[i])
        t0cit_citdoi_dict.append(dummy2)
    del dummy2
    
    stop = timeit.default_timer()

    #################################################    

    def get_top_superstar_doi_ordered(ss,year_):
        pprs = [doi for doi in self.authorid_docs_dict[ss] if self.doc_year_dict[doi] <= year_]
        cit_count = []
        for ppr in pprs:
            cit_count.append(len([doi for doi in self.citations_dict[ppr] if ((self.doc_year_dict[doi] <= year_) and (self.doc_year_dict[doi] != 0))]))
        return ([x for _, x in sorted(zip(cit_count, pprs))][::-1])

    def get_inf_and_sim(dummy_lst):
        
        inf_0 = [float('nan') for x in range(len(lbls)+1)]
        if len(dummy_lst) == 0:
            doc_0 = [float('nan')]
        elif len(dummy_lst) == 1:
            doc_0 = [float('nan')]
        else:
            doc_0 = get_all_sim(self,dummy2)
        
        j = 0
        for i in range(len(lbls)):
            if len(dummy_lst) == 0:
                inf_0[j] = float('nan')
            else:
                inf_0[j] = np.nanmean([dicts[i][doi] for doi in dummy2])
                
            if lbls[i] == 'Disruption Index':
                j+=1
                if len(dummy_lst) == 0:
                    inf_0[j] = float('nan')
                else:
                    inf_0[j] = np.nanmax([dicts[i][doi] for doi in dummy2])
            j+=1

        return inf_0,doc_0

    ##################################################
    start = timeit.default_timer()
    for i,year in enumerate(years):
        for j in range(len(dummy)):
            dummy_papers = []
            dummy2 = []
            if i < len(years):
                dummypprs[j].append([t0cit_citdoi_dict[j][x] for x in np.arange(years[i]-(sw-1),years[i]+(1),1) if x >= 0])
                dummy2 = list(chain.from_iterable(dummypprs[j][-1]))
                prev_papers[j].append(dummy2)
                num_papers[j].append(len(dummy2))

                dummy_inf_sim = get_inf_and_sim(dummy2)

                inf_avg_sw[j].append(dummy_inf_sim[0])
                doc_sim_[j].append(dummy_inf_sim[1])
                for k,pprs in enumerate(dummypprs[j][::-1]):
                    # list of citation counts for ppr, k-years after publication
                    inspired_cit[j][k] += [self.doc_yr_cit_dict[doi][first_year+i] for doi in pprs[-1] if len(self.doc_yr_cit_dict[doi]) != 0]

    stop = timeit.default_timer()
    start = timeit.default_timer()
    inspired_cit_mean = [defaultdict(float) for x in end_bounds]
    for j in range(len(dummy)):
        for k in inspired_cit[j].keys():
            inspired_cit_mean[j][k] = np.nanmean(inspired_cit[j][k])
            
    stop = timeit.default_timer()
    
    return inf_avg_sw,num_papers,doc_sim_,inspired_cit_mean,perc_cit_top_sspapers,prev_papers,dummypprs #,insp_cit_sim

"""
def ingroup_df_3(self,data, metric,end_bounds = np.arange(0,1.001,.2),users = [],_sw_=1,ranking = False):
    print('Make dataframe of in group similarities for seaborn')
    if users == []:
        users = self.superstars
    dfs_low = []
    dfs_high = []
    dfs = [[] for i,y in enumerate(users)]
    labels = [str("%.2f" % (bound)) + '-' + str("%.2f" % end_bounds[i+1]) for i,bound in enumerate(end_bounds[:-1])]
    for ss in range(len(data)):
        for i in range(len(data[ss][0])):
            dfs[ss].append(pd.DataFrame(data = {'SS' : users[ss],'t_0' : range(len(data[ss][0][i])),
                    metric:data[ss][0][i],'Num Papers' : data[ss][1][i], 'similarities':[np.nanmean(x) for x in data[ss][2][i]],
                    'Inspired Citations':list(data[ss][3][i].values()),#'Percent Top 3 SS Citations':data[ss][4][i],
                                               'Sliding Window':_sw_})) #,'2nd Order Inspired Similarities':data[ss][7][i]}))


    dfs2 = []
    for i,group in enumerate(labels):
        dum = [dfs[ss][i] for ss in range(len(dfs))]
        dfs2.append(pd.concat(dum))
        dfs2[i]['group'] = group

    if metric == 'similarities':
        for i,df in enumerate(dfs2):
            dfs2[i] = dfs2[i].explode(metric)
            dfs2[i] = dfs2[i][dfs2[i][metric].notna()]

    iginf_dfs_2 = pd.concat(dfs2).reset_index().drop(columns = 'index')
    iginf_dfs_2['Year Range'] = float('nan')
    for i in range(5):
        lbl = str(i*10)+'-'+str((i+1)*10)+' years'
        if i == 4:
            lbl = '>='+str(i*10)+' years'
            iginf_dfs_2.loc[(iginf_dfs_2['t_0'] >= (i*10)),'Year Range'] = lbl
        else:
            iginf_dfs_2.loc[(iginf_dfs_2['t_0'] >= (i*10)) & (iginf_dfs_2['t_0'] < ((i+1)*10)),'Year Range'] = lbl
    return iginf_dfs_2

def get_mult_inf_cit(self,users,metric = 'Recombinant Novelty',group_perc = 'inspirators',end_bounds = np.arange(0,1.001,.2),_sw_ = 1,ranking = False):
    in_group_inf = []
    start1 = timeit.default_timer()
    start2 = timeit.default_timer()
    col = metric
    num_iterations = 10
    for superstar in tqdm(users):
        in_group_inf += [cit_distr_Inspirators_groups(self,superstar,col,group_perc,end_bounds,_sw_,ranking)]
            
    stop = timeit.default_timer()


    return in_group_inf

def get_inspiree_df(self, metric = 'Recombinant Novelty', partitioning_cat = 'inspirators', end_bounds = np.array([0,.33,.66,1]),_sw_ = 1, ranking = False, inf_uni_mean_of_dists = True):
    '''
    Inspirators - Person_df
    citations - df
    '''
    self.inf_uni_mean_of_dists = inf_uni_mean_of_dists
    col = check_metric(self,metric)
    if col == 0:
        return 0
    if not hasattr(self,'doc_yr_cit_dict'):
        self.generate_doc_yr_cit_dict()

    self_usrs = self.superstars
    by_category = get_mult_inf_cit(self,self_usrs,metric,partitioning_cat,end_bounds,_sw_,ranking)
    df = ingroup_df_3(self,by_category, col,end_bounds,self_usrs,_sw_,ranking)

    return df

"""
####################################################################################################

def get_inspiree_df_all(self, partitioning_cat = 'inspirators', end_bounds = np.array([0,.33,.66,1]),_sw_ = 1, ranking = False, inf_uni_mean_of_dists = True):
    if not hasattr(self,'doc_yr_cit_dict'):
        self = generate_doc_yr_cit_dict(self)

    self_usrs = self.superstars

    print('Make dataframe of in group similarities for seaborn')

    dfs_low = []
    dfs_high = []
    # thresh is set for memory allocation. After 200
    thresh = 200
    thresh = 10**6
    #dfs = [[] for i in range(thresh)]
    dfs = [[]]
    labels = [str("%.2f" % (bound)) + '-' + str("%.2f" % end_bounds[i+1]) for i,bound in enumerate(end_bounds[:-1])]
    nov_lbls = ['Shannon Entropy','Disruption Index','Max Disruption Index','Innovation','Citation Similarity']
    nov_lbls = ['Shannon Entropy','Disruption Index','Max Disruption Index','Innovation']

    j = 0
    k = 0
    dfs_large = []
    for ss in tqdm(self_usrs, desc = "Getting Group Stats Dataframe for All Novelties"):
        data = cit_distr_Inspirators_groups_all(self,ss,partitioning_cat,end_bounds,_sw_,ranking)
        novelty_data = data[0]
        for i in range(len(novelty_data)):
            if len(novelty_data[i])!=0:
                d = [list(x) for x in zip(*novelty_data[i])]
                d1 = {'SS' : ss,'t_0' : range(len(data[0][i])),
                        'Num Papers' : data[1][i], 'similarities':[np.nanmean(x) for x in data[2][i]],
                        'Inspired Citations':list(data[3][i].values())}
                d2 = {nov_lbls[l]:d[l] for l in range(len(nov_lbls))}
                d1.update(d2)
                dfs[j] += [pd.DataFrame(data = d1)] #,'2nd Order Inspired Similarities':data[ss][7][i]}))
            else:
                d1 = {'SS' : ss,'t_0' : float('nan'),
                        'Num Papers' : float('nan'), 'similarities':float('nan'),
                        'Inspired Citations':float('nan')}
                d2 = {nov_lbls[l]:[float('nan')] for l in range(len(nov_lbls))}
                d1.update(d2)
                dfs[j] += [pd.DataFrame(data = d1)]
        dfs.append([])
        j+=1
        if j >= thresh:
            dfs_large+= dfs[:-1]
            k+=1
            dfs = [[]]
            #dfs = [[] for i in range(thresh)]
            j = 0
    #dfs_large+=dfs
    #dfs_large+=dfs[:len(self_usrs)%thresh]
    dfs_large+=dfs[:-1]
    print('Assembling DataFrame')
    start = timeit.default_timer()
    dfs2 = []
    for i,group in enumerate(labels):
        dum = [dfs_large[ss][i] for ss in range(len(dfs_large))]
        dfs2.append(pd.concat(dum))
        dfs2[i]['group'] = group

    iginf_dfs_2 = pd.concat(dfs2).reset_index().drop(columns = 'index')
    iginf_dfs_2['Year Range'] = float('nan')
    for i in range(5):
        lbl = str(i*10)+'-'+str((i+1)*10)+' years'
        if i == 4:
            lbl = '>='+str(i*10)+' years'
            iginf_dfs_2.loc[(iginf_dfs_2['t_0'] >= (i*10)),'Year Range'] = lbl
        else:
            iginf_dfs_2.loc[(iginf_dfs_2['t_0'] >= (i*10)) & (iginf_dfs_2['t_0'] < ((i+1)*10)),'Year Range'] = lbl
            
    stop = timeit.default_timer()
    print(stop-start,'(s)')
    return iginf_dfs_2

def prep_ranked_network_10(self):
    ''' Setting up percentiles for person-level partitions and paper-level partitions'''
    print('Prepping Ranked Network')
    auths_10 = self.authorstats_df[self.authorstats_df['Num Papers'] >= 10]['Author'].to_list()
    self.ranked_ss_cit_network = self.ranked_ss_cit_network[self.ranked_ss_cit_network['inspirators'].isin(auths_10)]
    
    self.ranked_ss_cit_network = self.ranked_ss_cit_network.sort_values(by = ['superstar','num_citations','inspirators'], ascending = (True,False,True)).reset_index().drop(columns = 'index')
    inspirator_counts = self.ranked_ss_cit_network.groupby('superstar').count()['inspirators'].to_list()
    self.ranked_ss_cit_network['percentile_inspirators'] = 1
    self.ranked_ss_cit_network['dummy'] = 0
    self.ranked_ss_cit_network.loc[self.ranked_ss_cit_network.groupby('superstar').head(1).index,'dummy'] = inspirator_counts
    self.ranked_ss_cit_network['dummy'] = self.ranked_ss_cit_network.groupby('superstar').cumsum()['dummy'].to_list()
    self.ranked_ss_cit_network['percentile_inspirators'] = self.ranked_ss_cit_network.groupby('superstar').cumsum()['percentile_inspirators'].to_list()
    self.ranked_ss_cit_network['percentile_inspirators'] = self.ranked_ss_cit_network['percentile_inspirators']/self.ranked_ss_cit_network['dummy']
    self.ranked_ss_cit_network = self.ranked_ss_cit_network.drop(columns = 'dummy')

    self.ranked_ss_cit_network = self.ranked_ss_cit_network.sort_values(by = ['superstar','proportional_citations','inspirators'], ascending = (True,False,True))
    ss_citation_counts = self.ranked_ss_cit_network.groupby('superstar').count()['proportional_citations'].to_list()
    self.ranked_ss_cit_network['dummy'] = 0
    self.ranked_ss_cit_network['percentile_citations'] = 1
    self.ranked_ss_cit_network['percentile_citations'] = self.ranked_ss_cit_network.groupby('superstar').cumsum()['percentile_citations'].to_list()
    self.ranked_ss_cit_network.loc[self.ranked_ss_cit_network.groupby('superstar').head(1).index,'dummy'] = ss_citation_counts
    self.ranked_ss_cit_network['dummy'] = self.ranked_ss_cit_network.groupby('superstar').cumsum()['dummy'].to_list()
    self.ranked_ss_cit_network['sum_citations'] = self.ranked_ss_cit_network.groupby('superstar').cumsum()['num_citations'].to_list()
    self.ranked_ss_cit_network['percentile_citations'] = self.ranked_ss_cit_network['percentile_citations']/self.ranked_ss_cit_network['dummy']
    self.ranked_ss_cit_network = self.ranked_ss_cit_network.drop(columns = 'dummy')  
    
    print('FINISHED')
    
    return self

def generate_groupstats_df(self):
    self = set_norm_DTC(self)
    self = get_ss_cit_network_2(self)
    self = prep_ranked_network_10(self)
    self.groupstats_df = get_inspiree_df_all(self,'citations',np.array([0,.1,.25,.4,.65,1]),3)
    return self

def save_groupstats_df(self,ext = False):
    ext_lbl = ''
    if ext == True:
        ext_lbl = self.ext
    if hasattr(self,'groupstats_df'):
        print('Saving groupstats_df')
        pd.to_csv('../bin/Fields/'+self.field+ext_lbl+'/groupstats_df.csv')
    else:
        print('Object does not have groupstats_df')




