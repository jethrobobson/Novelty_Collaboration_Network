#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

import seaborn as sns


# In[ ]:


def set_norm_DTC(self):
    start = timeit.default_timer()
    if hasattr(self,'normDTC'):
        if len(self.normDTC) < len(list(self.doc_embeddings_dict.values())):
            print('Normalization started but not yet completed')
            print(str(len(self.normDTC)) + ' out of ' + str(len(self.doc_embeddings_dict)) + ' remaining')
            self.normDTC = defaultdict(list)
        else:
            print('Normalization Already Completed')
            return self
    else:
        print('Beginning l2 Normalizaiton of Document Embeddings') 
        self.normDTC = defaultdict(list)

    for doc in tqdm(self.docs, desc = "Getting norm DTC"):
        self.normDTC[doc] = self.doc_embeddings_dict[doc]/ np.linalg.norm(self.doc_embeddings_dict[doc])
    stop = timeit.default_timer()
    print('Completed Normalization of DTC:',(stop-start),'(s)')
    return self

#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#################### SIMILARITY CALCS ###############################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################

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

def create_nan_defaultdict(self):
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
                                                       'superstar_doi':citedpapers,'citing_doi':citingpapers})]

    self.ss_cit_network = pd.concat(ss_cit_network).reset_index()
    print('DONE!')

    # GET ALL AUTHORS OF CITING PAPERS
    print('Get All Authors of Citing Papers')
    self.ss_cit_network['citing authors'] = float('nan')
    self.ss_cit_network = self.ss_cit_network.sort_values(by = 'citing_doi')
    citing_doi_counts = self.ss_cit_network.groupby('citing_doi').count()['superstar'].to_list()
    authorid_list = []
    i = 0
    for citing_paper in tqdm(self.ss_cit_network.groupby('citing_doi').head(1)['citing_doi'].to_list(),dec = 'getting authorid list'):
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
    citdocs = self.ss_cit_network['citing_doi'].to_list()
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

            inspirators = dummy_ranked[(dummy_ranked['percentile_'+group_perc] > bound) & (dummy_ranked['percentile_'+group_perc] < (end_bounds[i+1]))]['inspirators'].to_list()
            
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


            dummy_cit = pd.DataFrame(data = {'superstar' : superstar, 'superstar_doi' : cited_doi, 'citing_doi':citing_doi,
                                            'cited_year' : cited_year, 'citing_year':citing_year})
            #,'t_0':t_0,'t_0_citing':t_0_citing})
            d.append(dummy_cit.groupby('citing_doi').head(1))

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
        citdoi = dummy[k]['citing_doi'].to_list()
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
                    inspired_cit[j][k] += [self.doi_yr_cit_dict[doi][first_year+i] for doi in pprs[-1] if len(self.doi_yr_cit_dict[doi]) != 0]

    stop = timeit.default_timer()
    start = timeit.default_timer()
    inspired_cit_mean = [defaultdict(float) for x in end_bounds]
    for j in range(len(dummy)):
        for k in inspired_cit[j].keys():
            inspired_cit_mean[j][k] = np.nanmean(inspired_cit[j][k])
            
    stop = timeit.default_timer()
    
    return inf_avg_sw,num_papers,doc_sim_,inspired_cit_mean,perc_cit_top_sspapers,prev_papers,dummypprs #,insp_cit_sim

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
    if not hasattr(self,'doi_yr_cit_dict'):
        self.generate_doi_yr_cit_dict()

    self_usrs = self.superstars
    by_category = get_mult_inf_cit(self,self_usrs,metric,partitioning_cat,end_bounds,_sw_,ranking)
    df = ingroup_df_3(self,by_category, col,end_bounds,self_usrs,_sw_,ranking)

    return df


####################################################################################################

def get_inspiree_df_all(self, partitioning_cat = 'inspirators', end_bounds = np.array([0,.33,.66,1]),_sw_ = 1, ranking = False, inf_uni_mean_of_dists = True):
    if not hasattr(self,'doi_yr_cit_dict'):
        self = generate_doi_yr_cit_dict(self)

    self_usrs = self.superstars

    print('Make dataframe of in group similarities for seaborn')

    dfs_low = []
    dfs_high = []
    thresh = 200
    dfs = [[] for i in range(thresh)]
    labels = [str("%.2f" % (bound)) + '-' + str("%.2f" % end_bounds[i+1]) for i,bound in enumerate(end_bounds[:-1])]
    nov_lbls = ['Shannon Entropy','Disruption Index','Max Disruption Index','Innovation','Citation Similarity']
    nov_lbls = ['Shannon Entropy','Disruption Index','Max Disruption Index','Innovation']
    
    j = 0
    k = 0
    dfs_large = []
    for ss in tqdm(self_usrs, desc = "Getting Group Stats Dataframe for All Novelties"):
        data = cit_distr_Inspirators_groups_all(self,ss,partitioning_cat,end_bounds,_sw_,ranking)
        novelty_data = data[0]
        
        for i in range(len(data[0])):
            d = [list(x) for x in zip(*novelty_data[i])]
            d1 = {'SS' : ss,'t_0' : range(len(data[0][i])),
                    'Num Papers' : data[1][i], 'similarities':[np.nanmean(x) for x in data[2][i]],
                    'Inspired Citations':list(data[3][i].values())}
            d2 = {nov_lbls[k]:d[k] for k in range(len(nov_lbls))}
            d1.update(d2)
            dfs[j] += [pd.DataFrame(data = d1)] #,'2nd Order Inspired Similarities':data[ss][7][i]}))
            
        j+=1
        if j >= thresh:
            dfs_large+= dfs
            k+=1
            dfs = [[] for i in range(thresh)]
            j = 0
    dfs_large+=dfs[:len(self_usrs)%thresh]
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

