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

#from bs4 import BeautifulSoup
#import requests 
#import urllib.request as urllib2
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

import nltk
from nltk.corpus import stopwords
import string
import langdetect

nltk.download('wordnet')
nltk.download('omw-1.4')


#import snap
import itertools



class dataset:
    def __init__(self,field = 'Geography',minyr = ''):
        self.field = field
        
        if os.path.isfile("../dataset/Fields/"+field+"/metadata.pickle"):
            self.metadata_to_pd()
        else:
            print('No metadata found')
        print('Reading In citation graph')
        with open('../dataset/Fields/'+self.field+'/citations_dict.pickle', 'rb') as f:
            dummy = pickle.load(f)
        setattr(self,'citations_dict',dummy)
  
        if not os.path.isdir("../bin/Fields"):
            os.mkdir("../bin/Fields")
            
        if not os.path.isdir("../bin/Fields/"+field):
            print('Creating directory for',field)
            os.mkdir("../bin/Fields/"+field)
            
        
        self.read_dict('doc_abstracts_dict')
        if not hasattr(self,'doc_abstracts_dict'):
            self.doc_abstracts_dict = defaultdict(str)
            dummy = self.DDBB[self.DDBB['paperAbstract'].notna()]
            docs = dummy['id'].to_list()
            abstracts = dummy['paperAbstract'].to_list()
            for doc,abs_ in zip(docs,abstracts):
                self.doc_abstracts_dict[doc] = abs_
            
        self.documents_unfinished = False
        self.read_dict('eng_docs_checked')
        if not hasattr(self,'eng_docs_checked'):
            self.eng_docs_checked = False
        self.read_dict('docs_processed')
        if hasattr(self,'docs_processed'):
            self.docs_processed = True
            #self.read_dict('documents')
        else:
            self.docs_processed = False
        self.get_stopwords()
        
    ################################################################################
    def metadata_to_pd(self):
        print('Reading In metadata')
        with open('../dataset/Fields/'+self.field+'/metadata.pickle', 'rb') as f:
            dummy = pickle.load(f)
        setattr(self,'metadata',dummy)
        self.doc_concepts_dict = defaultdict(list)
        self.doc_lowest_level_dict = defaultdict(int)
        ids = []
        dois = []
        years = []
        authors = []
        abstracts = []
        refs = []
        for work,data in tqdm(self.metadata.items()):
            ids += [work]
            dois += [data['doi']]
            years += [data['year']]
            authors += [data['authors']]
            if 'abstract' in data.keys():
                abstracts += [data['abstract']]
            else:
                abstracts += [None]
            refs += [data['references']]
            if 'concepts' in data.keys():
                if len(data['concepts']) > 0:
                    #print(data['concepts'])
                    #self.concepts_dict[work] = [{'display_name': x['display_name'], 
                    #                                    'level':x['level'],'score':x['score']} for x in data['concepts']]
                    self.doc_concepts_dict[work] = list(chain.from_iterable([[{'concept': c, 'level':int(k[3:]),'score':s} for c,s in v.items()] for k,v in data['concepts'].items()]))
                    self.doc_lowest_level_dict[work] = max([x['level'] for x in self.doc_concepts_dict[work]]+[0])

        self.DDBB = pd.DataFrame(data = {'id':ids,'doi':dois,'year':years,'author':authors, \
                                          'paperAbstract':abstracts,'references':refs})

        self.DDBB = self.DDBB[(~self.DDBB['paperAbstract'].isnull()) & (self.DDBB['author'].str.len() > 0)]
        
        if not os.path.isfile('../bin/Fields/'+self.field+'/doc_concepts_dict.pickle'):
            self.write_dict(self.doc_concepts_dict,'doc_concepts_dict')
            self.write_dict(self.doc_lowest_level_dict,'doc_lowestlevel_dict')
        #ids = self.DDBB['id'].to_list()
        #rec_abs = [ids[i] for i,x in enumerate(self.DDBB['paperAbstract'].to_list()) if x[:8] == 'Received']
        #self.DDBB[self.DDBB['id'].isin(rec_abs)]['paperAbstract'].to_list()

        #del self.metadata
            
    ################################################################################
            
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
        
    def get_stopwords(self):
        ('Downloading stopwords for document processing')
        nltk.download('stopwords')
        stop_words_ = stopwords.words('english')
        #clean_stop = [' d ',' ll ',' m ',' o ',' re ',' ve ',' y ']
        #stop_words_ = [x.replace(x,(' '+x+' ')) for x in stop_words_]
        non_stop = ['up','down','off','each']

        stop_words_ = list(set([x for x in stop_words_ if x not in non_stop]))

        self.stopwords_ = stop_words_



###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
############################### PROCESS DOCUMENTS  AND MINE PHRASES IN CORPUS #############################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################




def get_english_abstracts(self):
    if self.eng_docs_checked == False:
        es3 = self.DDBB[self.DDBB['paperAbstract'].notna()]
        es3 = es3.reset_index()
        abstracts = es3['paperAbstract'].to_list()
        ids = es3['id'].to_list()
        id_abs_dict = defaultdict(str)
        for i,doc in enumerate(tqdm(abstracts,desc="Removing non-english abstracts from Documents…",miniters = 150,mininterval = .5)):
            doc_0 = doc.replace('.','')
            if (len(doc_0)>100):
                try:
                    lang = langdetect.detect(doc_0)
                except:
                    lang = "error"
                    print("This abstract throws an error:", i)
                if (lang == 'en'):
                    id_abs_dict[ids[i]] = doc_0
        self.doc_abstracts_dict = id_abs_dict
        self.write_dict(self.doc_abstracts_dict,'doc_abstracts_dict')
        self.write_dict(True,'eng_docs_checked')
        self.eng_docs_checked = True
    
    return self

def get_stopwords(self):
    nltk.download('stopwords')
    stop_words_ = stopwords.words('english')
    #clean_stop = [' d ',' ll ',' m ',' o ',' re ',' ve ',' y ']
    #stop_words_ = [x.replace(x,(' '+x+' ')) for x in stop_words_]
    non_stop = ['up','down','off','each']

    stop_words_ = list(set([x for x in stop_words_ if x not in non_stop]))

    self.stopwords_ = stop_words_
    return self


def process_documents(self):

    if self.docs_processed == True:
        print('Documents already processed')
    else:
        if (self.eng_docs_checked == False):
            print('Getting English Abstracts')
            #self.get_english_abstracts()
            get_english_abstracts(self)

            
        self.documents = list(self.doc_abstracts_dict.values())
        doc_ids = list(self.doc_abstracts_dict.keys())

        def addspace(x):
            return ' ' + x + ' '

        print('Beginning Processing Documents')

        #self.get_stopwords()

        puncs = ['Published by the','This article is available','under the terms of the Creative Commons Attribution','3.0 License.',
         '4.0 International license',
      'distribution of this work must maintain attribution','to the author\\(s\\) and the published article',
      's title, journal citation, and DOI.','\\\\phantom','\\\\rule','em\\}','ex\\}','\\\\fi','\\\\textpm','\\\\pm',
       'American Physical Society','\\\\text','\\\\textdegree','\\\\circ',
      'The authors of the Letter offer a Reply','American Physical Society','Physics Subject Headings','Research Areas',
        'doi.org','DOI:','\\(PhySH\\)','Further','Received','Accepted','Revised',
        'January','February','March','April','May','June','July','August','September','October','November','December']
        
        puncs += ["This is the final version - click for previous version","MoreSectionsPDF",
        "Download PDFDownload","PDFPlus","ToolsExport citationAdd to favoritesGet permissionsTrack citations Share",
        "Share onFacebookTwitterLinkedInEmail","Published Online","FiguresReferencesRelatedInformation","Back to Top Next",
        "More from this issue","Copyright & PermissionsCopyright","American Physiological Society","Keywords","PDF download",
        "PubMed | ISI | Google Scholar","Crossref","textperiodcentered","textperiodcentered","Licensed under Creative Commons Attribution",
        "mathit","Am J Physiol Lung Cell Mol Physiol","Am J Physiol Regul Integr Comp Physiol","Am J Physiol","Renal Fluid Electrolyte Physiol"]
        puncs += ['These metrics are regularly updated to reflect usage leading up to the last few days','Citations are the number of other articles citing this article',
         'calculated by Crossref and updated daily','Find more information about Crossref citation counts',
         'The Altmetric Attention Score is a quantitative measure of the attention that a research article has received online',
         'Clicking on the donut icon will load a page at altmetric.com with additional details about the score and the social media presence for the given article',
         'Find more information on the Altmetric Attention Score and how the score is calculated',
         'RIGHTS & PERMISSIONSArticle Views','Altmetric-Citations',
        'ExportRISCitationCitation','abstractCitation','referencesMore Options','Share onFacebookTwitterWechatLinked InReddit',
        'LEARN ABOUT THESE METRICS','Article Views are the COUNTER-compliant sum of full text article downloads since','(both PDF and HTML) across all institutions and individuals',
        'RETURN TO ISSUEPREVNewsNEXT','This publication is available under these Terms of Use','Request reuse permissions This publication is free to access through this site',
        'Get e-Alerts','American Chemical Society','Publication History Published online']

        
        for punc in tqdm(puncs,desc = "Removing Spurious Scraped Phrases"):
            for i,doc in enumerate(self.documents):
                self.documents[i] = re.sub(punc,"",doc)
        
        
        self.documents = list(map(str.lower,self.documents))
        self.documents = list(map(addspace,self.documents))
        #self.documents = [re.sub("s+"," ", doc) for doc in self.documents]
                
        self.documents = ["".join([i if i not in string.punctuation else ' ' for i in doc]) for doc in self.documents]
        rmv_spec_punctuation = ["“","”","‘","’","©","°","±","¯","≤","−",'اطلاع'
                                ,'اطلاعات','رسانی','فناوری','في','مرکز','و','کشاورزی','\u200b\u200bthe','‐','‐year',
                                '‒','–','–cm','–n','–p','–‰','—','—the','―','„','†','‡','•','…','‰','′','′e',
                                '′n','′s','′w','″','⁄','€','℃the','ⅰ','ⅱ','ⅲ','ⅳ','ⅴ','→','∑','∑pah','∕','∗',
                                '∘','∶','∼','∼–','≈','≥','①','②','③','■','●','◦','◦c','⩽','⩾','、',
                                '【objective】','것으로','년','대한','및','본','수','있다',u'\ue010',u'\uf0a7',
                                u'\uf0b7',u'\uf6d9',u'\ufeff','～','～cm','�','•','�','“','”','“','”',
                               'http', 'doiorg', 'physrevlett©','mathcal','textasciimacron',
                               'ensuremath','mathrm','mathbf','mathit','texttimes','ifmmode','textpm','else','http']
        self.documents = ["".join([i if i not in string.punctuation else ' ' for i in doc]) for doc in self.documents]
        for punc in tqdm(rmv_spec_punctuation,desc = "Removing special punctuation"):
            for i,doc in enumerate(self.documents):
                self.documents[i] = re.sub(punc,"",doc)
        self.documents = [re.sub(r'[0-9]',"",doc) for doc in self.documents]


        print('Tokenizing Words')
        docs = [nltk.tokenize.word_tokenize(doc) for doc in self.documents]
        docs = [[i for i in doc if i not in self.stopwords_] for doc in docs]
        #indices = [i for i in np.arange(0,len(docs),1) if len(docs[i]) >= 5]
        #docs = [doc for doc in docs if len(doc) >= 5]
        ps = nltk.PorterStemmer()
        print('Lemmatizing')

        docs = [' '.join([ps.stem(word) for word in doc]) for doc in docs]
        self.documents = docs

        print("documents count:",len(self.documents),"\n")
        print('Writing processed documents to pickle')

        self.doc_abstracts_dict = defaultdict(str)
        for id_,doc in zip(doc_ids,self.documents):
            self.doc_abstracts_dict[id_] = doc

        self.write_dict(self.doc_abstracts_dict,'doc_abstracts_dict')
        self.write_dict(self.documents,'documents')
        self.write_dict(True,'docs_processed')
    return self

###########################################################################################################################


def freq_phrase_mining(self, threshold = 300 ,processed = True):

    print('Begin Frequent Phrase Mining')
    if processed == False:
        print('Processing Documents \n')
        self.process_documents()
        print('Processing Complete')

    split_docs = list(map(str.split,list(self.doc_abstracts_dict.values())))
    e = threshold
    documents2 = split_docs.copy()
    
    if os.path.isfile('../bin/Fields/'+self.field+'/phrasecounts_df.csv'):
        print('Reading phrasecounts_df')
        self.phrasecounts_df = pd.read_csv('../bin/Fields/'+self.field+'/phrasecounts_df.csv')
        self.num_unigrams = self.phrasecounts_df[self.phrasecounts_df['count']==1].shape[0]
        n = self.phrasecounts_df.tail(1)['word length'].to_list()[0]
        print('Length of last phrases computed',str(n))
        word_counts_dict = defaultdict(int)
        words_ = self.phrasecounts_df['word'].to_list()
        counts_ = self.phrasecounts_df['count'].to_list()
        for word,count in tqdm(zip(words_,counts_),desc = "Getting Word Counts after Threshold"):
            word_counts_dict[word]=count
        del words_
        del counts_
        
        
    else:
        print('Getting List of all words')

        

        words = list(itertools.chain.from_iterable(split_docs))
        word_counts_dict = defaultdict(int)
        for word in tqdm(words,desc = "Getting Initial Word Counts"):
            word_counts_dict[word]+=1
        words_df = pd.DataFrame(data = {'word':list(word_counts_dict.keys()),'count':list(word_counts_dict.values()),'word length':1})
        
        self.phrasecounts_df = words_df[words_df['count']>=e]
        self.num_unigrams = self.phrasecounts_df.shape[0]
        del words_df
        del words
        n = 1

        
    start1 = timeit.default_timer()
    start2 = timeit.default_timer()
    print('BEGIN PHRASE MINING \n \n')
    print("documents2[0]:",documents2[0])
    while documents2 != []:
        print('Working on Phrases of length: ' + str(n+1))
        print(str(len(documents2)) + ' documents remaining \n')
        
        new_phrases = defaultdict(int)
        for doc in tqdm(documents2,desc="Mining Documents…",miniters = 10000,mininterval = 2):
            docstr = doc
            
            if len(docstr) <=1:
                #self.documents.remove(doc)
                documents2.remove(doc)
            else:
                if n!= 1:
                    docstr = [' '.join(docstr[j:j+n]) for j in range(len(docstr)-n)]
                A_d = (np.nonzero([x if word_counts_dict[x] >= e else 0 for x in docstr])[0])
                if (len(A_d) <= 1):
                    documents2.remove(doc)
                else:
                    for j in A_d:
                        if j+1 in A_d:
                            new_phrases[docstr[j] + ' ' + str.split(docstr[j+1])[-1]] += 1

        print('Finished iteration mining words')
        n+=1
        dummy_words = pd.DataFrame(data = {'word':list(new_phrases.keys()),'count':list(new_phrases.values()),'word length':n})
        dummy_words = dummy_words.sort_values(by = 'word')

        self.phrasecounts_df = pd.concat([self.phrasecounts_df,dummy_words[dummy_words['count']>=e]])

        words_ = dummy_words['word'].to_list()
        counts_ = dummy_words['count'].to_list()
        for word,count in zip(words_,counts_):
            word_counts_dict[word]=count

        stop = timeit.default_timer()
        if documents2 == []:
            print('ALL DOCUMENTS MINED')
        else:
            print("Total Time Elapsed:", str(round((stop-start1)/3600,4)),'hour(s)')
        start2 = timeit.default_timer()
        if n == 10:
            break

        print('Saving words and word stats to pickle and txt files')
        self.phrasecounts_df.to_csv('../bin/Fields/'+self.field+'/phrasecounts_df.csv')
        print('Saved')
    print('Saving words and word stats to pickle and txt files')
    self.phrasecounts_df.to_csv('../bin/Fields/'+self.field+'/phrasecounts_df.csv')
    print('Saved')

    #print('\n\nFINISHED MINING, cleaning phrases \n\n')
    #self.generate_docs_w_phrase_dict()
    #words = []
    #for doc,phrs in tqdm(test2.doc_phrases_dict.items()):
    #    words += phrs
    #words = set(list(words))
    self.phrasecounts_df = self.phrasecounts_df.reset_index()
    
    self.word_counts = word_counts_dict
    self.total_word_count = self.phrasecounts_df.sum()['count']
    print("DONE")
    return self

###################################################################################################################
###################################################################################################################
###################################################################################################################
################# CONSTRUCT PHRASES FOR EACH DOCUMENT BASED ON SIGNIFICANCE OF PHRASES ############################
###################################################################################################################
###################################################################################################################
###################################################################################################################

# Bottom up Construction from Ordered Tokens - Bag-of-Phrases construction
def phrase_construction(self,abstract, alpha_ = 5):
    doc = abstract
    contiguous_pairs = []
    sig_scores = []
    doc0 = doc.split()
    doc = [x if x in self.words else 'XXXXXX' for x in doc0]
    if len(doc) <= 1:
        return [[x for x in doc if x != 'XXXXXX'],sig_scores]
    else:
        for i in range(len(doc) - 1):
            contiguous_pairs.append(doc[i:i+2])
            #sig_scores.append(self.significance_score(contiguous_pairs[i]))
            sig_scores.append(significance_score(self,contiguous_pairs[i]))
        while (True & (sig_scores != [])):
            best_i = np.argmax(sig_scores)
            if sig_scores[best_i] >= alpha_:
                doc0[best_i] = ' '.join(doc0[best_i:best_i+2])
                doc0.pop(best_i+1)

                doc[best_i] = ' '.join(doc[best_i:best_i+2])
                doc.pop(best_i+1)

                if (best_i + 2) <= len(sig_scores):
                    contiguous_pairs[best_i+1] = doc[best_i:best_i+2]
                    sig_scores[best_i+1] = significance_score(self,contiguous_pairs[best_i+1])
                if (best_i) > 0:
                    contiguous_pairs[best_i-1] = doc[best_i-1:best_i+1]
                    sig_scores[best_i-1] = significance_score(self,contiguous_pairs[best_i-1])

                contiguous_pairs.pop(best_i)
                sig_scores.pop(best_i)
            else:
                break


        return [[x for x in doc if x != 'XXXXXX'],sig_scores]

def significance_score(self, Phrases):
    #print(Phrases)
    P1,P2 = Phrases
    if (P1 == 'XXXXXX') or (P2 == 'XXXXXX') or ((P1 + ' ' + P2) not in self.word_counts):
        return 0
    else:
        fp1p2 = self.word_counts[P1 + ' ' + P2]
        if fp1p2 == 0:
            return 0
        else:
            mu_p1p2 = self.word_counts[P1]*self.word_counts[P2]/self.total_word_count
            return (fp1p2 - mu_p1p2)/np.sqrt(fp1p2)

def get_time(start1,start2,stop2,i,elements,items = 'users',num_completed=0):
    print("Time: " + str(round((stop2-start2)/60,3)) + ' min(s)')
    print("Total Time: "  + str(round((stop2-start1)/3600,4)) + ' hour(s)')
    print('Percent finished: ' + str(round(i/(len(elements)),3)*100) + '%')
    print('Time Left: ' + str(round((stop2-start1)*(len(elements)/(i) - 1)/3600,3)) + ' hour(s)')
    print('')
    print(i,' out of ',len(elements),' ',items,' completed')
    print('')



def set_rand_topics(self):
    '''
    After Frequent Phrase Mining of documents, set_rand_topics begins the Gibbs Sampling Algorithm

    We begin by constructing the phrases in each document using the phrase construction algorithm, 
    giving us a bag-of-phrases for that document

    Phrases are randomly assigned topics

    A V x k matrix "self.Vk" (len of vocabulary to number of topics) holds the counts for all words i tagged with a topic j

    Return:
        self.doc_phrases_dict_topics - an array where each element is two arrays: the bag of phrases, and their associated topics
        
    ####
    IF NOT USING FOR SEEDING DOCUMENT TOPICS, USED FOR doc_phrases_dict
    ####

    '''
    self.num_topics = 45
    num_docs = len(list(self.doc_abstracts_dict.values()))
    self.read_dict('doc_phrases_dict')
    self.read_dict('doc_topics_dict')
    self.read_dict('doc_topic_count')
    self.read_dict('Vk')
    if not hasattr(self,'phrasecounts_df'):
        if os.path.isfile('../bin/Fields/'+self.field+'/phrasecounts_df.csv'):
            print('Reading in phrasecounts_df.csv')
            self.phrasecounts_df = pd.read_csv('../bin/Fields/'+self.field+'/phrasecounts_df.csv')
            self.total_word_count = self.phrasecounts_df.sum()['count']
        else:
            print('No phrasecounts_df, Run Frequent Phrase Mining')
            return 0
    self.words = self.phrasecounts_df['word'].to_list()
    self.phr_count = self.phrasecounts_df['count'].to_list()
    self.word_counts = defaultdict()
    for word,count in tqdm(zip(self.words,self.phr_count),desc = "Adding new phrases to word_count_dict..."):
        self.word_counts[word]=count
    self.num_unigrams = self.phrasecounts_df[self.phrasecounts_df['word length']==1].shape[0]
    if not hasattr(self,'doc_phrases_dict'):
        self.doc_phrases_dict = defaultdict(list)
        self.doc_topics_dict = defaultdict(list)
        self.doc_topic_count = defaultdict(list)
        self.Vk = defaultdict(list)
        for i,word in enumerate(self.words):
            self.Vk[word] = [0 for i in range(self.num_topics)]
    self.docsSeeded = len(list(self.doc_phrases_dict.keys()))

    start1 = timeit.default_timer()
    start2 = timeit.default_timer()
    print('Getting Bag-of-Phrases and Setting Topics for each word in each document')
    print('Aggregating word-topic counts over entire corpus')
    num_iterations = 10
    startdoc = self.docsSeeded
    remaining_ids = set(self.doc_abstracts_dict.keys()).difference(set(self.doc_phrases_dict.keys()))
    for doc in tqdm(remaining_ids,desc = 'Setting Phrases and Random Topic Associations to Documents'):
        phrases = phrase_construction(self,self.doc_abstracts_dict[doc])[0]

        #print ('DOC NUMBER : ' + str(self.docsSeeded))
        dummyrand = np.random.rand(1, len(phrases))[0]
        topics = [int(self.num_topics*k) for k in dummyrand]
        jump = 0
        for i,phrase in enumerate(phrases):
            if len(phrase.split()) > 1:
                dummyphrase = phrase.split()
                j=1
                self.Vk[phrase][topics[i]]+=1
                for word in dummyphrase:
                    self.Vk[word][topics[i]]+=1
                    j+=1
            else:
                self.Vk[phrase][topics[i]]+=1

        self.doc_phrases_dict[doc]=phrases
        self.doc_topics_dict[doc] = topics
        dtc_dummy = []

        for word_len,topic in zip(list(map(len,list(map(str.split,phrases)))),topics):

            if word_len == 1:
                dtc_dummy.append(topic)
            else:
                dtc_dummy += [topic for i in range(word_len)]
        self.doc_topic_count[doc]=[dtc_dummy.count(i) for i in range(self.num_topics)]

        self.docsSeeded+=1

        #if (self.docsSeeded-1)%(int(num_docs/num_iterations)) == 0:
        #    print('WRITING INCOMPLETE PARAMS')
        #    self.write_dict(self.doc_phrases_dict,'doc_phrases_dict')
        #    self.write_dict(self.doc_topics_dict,'doc_topics_dict')
        #    self.write_dict(self.doc_topic_count,'doc_topic_count')
        #    self.write_dict(self.Vk,'Vk')
        #    print('FINISHED WRITING')


    self.alpha = [.5 for i in range(self.num_topics)]
    self.beta = [.05 for i in range(self.num_unigrams)]
    self.total_iterations = 0
    self.total_topic_count = np.sum(self.Vk,axis=0)

    self.write_dict(self.doc_phrases_dict,'doc_phrases_dict')
    self.write_dict(self.doc_topics_dict,'doc_topics_dict')
    self.write_dict(self.doc_topic_count,'doc_topic_count')
    self.write_dict(self.Vk,'Vk')
    self.write_dict(self.total_topic_count,'total_topic_count')
    
    return self

########################## POINTWISE-MUTUAL-INFORMATION for INNOVATION MEASURE #############################################

# In[7]:


#start = words_completed

def generate_docs_w_phrase_dict(self):
    self.docs_w_phrase_dict = defaultdict(list)
    for doc,phrases in tqdm(self.doc_phrases_dict.items()):
        for phr in set(phrases):
            self.docs_w_phrase_dict[phr].append(doc)
            
    return self
            
def docs_w_phrase(self,phr):
    return [i for i in range(len(self.doc_phrases_dict)) if phr in self.doc_phrases_dict[i]]

def P_phrase(self,phr):
    return len(self.docs_w_phrase_dict[phr])/len(self.doc_phrases_dict)

def C_phrase(self,phr):
    return len(self.docs_w_phrase_dict[phr])

def PMI(self,phr1,phr2):
    c1 = C_phrase(self,phr1)
    c2 = C_phrase(self,phr2)
    if (c1 == 0) or (c2 == 0):
        return 0
    if (phr1 in phr2) or (phr2 in phr1):
        return 0
    c_12 = len(list(set(self.docs_w_phrase_dict[phr1]).intersection(set(self.docs_w_phrase_dict[phr2]))))
    if c_12 < 10:
        return 0
    return np.log(c_12*(len(self.doc_phrases_dict))/(c1* c2))

def compute_all_PMI(self):
    words = []
    for doc,phrs in tqdm(self.doc_phrases_dict.items(),desc = "Getting all used Phrases"):
        words += phrs
    self.words = list(set(list(words)))
    del words
    if os.path.isfile('../bin/Fields/'+self.field+'/PMI_dict.pickle'):
        print('Reading in Pickle File')
        with open('../bin/Fields/'+self.field+'/PMI_dict.pickle', 'rb') as f:
            PMI_dict = pickle.load(f)
        with open('../bin/Fields/'+self.field+'/words_completed.pickle', 'rb') as f:
            words_completed = pickle.load(f)
            start = words_completed
        print(len(self.words) - start,'phrases remaining')
    else:
        print('No Files Found')
        start = 0
        PMI_dict = dict()

    i = start
    num_iterations = 1000
    print('Begin Writing PMI Scores to Dictionary')
    stop = timeit.default_timer()
    start_t = timeit.default_timer()
    for phr1 in tqdm(self.words[start:]):
        i+=1
        if (i < int(len(self.words)/2)):
            num_iterations = 1000
        if (i < int(len(self.words)/4)):
            num_iterations = 4000
        else:
            num_iterations = 50
        if C_phrase(self,phr1) == 0:
            pass
        for phr2 in self.words[i:]:
            dum = PMI(self,phr1,phr2)
            if dum != 0:
                PMI_dict[phr1+','+ phr2] = PMI(self,phr1,phr2)
        #if (i-1)%1000 == 0:
        stop = timeit.default_timer()
        #if (i-1)%(int(len(self.words[start:])/num_iterations)) == 0:
        if stop-start_t > 3600:
            #with codecs.open('../bin/PMI_dict.pickle', 'wb', 'utf-8') as f:
            with open('../bin/Fields/'+self.field+'/PMI_dict.pickle', 'wb') as f:
                pickle.dump(PMI_dict, f)
            with open('../bin/Fields/'+self.field+'/words_completed.pickle', 'wb') as f:
                pickle.dump(i, f)
            print('Finished Writing')
            start_t = timeit.default_timer()

    print('\n Writing Dictionary to Pickle File')

    with open('../bin/Fields/'+self.field+'/PMI_dict.pickle', 'wb') as f:
        pickle.dump(PMI_dict, f)
    with open('../bin/Fields/'+self.field+'/words_completed.pickle', 'wb') as f:
        pickle.dump(i, f)
    print('Finished Writing')
    self.PMI_dict = PMI_dict
    
    return self


