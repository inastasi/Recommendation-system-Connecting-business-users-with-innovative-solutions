# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 01:29:39 2021

@author: Ivana Nastasic
"""

import logging
import os
import numpy as np
import json
import pandas as pd
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel
from heapq import nlargest

# Create or get the logger
logger = logging.getLogger(__name__)  

# set log level
logger.setLevel(logging.INFO)

# define file handler and set formatter
file_handler = logging.FileHandler('LDA.log')
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

if (logger.hasHandlers()):
        logger.handlers.clear()


# add file handler to logger
logger.addHandler(file_handler)


def CV(X, dictionary, k, alpha):
    
    '''Calculate c_v coherence score and perplexity score for a given number of topics k. 
    In each iterration it applies 3-cross fold validation and takes measures as a mean of 3 calculated values
    
    
    Args:
        X: input documents in form of list of the list
        dictionary: gensim mapping between normalized words and their integer ids
        k: number of topics
        alpha: symmetric or assymetric
    Returns: C_v coherence score, perplexity score  
    '''
    coherence_scores_tr = []
    perplexity_scores_tr = []

    for i in range(3):
    
        corpus_tr = [dictionary.doc2bow(text) for text in X]
        
        # Gensim LDA model
        model_tr = LdaModel(corpus=corpus_tr, num_topics=k, id2word=dictionary,alpha=alpha)
        # Gensim Coherence Model
        coherence_model_tr = CoherenceModel(model=model_tr, texts=X, dictionary=dictionary, coherence='c_v')
        
        # Perplexity is automatically calculated when model is trained
        perplexity_scores_tr.append(model_tr.log_perplexity(corpus_tr))

        # Calculate coherence score
        coherence_tr = coherence_model_tr.get_coherence()
        coherence_scores_tr.append(coherence_tr )

    return np.mean(coherence_scores_tr), np.mean(perplexity_scores_tr)


def CV_exec(X,dictionary,K_max):
    '''Calculates for a given dataset coherence c_v and perplexity scores for different number of topics
    
    Args:
        X: input documents in form of list of the list
        dictionary: gensim mapping between normalized words and their integer ids
        K_max: maximal number of topics
    '''
    alpha_vals = ['symmetric','asymmetric']
    
    all_poss_conf = []
    # Starts from 10 topics and moves with step size 5, until it gets to K_max
    for k in range(10,K_max,5):
        for a in alpha_vals:
            all_poss_conf.append((k,a))
    
    df = pd.DataFrame (columns=['K', 'alpha','coh_tr', 'perp_tr'])
    
    for v in all_poss_conf:
            ch_tr,p_tr = CV(X, dictionary, v[0],v[1])
            df = df.append ({'K': v[0], 'alpha':v[1],'coh_tr': ch_tr,'perp_tr': p_tr}, ignore_index=True)
   
    return df
    

class LDA_Class:
    
    def __init__(self, load_model_path, save_model_path, model_name, prj_df, empl_df, N, K=1, nw=200, word_prob=0.002, topic_prob=0, infer_n=-1):
        
    
        self.prj_df = prj_df  
        self.empl_df = empl_df
        self.save_model_path = save_model_path
        self.load_model_path = load_model_path
        self.model_name = model_name
        self.K = K # number of topics
        self.N = N # top N the most similar employees
        self.nw = nw # Number of words within topic to be taken into account
        self.word_prob = word_prob # Threshold for word probability within the topic
        self.topic_prob = topic_prob # Threshold for topic probability, default is 0 to take into account all topics
        self.infer_n = infer_n
        
        # Make projects in form of list of lists [[word1,word2,...,wordn],[word1,word2,...,wordk]...]
        self.prj_docs = (self.prj_df['Description'].apply(lambda x: x.split(' '))).to_list()
        
        self.dictionary = None
        self.corpus = None
        self.ldamodel = None
        self.topic_ref_words = None
        
        
        
    def create_model(self):
        '''
        Create dictinary, corpus, LDA model and reference dictionary for the most relevant words within 
            each topic.
            
            dictionary: gensim model dictionary
            corpus: gensim model corpus
            ldamodel: LDA model
            topic_ref_words: dictionary of the most characteristic words per each topic with their assigned value
                            
        '''
        
        self.dictionary = corpora.Dictionary(self.prj_docs)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.prj_docs]
        self.ldamodel = LdaModel(corpus=self.corpus, num_topics=self.K,
                                     id2word=self.dictionary,alpha='symmetric',random_state=101)
        self.topic_ref_words = self.words_per_topic()
        
    def save_model(self):
        
        '''Save gensim LDA model, dictionary and reference words with their assigned weights (probabilities).'''
        
        # Save model
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)
        
        self.dictionary.save(self.save_model_path + self.model_name + '_dict')
        self.ldamodel.save(self.save_model_path + self.model_name)
        with open(self.save_model_path + self.model_name + '_data.txt', 'w') as outfile:
            json.dump(self.topic_ref_words,outfile)
    
 
    def load_model(self):
        '''Load dictionary, LDA model and reference words per topic.'''
        
        self.dictionary = corpora.Dictionary.load(self.load_model_path + self.model_name + '_dict')
        self.ldamodel = LdaModel.load(self.load_model_path + self.model_name)
        
        with open(self.load_model_path + self.model_name + '_data.txt') as json_file:
            self.topic_ref_words = json.load(json_file)
        
        self.topic_ref_words = {int(k):v for k,v in self.topic_ref_words.items()}
        # Loading topic reference words gives pairs of word and assigned value in form of a list
        # It needs to be converted into tuple
        for k in self.topic_ref_words.keys():
            val = []
            for v in self.topic_ref_words[k]:
                val.append(tuple(v))
            self.topic_ref_words[k] = val   
            
    
    
    def norm_topic_vec(self, data):
        '''Normalize word probability within the topc, to sum up to 1.
        
        Args:
            data: dictionary {key=topic number : value=(word,word probability)}
        Returns:
            Modified data dictionary where word probabilites per topic sum up to 1.
            
        '''
        
        for k in data.keys():
            val = []
            for v in data[k]:
                val.append(v[1])
            val = [x/sum(val) for x in val]
            updated_val = []
            for v,i in zip(data[k],val):
                updated_val.append((v[0],round(i,4)))
            data[k] = updated_val
            
        return data   
    
    def words_per_topic(self):
        '''
        Create dictionary of key words per topic whose probability is above given threshold.
        Important is that Gensim LDA calculate word probability over whole dictionary so we need to normalize it
        to sum up to 1.
        
        Args:
            model: LDA model
            n: number of topics in the model
            nw: number of the most important words 
            prob: probability of the word in the topic
        Returns: 
            topic_words: dictionary {key=topic number: value=(word,assigned value(word probability))}
        '''
        topic_ref_words = {}
        
        # gensim show_topics gives info about discovered topics
        topics = self.ldamodel.show_topics(num_topics=self.K, num_words=self.nw, log=False, formatted=False)
        
        # Loop over all topics
        for t in topics:
            # Initialize dictionaries with empty lists
            topic_ref_words.update( {t[0] : []} )
            
            # Take only words above given probability threshold
            for x in t[1]:
                if x[1]>=self.word_prob:
                    topic_ref_words[t[0]].append((x[0],round(x[1],4)))
                    #topic_words_prob[t[0]].append(round(x[1],4))
                else:
                    break
                
        #self.topic_words = topic_words
        #self.topic_words_prob = topic_words_prob
        
        # Normalize assigned values to topic reference words
        topic_ref_words = self.norm_topic_vec(topic_ref_words)
        
        
        return topic_ref_words
    
    
    
    
    def get_all_topic(self):
        
        ''' Create distribution of topics per project document.
        
            Args:
                model: LDA model
                corp: Gensim corpus for input projects
                prob: minimal topic probability
            Returns: all_topics_df: pandas data frame which contains topic distribution per project taking into account
            only topics above given threshold, which is by default 0 (all topics).
        '''
        # Get all topics for projects in the corpus (gensim function)
        all_topics = self.ldamodel.get_document_topics(self.corpus,minimum_probability=self.topic_prob,per_word_topics=True)
        
        all_topics_df = []
        for doc_topics, word_topics, phi_values in all_topics:
            df = pd.DataFrame(doc_topics, columns =['Topic', 'Probability'])
            df.set_index('Topic',inplace=True)
            all_topics_df.append(df.transpose())
            
            
        all_topics_df = pd.concat(all_topics_df,ignore_index=True)
        
        # Calculate number of found topics above given topic prob threshold
        # If threshold is 0 all topics will be represented
        all_topics_df['num_topics'] = all_topics_df.count(axis=1)
        
      
        return all_topics_df
    
     
    def order_topics_probs(self, df):
        
        '''
        Order topics by its importance per each project document from the data frame created by get_all_topic function.
        
        Args: df: pandas dataframe with projects in rows and topic numbers in columns. Values are probability
        of the topic in the project.
        Returns: modified data frame with 3 columns: 'num_topics':number of topics above given threshold within project,
        topics':list of topics, 'topics_prob':topic probabilities, ordered by topic importance

        '''
        idx = np.argsort(-df.loc[:, df.columns != 'num_topics'].values, 1)[:, :]
        
        df['topics'] = df.apply(lambda x: [], axis=1)
        df['topics_prob'] = df.apply(lambda x: [], axis=1)
        
        for i in range(len(df.index)):
            num_topic = df.loc[i,'num_topics']
            
            if num_topic>0: #There are projects without any topic in case that we set up probability threshold >0
                df_1 = (df.iloc[[i],idx[i,:]]).iloc[:,0:num_topic]
    
                df.at[i,'topics'] = df_1.columns.to_list()
                df.at[i,'topics_prob'] = (df_1.values[0]).tolist()
                #df.at[i,'dominant_topic'] = int(df_1.columns.to_list()[0])
            else:
                continue
        
        res = df[['num_topics','topics', 'topics_prob']]
        return res
    
    def get_common_words(self,d,topic_num):
        '''
        Gets common words between list of words and reference words of a given topic.
        Args:
            d: list of words
            topic_num: ordinary number of the topic (0,1,...K-1)
        '''
        ref_words = []
        for x in self.topic_ref_words[topic_num]:
            ref_words.append(x[0])
            
            
        return list(set(d).intersection(set(ref_words)))

    def calc_value(self, word_list, topic_num, topic_pr):
        
        ''' 
        Assign value to employee based on the similarity of words from his skill description and keywords for the topic
        Word value within the topic sums up to 1.
        
        Args:
            word_list: list of common words between employee and topic key words
            topic_num: ordinary number of the topic (0,1,...,K-1)
            topic_probability: probability of a given topic in the project
            word_ref_list: key word list for all topics
            word_ref_value: importance of each keyword for all topics
            
        Returns:
            val: calculated value (sum for all words (probability value of the word in the topic * 100) * topic probability)
            
            '''
        val = 0
        if len(word_list)>0:
            for x in word_list:
                # Look for the tuple in topic reference words for a given topic which contains word x
                w = [item for item in self.topic_ref_words[topic_num] if item[0] == x]
                val += (w[0][1]*100)*topic_pr
        return(val)       
    
    def get_empl_list(self, topic_list, topic_pr_list):
        
        '''
        Method which creates sorted list of suggested employees for a given project.
        Project info is provided via list of topics and their probabilities.
        
        Args:
            topic_list: sorted list of project topics
            topic_pr_list: corresponding list of topic probabilities
        Returns: 
            empl_list: sorted list of suggested employees (from the most adeguate towards the least adeguate)
            empl_val_list: corresponding list of calculated similarity values
            
        '''
        
        # Pandas data frame which has employees in rows and project document
        empl_doc_df = pd.DataFrame(index=self.empl_df.index)
        empl_doc_df['val'] = 0
        
        for topic_num,topic_pr in zip(topic_list,topic_pr_list):
            
            #For each employee find if any of the skill words appears within the topic keywords
            skill_doc_df = pd.DataFrame(self.empl_df['skill_list'].apply(lambda x: self.get_common_words(x,topic_num)))
            #Calculate how much found keywords are important within the topic, word_value*topic_prob
            skill_doc_df['val'] = skill_doc_df['skill_list'].apply(lambda x:self.calc_value(x,topic_num,topic_pr))
            del skill_doc_df['skill_list']
            empl_doc_df = pd.DataFrame(pd.merge(empl_doc_df, skill_doc_df, left_index=True, right_index=True).sum(axis=1),columns=['val'])
           
        
        #Sort employees in decending order based on the calculated value
        empl_doc_doc_df = empl_doc_df.sort_values('val',ascending=False)
        #Take only employees for whom calculated value is greater than 0
        empl_idx = skill_doc_df[skill_doc_df['val']>0].index
        empl_list = self.empl_df['idc_personid_ext'][empl_idx].to_list()
        empl_val_list = empl_doc_doc_df['val'][empl_idx].to_list()
        
        return empl_list,empl_val_list
    

    
    def infer_topics(self):
        
        ''' In case of applying already trained LDA model  on unseen data this function calculates per each project number of topics, list of topics 
             ordered from the most probably towards less probably and list of topic probabilities.
        
            Returns: Pandas data frame prj_topics_df  contains per each project 3 columns:
                    "num_topics": number of topics present in project document above given threshold,
                    "topics":list of topics,
                    "topics_prob":topic probabilities, ordered by topic importance
        '''
        
        # Create Gensim corpus for unseen documents
        self.corpus = [self.dictionary.doc2bow(text) for text in self.prj_docs]
        # Apply ldamodel on unseen documents to get list of list of tuples (topic, topic probability)
        lda_vec = [self.ldamodel[x] for x in self.corpus]
        
        # Create pandas dataframe which in rows has projects and in columns has topic number
        # Values are topic probabilities
        columns = ['topic', 'topic_prob']
        dfs = [pd.DataFrame([y for y in x],columns=columns).set_index('topic').transpose() for x in lda_vec]
        all_topics_df = pd.concat(dfs,ignore_index=True)
        
        for i in range(all_topics_df.shape[1]):
            for j in range(all_topics_df.shape[0]):
                # Remove topics with probability less than a given threshold
                # If threshold is 0 all the topics are taken into account
                if all_topics_df.iloc[j, i] < self.topic_prob:
                    all_topics_df.iloc[j, i] = ""
        # Calculate number of found topics above given topic prob threshold
        # If threshold is 0 all topics will be represented
        all_topics_df['num_topics'] = all_topics_df.count(axis=1)
        
        return self.order_topics_probs(all_topics_df)
    
    
    def calculate_topics(self):
        
        ''' In case of training new LDA model this function calculates per each project number of topics, list of topics 
        ordered from the most probably towards less probably and list of topic probabilities.
        
        Returns: Pandas data frame prj_topics_df  contains per each project 3 columns:
                "num_topics": number of topics present in project document above given threshold,
                "topics":list of topics,
                "topics_prob":topic probabilities, ordered by topic importance
        '''
        
        prj_topics_df = self.order_topics_probs(self.get_all_topic())
        
        return prj_topics_df
    
    
    
    def calculate_sim(self, prj_topics_df):
        
        '''
        Creates list of top N suggested employees per project.
        
        Args: prj_topics_df: pandas data frame per project with number of topics, topic list and topic probability list
        Returns: pandas data frame with 2 columns:
            'Solution: Solution Name': project ID
            'top_N_empl': list of suggested N employee IDs
        '''
        
        # If we don't want to create suggestion list for all project documents, infer_n != -1
        # then we take last infer_n rows of projects data frame
        if self.infer_n != -1:
            prj_topics_df = prj_topics_df.tail(self.infer_n).reset_index(drop=True)
            prj = (self.prj_df[['Solution: Solution Name']]).tail(self.infer_n).reset_index(drop=True)
        else:
            prj = self.prj_df[['Solution: Solution Name']]
        
        
        # Add two additional columns to contain info about suggested employees and calculated similarity value
        prj_topics_df['empl_id'] = prj_topics_df.apply(lambda x: [], axis=1) # column to hold suggested employees
        prj_topics_df['sim_val'] = prj_topics_df.apply(lambda x: [], axis=1) # column to hold scores for suggested employees
        
        # Iterate over all project documents and assign list of employees and similarity value
        M = len(prj_topics_df.index) # number of projects
        
        for i in range(M):
            # Logging info about number of projects that are processed till now
            # Logging is set up after every 50 projects
            if i % 50 == 0:
                r = M-i
                logger.info('LDA Calculated similarity for ' + str(i) + ', ' + str(r) +  ' to go.')
                
            
            num_topic = prj_topics_df.loc[i,'num_topics'] 
            num_topic_list = prj_topics_df.loc[i,'topics']
            topic_prob_list = prj_topics_df.loc[i,'topics_prob']
        
            if num_topic>0:
                
                empl_list,empl_val_list = self.get_empl_list(num_topic_list,topic_prob_list)
                prj_topics_df.at[i,'empl_id'] = empl_list
                prj_topics_df.at[i,'sim_val'] = empl_val_list
            else:
                continue
        logger.info('LDA Calculated similarity for ' + str(M) + ', ' + str(0) +  ' to go.')    
            
        # Get top N highest scored employees per project   
        prj_topics_df['empl_score_dict'] = prj_topics_df.apply(lambda x: dict(zip(x.empl_id, x.sim_val)), axis=1)
        prj_topics_df['top_N_empl'] = prj_topics_df['empl_score_dict'].apply(lambda x: nlargest(self.N, x, key = x.get)).apply(lambda x:','.join([str(y) for y in x]))
        
        sim = prj_topics_df[['top_N_empl']].copy()
        
        
        # Merge similarity results with project Solution ID in order to be identified in results file
        
        fin_sim = pd.merge(prj, sim, left_index=True, right_index=True)
        
        return fin_sim