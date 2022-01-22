#!/usr/bin/env python3

"""
Created on Wed Mar 10 01:29:39 2021

@author: Ivana Nastasic
"""

# Import libraries
import pandas as pd
import fasttext
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
import csv

import logging

# Create or get the logger
logger = logging.getLogger(__name__)  

# set log level
logger.setLevel(logging.INFO)

# define file handler and set formatter
file_handler = logging.FileHandler('preprocessing.log')
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

if (logger.hasHandlers()):
        logger.handlers.clear()

# add file handler to logger
logger.addHandler(file_handler)


class DataPreprocessing:
    def __init__(self, prj_data_file, load_trans_file, save_trans_path, save_path, lang_detect_file,
                 custom_stop_words_flag=False, custom_stop_words_file=""):
        
        
        self.input_prj = pd.read_csv(prj_data_file, engine ='python')
        self.load_trans_file = load_trans_file # File which contains translated descriptions (full path)
        self.save_trans_path = save_trans_path # Location where to save file for translation
        self.save_path = save_path # Save preprocessed texts
        self.lang_detect_file = lang_detect_file # fastText file to language detection file (full path)
        self.custom_stop_words_flag = custom_stop_words_flag # Flag that indicates loading custom stop words
        self.custom_stop_words_file = custom_stop_words_file # Location of custom stop words file in csv format
    
    
    def remove_url(self, text):
        
        '''Remove URL
            Args: 
                text: input string
            Returns: text without URL
            
        '''
        url_pattern = r'((http|ftp|https):\/\/)?[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?'
        text = re.sub(url_pattern, ' ', text)
        return text
    
    
    def split_spec_char(self, word_list, ch):
        
        '''
        Split tokens from input list based on input character ch.
        Args:
            word_list: list of tokens
            ch: character (example:'/')
        Returns:
            split_list: list of tokens
        '''
        
        split_list = [el.split('/') for el in word_list] # becomes list of lists
        split_list = [j for i in split_list for j in i] # join innner lists in one list
        
        return split_list
    
    def get_wordnet_pos(self, treebank_tag):
        
        ''' POS tagger assignes parts of speech to each word (and other token), such as noun, verb, adjective, etc.
            Here is used Wordnet POS Tagger
        
        '''

        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return 'n'
        
    def lemmatize(self,l):
        
        '''
            Transforms words into lemmas.
            Args:
                l: list of words
            Returns: 
                res: list of lemmas
            
        '''
        
        lemmatizer = WordNetLemmatizer()
        res = []
        for el in l:
            res.append(lemmatizer.lemmatize(el[0], pos = self.get_wordnet_pos(el[1])))
        return res
    
    def read_stop_words(self):
        
        '''Read csv file which contains custom stop words in csv format without header.'''
        
        with open(self.custom_stop_words_file, 'r') as file:
            reader = csv.reader(file)
            custom_stop_words = []
            for row in reader:
                custom_stop_words = custom_stop_words + row 
            
        return custom_stop_words
        
    def process(self):
        
        # Delete all other columns apart Solution: Solution Name and Description
        
        clean_prj = self.input_prj
        clean_prj.drop(clean_prj.columns.difference(['Description','Solution: Solution Name']), 1, inplace=True)
        
        # Filter out rows with empty Description column
        clean_prj = clean_prj[clean_prj['Description'].notnull()]
        
        # Filter out rows where Description column contains only special characters (different from letters and numbers)
        mask = clean_prj['Description'].apply(lambda x: any(c.isalpha() for c in x))
        clean_prj = clean_prj[mask]
        
        # Filter out descriptions which are not in English language
        # Save them in other file to be translated
        lang_model = fasttext.load_model(self.lang_detect_file)
        mask = clean_prj['Description'].apply(lambda x: lang_model.predict(x[:1000],k=1)[0][0]=='__label__en')
        
        
        clean_prj[~mask].to_csv(self.save_trans_path + "\\project_desc_non_english.csv", index=False)
        
        clean_prj = clean_prj[mask]
        
        
        # Read file with already translated projects
        if self.load_trans_file != "":
            
            input_prj_trans = pd.read_csv(self.load_trans_file) 
            
            # Delete all other columns apart Solution: Solution Name and Description_text_en
            input_prj_trans = input_prj_trans.loc[:, input_prj_trans.columns.intersection(['Description_text_en','Solution: Solution Name'])]
            
            # Rename column Description_text_en to Description
            input_prj_trans = input_prj_trans.rename(columns={'Description_text_en':'Description'})
            
            
            # Merge project originaly in english with translated ones  
            clean_prj = pd.concat([clean_prj,input_prj_trans], ignore_index=True)
            
        
        # Remove URLs
        clean_prj['Description'] = clean_prj['Description'].apply(lambda x: self.remove_url(x))
        
        # Make a lower case
        clean_prj['Description'] = clean_prj['Description'].apply(lambda x: x.lower())
        
        # Tokenize Description
        clean_prj["nltk_tokens"] = clean_prj['Description'].apply(word_tokenize)
        
        # Add additional tokenization, split tokens with / and -
        clean_prj['nltk_tokens'] = clean_prj['nltk_tokens'].apply(lambda x: self.split_spec_char(x,'/'))
        clean_prj['nltk_tokens'] = clean_prj['nltk_tokens'].apply(lambda x: self.split_spec_char(x,'-'))
        
        # Remove tokens which contain special characters
        regex = re.compile('[\'`.@_=!#+$%^&*()<>?/\|}{~:\d]') 
        clean_prj['nltk_tokens'] = clean_prj['nltk_tokens'].apply(lambda x: [el for el in x if regex.search(el) == None])
        
        #Remove empty tokens
        clean_prj['nltk_tokens'] = clean_prj['nltk_tokens'].apply(lambda x: [el for el in x if el!=''])
        
        # Remove stop words
        stop = stopwords.words('english')
        clean_prj['nltk_tokens'] = clean_prj['nltk_tokens'].apply(lambda x: [el for el in x if el not in stop])
        
        
        # Create additional column which contains tuples (token, POS)
        clean_prj['nltk_tokens_pos'] = clean_prj['nltk_tokens'].apply(nltk.pos_tag)
        
        
        # Create lemmas
        clean_prj['nltk_tokens_lem'] = clean_prj['nltk_tokens_pos'].apply(self.lemmatize)
        
        # Filter out short lemmas, less than 3 characters
        clean_prj['nltk_tokens_lem'] = clean_prj['nltk_tokens_lem'].apply(lambda x:[el for el in x if len(el)>2])
        
        # If custom stop words are defined
        if self.custom_stop_words_flag:
            # Read custom stop words from comma-separated file
            custom_stop_words = self.read_stop_words()
            # Remove custom stop words
            clean_prj['nltk_tokens_lem'] = clean_prj['nltk_tokens_lem'].apply(lambda x: [el for el in x if el not in custom_stop_words])
        
        # Remove rows without lemmas
        clean_prj = clean_prj[clean_prj['nltk_tokens_lem'].apply(len)>=1]
        
        # Final cleaned text
        clean_prj['Description'] = clean_prj['nltk_tokens_lem'].apply(lambda x:' '.join(x))
        
        
        # Save cleaned text
        clean_prj[['Solution: Solution Name','Description']].to_csv(self.save_path+'Projects_desc_cleaned.csv', index=False)

        return clean_prj