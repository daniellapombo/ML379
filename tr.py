# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 23:57:28 2019

@author: danie
"""

#Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

#Functions
import re
import random
import math

#Objects
import numpy as np
import pandas as pan

class NLP_:
    def __init__(self, text1, class1, text2, class2, are_labeled):
        
        
        self.text1 = text1 #Is the first input file
        self.text2 = text2 #Is the second input file
        
        self.class1 = class1 #One unique type class/label for the samples
        self.class2 = class2 #One unique type class/label for the samples
        
        self.C = 0 #Best chosen classifier hyperparameter
        self.k = 0 #Best chosen n fold cross validation variable, the fold number
        #self.gamma = 0
        self.final_hyper = False
        
        self.are_labeled = are_labeled #Indicates is the input data came w/ labels, if false indicates that lables must be assigned
        
        #self.run_model(True) #Runs MLA 
        
    
    
    def proprocessing(self): 
        """Imports, converts, cleans and splits input data for train and test sets"""
    
        if not self.are_labeled:
            pos = self.label_raw_dat(self.text1, self.class1) #Labels data 
            neg = self.label_raw_dat(self.text2, self.class2) #Labels data
            #print(pos.head())
            #print(neg.head())
            
            dat = pan.concat([pos, neg]) #Concatenates 2 input data sets into one large one
            dat['Text'].str.lower() #Converts all words into lowercase characters
            dat.replace(to_replace = r'[^a-zA-Z]', value = ' ', regex = True, inplace = True) #Get help w/ Regex!!
            
            #dat.replace(to_replace = r'[^a-zA-Z0-9]', value = ' ', regex = True, inplace = True)
            #dat.replace(r'[^a-zA-Z0-9]', ' ',line)
            
            self.dat = dat.sample(frac=1).reset_index(drop=True)
            
            #print(self.dat.head())
            
            self.labels = np.array(self.dat['Label'])
            self.words = np.array(self.dat['Text'])            
            
            self.train_txt, self.test_txt, self.train_label, self.test_label = train_test_split(self.words[:len(self.words)//1], self.labels[:len(self.labels)//1], test_size= .20, train_size = .80, shuffle = True)
            
            self.num_docs = self.train_txt.shape[0]
            
            print('Number of total documents w/ in train set', self.num_docs)
            
            #print(train_txt[1:10])
            #print(train_label[1:10])
            
            return (self.train_txt, self.train_label, self.test_txt, self.test_label) #Return sparse matrix of word frequencies w/ in document
        
        else: 
            
            print('Must not include labels')
            return False
        
    def label_raw_dat(self, text1, class_):
        """Transforms/assigns data numeric labels corresponding to string class labels
        Returns pandas dataframe w/ data mapped to corresponding class label"""
        
        txt = pan.read_csv(text1, sep = '\n', names = ['Text', 'Label']) #Import and reads file
        
        if class_ == '+': #Positive class
            txt['Label'] = 1 #Assigns corresponding label
            
        elif class_ == '-': #Negative class
            txt['Label'] = 0 #Assigns corresponding label
            
        return txt 
    
    def vectorizer(self, txt_dat):
        """Tokenizes the data aka splits txt into list of words"""
        
        splt_words = [] #List of all words that appear in data
        
        for t in txt_dat:
            splt_words.extend(t.split()) #Adds words to splt_words
            
        return splt_words
    
    def unique_words(self, splt, txt_dat, n, m):
        
        ps = PorterStemmer() 
        splt = [ps.stem(w) for w in splt] #List of root words
    
        stop_words_eng = set(stopwords.words('english')) #Set of english 'filler' words, non indicative words
        stop_words_esp = set(stopwords.words('spanish'))  #Set of spanish 'filler' words, non indicative words
        
        stop_words = stop_words_eng.union(stop_words_esp) #Concatinates/united set of 'filler' words
        
        unique_words = {w for w in splt if w not in stop_words} #List of unique words (no duplicates/repetitions)
        
        list_unq = []
        
        #print('unique words', unique_words)
        
        for key in unique_words:
            
            
            doc_freq = splt.count(key)
            
            #print('key', key, 'count', doc_freq)
            
            if n <= doc_freq <= m:
                
                list_unq.append(key)

            else:
                continue
        
        #print('Unique words dictionary', unq_w_dic)
        
        #print('List of unique words', list_unq)
       
        
        return list_unq
    
    def coder(self, dic_keys):
        
        incode = {}
        i = 0
        
        for key in dic_keys:
            incode[key] = i
            
        return incode
    
    def elem_freq(self, txt_dat, unique_words, sparse_index):
        """Converts data into unigram form - sparse matrix where the indices of the words
        are indicated by the parameter sparse_index"""
        
        print('in')
        
        doc_frq = dict()
        
        print('Length unique words', len(unique_words))
        
        b_of_w = np.zeros((txt_dat.shape[0], len(unique_words))) #Initalize sparse matrix
        
        ps = PorterStemmer()
        txt_dat = [[ps.stem(w) for w in txt.split()] for txt in txt_dat] #Convert input data into matrix of list of root words
        
        i = 0 #Index of row in sparse matrix b_of_w
        
        for txt in txt_dat: #Iterate over data
            for key in unique_words: #Iterate over key words
                #ls =  re.escape(str(key))
                #print('txt', txt, 'key', key)
                #count = len(re.findall(r'\A'+ re.escape(str(key)) + '.*\b', txt))
                if key in txt:
                    #count = txt.count(key) #Count how many times key word appears in document txt
                    index = sparse_index[key] #Index of key word in spare matrix b_of_w
                    b_of_w[i, index] = 1 #Add count of key word to sparse matrix
                else:
                    index = sparse_index[key] #Index of key word in spare matrix b_of_w
                    b_of_w[i, index] = 0 #Add count of key word to sparse matrix
                  
            i += 1 #Increment index
                
        print("")
        print('bag of words', b_of_w, len(b_of_w))
        
        return b_of_w
                    
    
    def elem_unigram(self, txt_dat, n, m, if_FinalTrain = False, if_TestSet = False):
        """Converts input data into proper unigram form depending on what data type
        the input data is"""
        
        txt_dat, txt_lab, tst_dat, tst_lab = self.proprocessing()
        
        
        split = self.vectorizer(txt_dat)
        
            
        if not if_FinalTrain and not if_TestSet:
            unq_list = self.unique_words(split, txt_dat, n, m)
        
            train_unigram = self.elem_freq(txt_dat, unq_list, self.coder(unq_list))
            
            return train_unigram
            
        if if_FinalTrain:
            self.unq_list = self.unique_words(split, txt_dat, n, m)
            self.decoder_tool = self.coder(self.unq_list)
            self.unigram = self.elem_freq(txt_dat, self.unq_list, self.decoder_tool)
            print('Pre fitting')
            self.r_ = LogisticRegression(C = 100)
            self.r_.fit(self.unigram, txt_lab)
            print('Post fitting')
            

            #return self.unigram
            
        if if_TestSet:
            print('Test')
            test_unigram = self.elem_freq(tst_dat, self.unq_list, self.decoder_tool)
            prediction = self.r_.predict(test_unigram)
            print('Accuracy score!', accuracy_score(prediction, tst_lab))
            
            return test_unigram
            
    
def main():
    
    run_ = NLP_("rt-polarity.pos.txt", '+', "rt-polarity.neg.txt", '-', False)
    run_.elem_unigram(None, 50, 700, True, True)
    
main()
    