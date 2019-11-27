# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 10:32:23 2019

@author: danie
"""


#Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

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
        
        self.run_model(True) #Runs MLA 
        
    
    
    def proprocessing(self): 
        """Use regex to get rid of punctuation Standardize data after transformed into bag-of-words"""
    
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
            
            self.train_txt, self.test_txt, self.train_label, self.test_label = train_test_split(self.words, self.labels, test_size= .20, train_size = .80, shuffle = True)
            
            self.num_docs = self.train_txt.shape[0]
            
            print('Number of total documents w/ in train set', self.num_docs)
            
            self.train_txt, self.test_txt = self.vectorizer(self.train_txt, self.test_txt)
            
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
    
    def rand_hyperparam_gen(self): 
        """Generates the  hyperparameters:Generates list of possible 
        hyperparameter values for k, c, n and m"""
        
        return ([0.1, 1, 10, 1000], [2500, 3500, 5000]) #C_, n, m, K
    
    def vectorizer(self, trn, tst):
        vectorize = TfidfVectorizer(stop_words = 'english', min_df = 10, max_df = 1000)
       
        return vectorize.fit_transform(trn), vectorize.transform(tst)
        
    def gridSearch(self):
    #Implement grid search maybe use np.random.seed(n) this will keep the output of that function the same each time
    #MAke 5 parameter options for each classfier
        C_, K = self.rand_hyperparam_gen()
        
        best_ = 0
        
        best_k = 0
        best_c = 0
    
        print('Passed preprocessing')
        

        for k in K:
            for c in C_:
                    temp = best_
                    run_ = LinearSVC(C = c)
                    best_ = max(best_, cross_val_score(run_, X = self.train_txt, y = self.train_label, cv = k).mean())
                    print('Passed n fold cross validation')
                    
                    print('Best', best_)
                    if best_ != temp:
                    
                        best_k = k
                        best_c = c
                      
        self.k = best_k 
        self.C = best_c
        self.final_hyper = True
                                
        return (best_k, best_c)
    
    
    def trainModel_best(self):
        """Train model w/ best found hyperparameters"""
        
        self.final_run_ = LogisticRegression(C = self.C)
        pred_avg = cross_val_score(self.final_run_, X = self.train_txt, y = self.train_label, cv = self.k)
        
        print('Average prediction accuracy rate for final model:', pred_avg)
        
        return pred_avg
    
    def predictionTest(self):
        """Evaluates model based on best found hyperparameters"""

        test_pred = self.final_run_.predict(self.test_txt)
        success_rate = accuracy_score(test_pred, self.test_label)
        
        print('Accuracy score for the test set is', success_rate, 'given parameters:', 'c:', self.C,'k:')
        
        return success_rate
        
    def run_model(self, go = False): #Runs the MLA based on if would like to run just on train or train and test\
        #go = False is just train MLA
        #go = True is train MLA and run it on test to compute MLA's accuracy
        
        self.proprocessing() #Imports data and splits into test and train datasets
        
        if self.proprocessing():
            print('in')
            self.gridSearch() #Finds best hyperparameters
            print('Done with the grid search, found best hyperparameters to be:', 'c:', self.C,'k:', self.k,'n:', self.n,'m:', self.m)
                
        if go == True: #Would like to test/evaluate the model
            if self.final_hyper:
                self.trainModel_best() #Train's the model w/ the best hyperparameters
                self.predictionTest() #Uses test data to find accuracy of model
            
def main():
    run_ = NLP_("rt-polarity.pos.txt", '+', "rt-polarity.neg.txt", '-', False)
    
main()



