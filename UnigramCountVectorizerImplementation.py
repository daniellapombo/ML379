# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 17:40:02 2019

@author: danie
"""
#import nltk
#nltk.download('stopwords')

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
        self.final_hyper = False
        
        self.are_labeled = are_labeled #Indicates is the input data came w/ labels, if false indicates that lables must be assigned
        
        self.run_model(True) #Runs MLA 
    
    def proprocessing(self): 
        """Imports, converts, cleans and splits input data for train and test sets"""
        lbl = ['+', '-']
        
        if not self.are_labeled and (self.class1 in lbl or self.class2 in lbl):
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
        
        for key in unique_words:
            
            doc_freq = splt.count(key)
            
            if n <= doc_freq <= m:
                
                list_unq.append(key)

            else:
                continue
       
        return list_unq
    
    def coder(self, dic_keys):
        
        incode = {} #Record the unique word's index w/ in the unigram sparse matrix
        i = 0 #Index
        
        for key in dic_keys:
            incode[key] = i #Assigns unique word a index position
            i += 1
            
        return incode
    
    def elem_freq(self, txt_dat, unique_words, sparse_index):
        """Converts data into unigram form - sparse matrix where the indices of the words
        are indicated by the parameter sparse_index"""
        
        #print('Length unique words', len(unique_words))
        
        b_of_w = np.zeros((txt_dat.shape[0], len(unique_words))) #Initalize sparse matrix
        
        ps = PorterStemmer()
        txt_dat = [[ps.stem(w) for w in txt.split()] for txt in txt_dat] #Convert input data into matrix of list of root words
        
        i = 0 #Index of row in sparse matrix b_of_w
        
        for txt in txt_dat: #Iterate over data
            for key in unique_words: #Iterate over key words

                count = txt.count(key) #Count how many times key word appears in document txt
              
                index = sparse_index[key] #Index of key word in spare matrix b_of_w
                b_of_w[i, index] = count #Add count of key word to sparse matrix
              
            i += 1 #Increment index
    
        return b_of_w
                    
    def elem_unigram(self, n, m, if_FinalTrain = False, if_TestSet = False):
        """Converts input data into proper unigram form depending on what data type
        the input data is"""
            
        if not if_FinalTrain and not if_TestSet:
            unq_list = self.unique_words(self.split_tr, self.train_txt, n, m)
        
            train_unigram = self.elem_freq(self.train_txt, unq_list, self.coder(unq_list))
            
            return train_unigram
            
        if if_FinalTrain:
            self.unq_list = self.unique_words(self.split_tr, self.train_txt, n, m)
         
            self.decoder_tool = self.coder(self.unq_list)
            self.unigram = self.elem_freq(self.train_txt, self.unq_list, self.decoder_tool)
            
            return self.unigram
            
        if if_TestSet:
            test_unigram = self.elem_freq(self.test_txt, self.unq_list, self.decoder_tool)
            
            return test_unigram
        
    def gridSearch(self):
    #Implement grid search maybe use np.random.seed(n) this will keep the output of that function the same each time
    #MAke 5 parameter options for each classfier
        #C_, n, m, K = [.01, 1], [0, 100, 500], [1500, 2500, 4000], [10]
        C_, n, m, K = [1], [0], [2500], [10]
        
        self.split_tr = self.vectorizer(self.train_txt)
        
        best_ = 0
        best_k = 0
        best_c = 0
        best_n = 0
        best_m =0
        
        for lower_b in n:
            for upper_b in m:
                for k in K:
                    for c in C_:
                        if k < (upper_b - lower_b):
                            
                            temp = best_
                            best_ = max(best_, self.nFoldCrossValidation(k, c, lower_b, upper_b))
                            
                            if best_ != temp:
                               
                                best_k = k
                                best_c = c
                                best_n = lower_b
                                best_m =upper_b
 
        self.k = best_k 
        self.C = best_c
        self.n = best_n 
        self.m = best_m
        self.final_hyper = True
                                
        return (best_k, best_c)
    
    def nFoldCrossValidation(self, k, c_, lower, upper, isFinal = False):
        """Divide the data into k folds
         Iterate throught the data and take kth element out and make it the development set
         Find the average accuracy metric for the development set and return that as the predicted accuracy
         Implement Back-of-words aka Unigram vectorizer """
         
        sum_acc = 0 #Total sum of accuracy for given n fold cross validation
        denom = 0 #Number of values summed for sum_acc (is the denomiator for average accuracy score measure)
        
        train = self.elem_unigram(lower, upper, isFinal) #Transforms data set into unigram formate
        
        num_iterations = train.shape[0] #Shape of the unigram training dataset
        
        #Convert/divide the data into k folds (groups/pieces)
        fds_dat = np.array_split(train, k) #Divides input data into k folds
        fds_lbls = np.array_split(self.train_label, k) #Divides input labels into k folds
        
        if not isFinal: #Training process
            
            for fold in range(0, k): #Iterates through train data by k folds
                
                trial_s = LinearSVC(C = c_)
               
                dev = fds_dat[fold]  #Develpoment set
                
                dev_lab = fds_lbls[fold] #Development labels
                
                hold_out = [fds_dat[i] for i in range(len(fds_dat)) if i != fold] #Training data
                hold_out = np.concatenate(hold_out, axis = 0)
                
                hold_out_labels = [fds_lbls[i] for i in range(len(fds_lbls)) if i != fold] #Training data labels
                hold_out_labels = np.concatenate(hold_out_labels, axis = 0)
            
                trial_s.fit(hold_out, hold_out_labels) #Fit the classifier based on all data but the kth group
                
                prediction = trial_s.predict(dev) #Predictions for the development set
                
                #print('acurracy score', accuracy_score(prediction, dev_lab))
                
                sum_acc += accuracy_score(prediction, dev_lab)
                
                denom += 1
        
        elif isFinal: #Finalizes model w/ final training model and its finalized hyperparameters
            
            for fold in range(0, k): #Iterates through train data by k folds
            
                self.trial_s = LinearSVC(C = c_)
                
                dev = fds_dat[fold]  #Develpoment set
                
                dev_lab = fds_lbls[fold] #Development labels
                
                hold_out = [fds_dat[i] for i in range(len(fds_dat)) if i != fold] #Training data
                hold_out = np.concatenate(hold_out, axis = 0)
                
                hold_out_labels = [fds_lbls[i] for i in range(len(fds_lbls)) if i != fold] #Training data labels
                hold_out_labels = np.concatenate(hold_out_labels, axis = 0)
                
                self.trial_s.fit(hold_out, hold_out_labels) #Fit the classifier based on all data but the kth group
                
                prediction = self.trial_s.predict(dev) #Predictions for the development set
                
                #print('acurracy score', accuracy_score(prediction, dev_lab))
                
                sum_acc += accuracy_score(prediction, dev_lab)
                
                denom += 1
                
        hold_out_avg_acc = sum_acc/denom
       
        return hold_out_avg_acc
    
    def trainModel_best(self):
        """Train model w/ best found hyperparameters"""
        pred_avg = self.nFoldCrossValidation(self.k, self.C, self.n, self.m, True)
        
        print('Average prediction accuracy rate for final model:', pred_avg, 'given final parameters:', 'c:', self.C,'k:', self.k,'n:', self.n,'m:', self.m)
        
        return pred_avg
    
    def predictionTest(self):
        """Evaluates model based on best found hyperparameters"""
        
        transformed_test = self.elem_unigram(self.n, self.m, False, True) #Converts test data into unigram formate
        test_pred = self.trial_s.predict(transformed_test)
        success_rate = accuracy_score(test_pred, self.test_label)
        
        print('Accuracy score for the test set is', success_rate, 'given parameters:', 'c:', self.C,'k:', self.k,'n:', self.n,'m:', self.m)
        
        return success_rate
        
    def run_model(self, go = False): #Runs the MLA based on if would like to run just on train or train and test\
        #go = False is just train MLA
        #go = True is train MLA and run it on test to compute MLA's accuracy
        
        self.proprocessing() #Imports data and splits into test and train datasets
        
        if self.proprocessing():
            self.gridSearch() #Finds best hyperparameters
            print('Done with the grid search, found best hyperparameters to be:', 'c:', self.C,'k:', self.k,'n:', self.n,'m:', self.m)
                
        if go == True: #Would like to test/evaluate the model
            if self.final_hyper:
                self.trainModel_best() #Train's the model w/ the best hyperparameters
                self.predictionTest() #Uses test data to find accuracy of model
            
def main():
    
    run_ = NLP_("rt-polarity.pos.txt", '+', "rt-polarity.neg.txt", '-', False)
    #run_.elem_unigram(None, 0, 700, True, True)
    
main()
    