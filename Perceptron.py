# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 13:06:56 2019

@author: danie
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pan
import seaborn as sns

#Perceptron single layer neural network
class Perceptron:
    def __init__(self, tr_inpt, labels, epoch, Lr):
        #Initalize paramters
        self.epoch = epoch #Number of iterations through the whole data set
        self.tr_inpt = tr_inpt #Training data set w/out labels
        self.Lr = Lr #Learning rate
        self.labels = labels #Training data labels

        if self.tr_inpt.ndim == 1:#If self.tr_input.shape == (a, ) where a member of integer set
            self.sz = 1 #For 1D array there is 1 column = column length/width is 1 rather than nothing
        else:
            self.sz = self.tr_inpt.shape[1] #Length of row (number of features per sample)
        #.shape returns a tuple and element at index 1 indicates the length of the row
        
        self.w = self.weights() #Initalizing weights to random numbers
        self.fit() #Calling execution of learning algorithm
        self.plotErrors() #Plots the errors per epoch to demonstrate convergency of data
        

    def z_input(self, x): 
        #generate dot product between w and features x
        return np.dot(self.w[1:], x) + self.w[0] #Returns dot product of weights, bias and sample
       # return np.dot(np.transpose(self.w),x)
    
    def weights(self):
        self.w = np.random.random(self.sz+1) #Creates a weight vector of size self.sz+1 composed of random variables
        return self.w
    
    def predict(self, z): #Step function:Activation function
       if z >= 0.0:
           return 1
       else:
           return 0
    
    def fit(self):
        self.updates = [] #Initalize vector to store update number per epoch, the update is dependent on the error
        
        for m in range(self.epoch):
            update_num = 0 #Initialize total update per epoch to 0
            for k in range(self.tr_inpt.shape[0]): #Iterates through each row within data set
                
                z = self.z_input(self.tr_inpt[k]) #Net input
                prediction = self.predict(z) #Activation function
                target = self.labels[k] 
                error = target - prediction
                
                dw = self.Lr*error*self.tr_inpt[k] #Value to update the weights by
                    
                #self.w += dw
                self.w[1:] += dw #Update the weights
                self.w[0] += self.Lr*error #Update the bias; inspired by the text book "Python Machine Learning" by Sebastian Raschka
                
                update_num += int(self.Lr*error != 0.0) #Increments the updates, inspired by text book "Python Machine Learning" by Sebastian Raschka
                
            self.updates.append(update_num) #Store the total updates for the epoch
            

        
    def testIt(self, testDat, testLabels): #Test after train
           test_result = [] #Initalize storage of predictions for test data
           right = 0 #Initalize number of right predictions 
           for k in range(testDat.shape[0]): #Iterate through the whole test data set sample by sample
               
               z = self.z_input(testDat[k]) #Net input
               
               prediction = int(self.predict(z)) #Step function
               
               test_result.append(prediction) #Storge the results in vector
               
               if prediction == testLabels[k]: #Count the number of correct predictions
                   right += 1
                   
           return (right/len(test_result))*100 #Returns the accuracy of the perpectron on the training data set
        
    def plotErrors(self):
        errorFig = plt.figure() #Initalizes new plot
        plt.title("Number of updates vs Epochs") 
        plt.plot(range(1,len(self.updates)+1), self.updates) #range(1,len(self.updates)+1) is the epochs
        #x = epochs, y = self.updates (number of updates per epoch)
        plt.xlabel('Epochs')
        plt.ylabel("Number of updates")
        plt.show() #Generates/shows the plot