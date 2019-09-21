# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 13:06:56 2019

@author: danie
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pan


from TitanicDataProcessing.py import TitanicData

class Perceptron:
    def __init__(self, tr_inpt, labels, epoch, Lr, si):
        self.epoch = epoch
        self.tr_inpt = tr_inpt
        self.sz = self.tr_inpt.shape[1]
            #.shape returns a tuple and I want only the element in the 0 position of that tuple
        self.w = self.weights(self.sz)
        self.Lr = 0.001
        self.labels = labels
        
        self.fit()
        self.plotErrors()
        

    def z_input(self, x):
        #generate dot product between w and features x
        return np.dot(self.w[1:], x) + self.w[0]
       # return np.dot(np.transpose(self.w),x)
    
    def weights(self, sz):
        #where sz is the size of x (number of x)
        self.w = np.random.random(self.sz+1)
        #random.randfl ? generate float of random wieght
        return self.w
    
    def predict(self, z):
       if z >= 0:
           return 1
       else:
           return 0
    
    def fit(self):
        update_num = 0
        self.updatesN = []
        self.epochNum = []
        for m in range(self.epoch):
            for k in range(self.tr_inpt.shape[0]): #pick one the number of rows to be length of iteration
                self.z_input(self.tr_inpt[k])
                z = self.z_input(self.tr_inpt[k])
                prediction = self.predict(z)
                target = self.labels[k]
                #Is this interpretation of updating weights correct?
                error = target - prediction
                dw = self.Lr*error*self.tr_inpt[k]
                
                if error != 0:
                    update_num += 1
                    
                #self.w += dw
                self.w[1:] += dw # Is this correct?
                self.w[0] += self.Lr*error
                
                
            self.updatesN.append(update_num)
            self.epochNum.append(m)
        
    def testIt(self, testDat, testLabels):
           test_result = []
           right = 0
           for k in range(testDat.shape[0]):
               
               z = self.z_input(testDat[k])
               prediction = self.predict(z)
               
               test_result.append(prediction)
               
               if prediction == testLabels[k]:
                   right += 1
                       
           return (right/len(test_result))*100
        
    def plotErrors(self):
        plt.title("Number of updates vs Epochs")
        plt.plot(self.epochNum, self.updatesN)
        plt.xlabel('Epochs')
        plt.ylabel("Number of updates")
        plt.show()
        
def main():
    #self, tr_inpt, labels, epoch, Lr, si
    p = Perceptron(t1, l1, 100, 0.001, sz)
    print("Accuracy", (p.fit(False)), "%")
    print(p.fit(True))
main()