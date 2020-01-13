# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 20:55:49 2019

@author: danie
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pan
import seaborn as sns
import random

class AdalineSGD:
    def __init__(self, tr_inpt, labels, epoch, Lr):
        self.epoch = epoch
        self.tr_inpt = tr_inpt
        self.sz = self.tr_inpt.shape[1] #.shape returns a tuple and element at index 1 indicates the length of the row
        self.w = self.weights()
        self.Lr = Lr
        self.labels = labels
        self.learning()
    
        
    def weights(self):
        self.w = np.random.random(self.sz+1)

        #random.randfl ? generate float of random wieght
        return self.w
    
    def z(self, x):
        #generate dot product between w and features x
        return np.dot(self.w[1:], x) + self.w[0]
    # return np.dot(np.transpose(self.w),x)
       
    def Id(self,x):

        return self.z(x)
       
    def learning(self):
        self.cost = []
        for e in range(self.epoch):
            cst = 0
            for k in range(self.tr_inpt.shape[0]):
                X = self.tr_inpt[k]
                error = (self.labels[k]-self.Id(X))
                dw = self.Lr*error*X
                
                self.w[1:] += dw
                self.w[0] += self.Lr*error
                
                cst += .5*(error**2)
               
            self.cost.append(cst)
        
    def quantizer(self, z):
        if z >= 0:
            return 1
        elif z < 0:
            return 0
    
    def testIt(self, testDat, testLabels):
           test_result = []
           right = 0
           for k in range(testDat.shape[0]):
               z = self.z(testDat[k])
               prediction = self.quantizer(z)
               
               test_result.append(prediction)
               
               if prediction == testLabels[k]:
                   right += 1
                       
           return (right/len(test_result))*100
            
    
class TitanicData():
    def __init__(self, fileName, runStats, runGraphs):
        self.fileName = fileName
        self.runStats = runStats
        self.runGraphs = runGraphs
        
        self.import_clean_TitanicData()
        
        if self.runStats:
            self.stats()
            
        if self.runGraphs:
            self.scatterPlots()
            
        
    def import_clean_TitanicData(self):
        ti_file =pan.read_csv(self.fileName)
        
        """col = list(ti_file.columns) 
        ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 
        'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']"""
        
        self.tiDat = ti_file.iloc[:, [0,1,2,4,5,6,7,9]] 
        
        """up_col = list(tiDat.columns) #['PassengerId', 'Survived', 'Pclass',
        'Sex', 'Age', 'SibSp', 'Parch', 'Fare']"""
        
        lbls = []
        
        for t in self.tiDat.iloc[:,1]:
            lbls.append(np.array(t))
        
        convert = self.tiDat.iloc[:,2:]
        #FEMALE IS 0 and MALE IS 1
        convert.loc[convert['Sex'] == 'male', 'Sex'] = 0
        convert.loc[convert['Sex'] == 'female', 'Sex'] = 1
        
        """colP = list(convert.columns)
        ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']"""
        
        class_avg = convert["Pclass"].mean()
        sex_avg = convert['Sex'].mean()
        age_avg = convert['Age'].mean()
        sib_avg = convert['SibSp'].mean()
        parch_avg = convert['Parch'].mean()
        fare_avg = convert['Fare'].mean()
        
        convert["Pclass"].fillna(class_avg, inplace = True)
        convert["Sex"].fillna(sex_avg, inplace = True)
        convert["Age"].fillna(age_avg, inplace = True)
        convert["SibSp"].fillna(sib_avg, inplace = True)
        convert["Parch"].fillna(parch_avg, inplace = True)
        convert["Fare"].fillna(fare_avg, inplace = True)
        
        self.size = convert.count().max() 
        
        wholeData = []
        #['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
        for k in range(self.size):
            wholeData.append(np.array(convert.iloc[k,:]))
            
        train = []
        labels = []
        random.shuffle(wholeData)
        #Create split training data
#        
        for j in range(1, int(self.size*.70)): #!!! Why is it that when change increment to 3 accuracy goes down to about 40% but is 68% when at 10?
            train.append(wholeData[j])
            labels.append(lbls[j])
        print(len(train))
        
        #Convert input data into numpy array that is 2d
        #Make the training data into numpy arrays
        self.trainingData = np.array(train)
        self.trainingLabels = np.array(labels)
        random.shuffle(wholeData)
        
        test = []
        test_labels = []
        
        #Create split training data
        for j in range(1, int(self.size*.3)):
            test.append(wholeData[j])
            test_labels.append(lbls[j])
            
        print(len(test))
            
        self.testData = np.array(test)
        self.testDataLabels = np.array(test_labels)
        
        return(self.trainingData, self.trainingLabels, self.size, self.testData, self.testDataLabels)
        
    def stats(self):
        dat = self.tiDat.groupby(['Survived', 'Pclass', 'Sex'])['Survived'].count()
        print(dat)
        
        
        tisum = self.tiDat['Survived'].sum()
        print("Total passengers survived", tisum)
        
    def scatterPlots(self):
        print("1 indicates survived, 0 indicates death")
        clas = sns.catplot(x='Pclass', y="PassengerId", hue = 'Survived', data=self.tiDat)
        clas
        
        gen = sns.catplot(x='Sex', y="PassengerId", hue = 'Survived', data=self.tiDat)
        gen
        
        age = sns.lmplot(x="PassengerId", y="Age", hue = 'Survived', data=self.tiDat)
        age
        
        gen_age = gen = sns.catplot(x='Sex', y="Age", hue = 'Survived', data=self.tiDat)
        gen_age
        
def main():
    TD = TitanicData("train.csv", False, False) 
    """Returns tuple of (trainingData, trainingLabels, size, testData, testDataLabels)"""
    titanic = TD.import_clean_TitanicData()
   
    p = AdalineSGD(titanic[0], titanic[1], 100, 0.0001)  #self, tr_inpt, labels, epoch, Lr
    print("Accuracy", p.testIt(titanic[3], titanic[4]), "%")
    
main()