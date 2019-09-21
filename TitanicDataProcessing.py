# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 12:41:13 2019

@author: danie
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pan

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
        
        self.size = convert.count().max() 
        
        wholeData = []
        #['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
        for k in range(self.size):
            wholeData.append(np.array(convert.iloc[k,:]))
            
        train = []
        labels = []
        #Create split training data
        for j in range(1, self.size, 3):
            train.append(wholeData[j])
            labels.append(lbls[j])
        
        #Convert input data into numpy array that is 2d
        #Make the training data into numpy arrays
        self.trainingData = np.array(train)
        self.trainingLabels = np.array(labels)
        
        test = []
        test_labels = []
        #Create split training data
        for j in range(1, self.size, 2):
            test.append(wholeData[j])
            test_labels.append(lbls[j])
            
        self.testData = np.array(test)
        self.testDataLabels = np.array(test_labels)
        
        return(self.trainingData, self.trainingLabels, self.size, self.testData, self.testDataLabels)
        
    def stats(self):
        dat = self.tiDat.groupby(['Survived', 'Pclass', 'Sex'])['Survived'].count()
        print(dat)
        
        
        tisum = self.tiDat['Survived'].sum()
        print("Total passengers survived", tisum)
        
    def scatterPlots(self):
        print("")
        print("Blue survival, red is death")
    
        x = self.tiDat['PassengerId']
        y = self.tiDat['Pclass']
        color = ('red', 'blue')
        groups = (int(0), int(1))
        plt.scatter(x, y, marker = 'o', label = groups, color = color)
        plt.xlabel("PassengerId")
        plt.ylabel("Pclass")
        plt.show()
        
    
        x = self.tiDat['Pclass']
        y = self.tiDat['Fare']
        color = ('red', 'blue')
        groups = (int(0), int(1))
        plt.scatter(x, y, marker = 'o', label = groups, color = color)
        plt.xlabel("Pclass")
        plt.ylabel("Fare")
        plt.show()
    
        x = self.tiDat['PassengerId']
        y = self.tiDat['Age']
        color = ('red', 'blue')
        groups = (int(0), int(1))
        plt.scatter(x, y, marker = 'o', label = groups, color = color)
        plt.xlabel("Passenger")
        plt.ylabel("Age")
        plt.show()
        
        
        x = self.tiDat['Age']
        y = self.tiDat['Fare']
        color = ('red', 'blue')
        groups = (int(0), int(1))
        plt.scatter(x, y, marker = 'o', label = groups, color = color)
        #plt.scatter(x, y, marker = 'o', label = 'Survived')
        #plt.title("Pclass", "vs. survived")
        plt.xlabel("Age")
        plt.ylabel("Fare")
        plt.show()
        
        x = self.tiDat['Sex']
        y = self.tiDat['Fare']
        color = ('red', 'blue')
        groups = (int(0), int(1))
        plt.scatter(x, y, marker = 'o', label = groups, color = color)
        #plt.scatter(x, y, marker = 'o', label = 'Survived')
        #plt.title("Pclass", "vs. survived")
        plt.xlabel("Sex")
        plt.ylabel("Fare")
        plt.show()
            
def main():
    tr = TitanicData("C:\\Users\\danie\\Documents\\School\\CS\\COMP 379\Hw\\train.csv", True, True)
    #print(tr.import_clean_TitanicData("C:\\Users\\danie\\Documents\\School\\CS\\COMP 379\Hw\\train.csv")[0])
main()