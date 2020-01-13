# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 12:41:13 2019

@author: danie
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pan
import seaborn as sns

class TitanicData():
    def __init__(self, fileName, runStats, runGraphs):
        self.fileName = fileName #File name as string
        self.runStats = runStats #Boolean indicates if would like to generate statistical table
        self.runGraphs = runGraphs #Boolean indicates if would liek to generate graphs/diagrams
        
        self.import_clean_TitanicData() #Exectues uploading & cleaning of the data
        
        if self.runStats:#Execute stats function if True
            self.stats()
            
        if self.runGraphs: #Execute stats function if True
            self.scatterPlots()
            
        
    def import_clean_TitanicData(self): #returns tuple (np.array(train['Sex']), np.array(train_labels), np.array(test['Sex']), np.array(test_labels))
        ti_file =pan.read_csv(self.fileName)
        
        """col = list(ti_file.columns) 
        ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 
        'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']"""
        
        #self.tiDat = ti_file.iloc[:, [0,1,2,4,5,6,7,9]]  #Keeping only columns/features required
        #New data frame has columns ['PassengerId', 'Survived', 'Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare']
        self.tiDat = ti_file[['PassengerId', 'Survived', 'Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
        lbls = [] #initalizing list to store label values (survival information)
        
        convert = self.tiDat.iloc[:,1:7]  #Creates new data frame w/ features ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

        #Change strings to binary integer counterparts
        #FEMALE IS 0 and MALE IS 1
        
        convert.loc[convert['Sex'] == 'male', 'Sex'] = 0
        convert.loc[convert['Sex'] == 'female', 'Sex'] = 1
        
        """colP = list(convert.columns)
        ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']"""
        
        #Convert nan values to mean value of given column
        
        #Find mean value of each column
        class_avg = convert["Pclass"].mean()
        sex_avg = convert['Sex'].mean()
        age_avg = convert['Age'].mean()
        sib_avg = convert['SibSp'].mean()
        parch_avg = convert['Parch'].mean()

        
        #Replace nan value w/ mean values
        convert["Pclass"].fillna(class_avg, inplace = True)
        convert["Sex"].fillna(sex_avg, inplace = True)
        convert["Age"].fillna(age_avg, inplace = True)
        convert["SibSp"].fillna(sib_avg, inplace = True)
        convert["Parch"].fillna(parch_avg, inplace = True)
 
        
        self.size = convert.count().max() #Find the maximum length of column
        
        wholeData = convert #Storage cleaned and corrected all of training data w/out labels
        
       
        np.random.shuffle(wholeData.values) #Makes the selection of samples random for creating training data set
        #print(wholeData)
    
        train = [] #Initalize list that will store only a fragement of the training data

        train = wholeData.sample(frac=0.7)

        test = wholeData.loc[~wholeData.index.isin(train.index)] #Take the rest of wholeData that was Not used in train and uses those unuses values to create test
        
        
        #print(train.columns.values) #Prints the values as nparray format
        
        train_labels = train["Survived"] #Extract training labels and create new np.dataFrame train_labels
        
        train.drop("Survived", inplace=True, axis=1)
        
        test_labels = test["Survived"] #Extract testing labels and creates new np.dataFrame test_labels
        
        test.drop("Survived" , inplace=True, axis=1)
        
        return(np.array(train['Sex']), np.array(train_labels), np.array(test['Sex']), np.array(test_labels))
        
    def stats(self): #Generates cross tabulated table 
        dat = self.tiDat.groupby(['Survived', 'Pclass', 'Sex'])['Survived'].count()
        #Generates a cross tabulated table based on the survival of passengers
        print(dat) 
        
        tisum = self.tiDat['Survived'].sum() 
        print("Total passengers survived", tisum)
        
    def scatterPlots(self): #Generates various statistical diagrams
        
        print("1 indicates survived, 0 indicates death")
        
        #Generates categorical scatter plot (looks like bar chart and scatter plot hybrid)
        clas = sns.catplot(x='Pclass', y="PassengerId", hue = 'Survived', data=self.tiDat)
        clas
        
        #Generates categorical scatter plot (looks like bar chart and scatter plot hybrid)
        gen = sns.catplot(x='Sex', y="PassengerId", hue = 'Survived', data=self.tiDat)
        gen
        
        #Generates categorical scatter plot (looks like bar chart and scatter plot hybrid)
        gen_age = gen = sns.catplot(x='Sex', y="Age", hue = 'Survived', data=self.tiDat)
        gen_age
        
        clas_age = sns.catplot(x='Pclass', y="Age", hue = 'Survived', data=self.tiDat)
        clas_age
        
        #Generates categorical scatter plot w/ boundry lines
        age = sns.lmplot(x="PassengerId", y="Age", hue = 'Survived', data=self.tiDat)
        age
        