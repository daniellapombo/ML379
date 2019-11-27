# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 17:55:57 2019

@author: danie
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import numpy as np
import pandas as pan
   
def run_SVC(dat, lab, D, dLab, tst, tstLab):
    #s = SVC(kernel = 'linear', C = 10, random_state = 1)
    s = SVC(kernel = 'linear')
    
    s.fit(dat, lab)
    
    Spred = s.predict(D) #Makes prediction for dataset D (the development dataset)
    score = accuracy_score(Spred, dLab) #Computes the accuracy metric of the prediction vs true labels
    print("SVC Accuracy:", score)


def run_LinReg(dat, lab, D, dLab, tst, tstLab):
    L = LogisticRegression(C = 5)
    #L = LogisticRegression()

    L.fit(dat, lab)
    
    Lpred = L.predict(D) #Makes prediction for dataset D (the development dataset)
    score = accuracy_score(Lpred, dLab) #Computes the accuracy metric of the prediction vs true labels
    print("Logistic Regression Accuracy for development set:", score)
    
    Tstpred = L.predict(tst) #Makes prediction for test dataset
    tscore = accuracy_score(Tstpred, tstLab) #Computes accuracy metric for test prediction vs true test labels
    print("Logistic Regression Accuracy for test set:", tscore)
    

def run_KNN(dat, lab, D, dLab, tst, tstLab):
    #knn = KNeighborsClassifier(n_neighbors = 0, p = 10, metric = 'minkowski')
    knn = KNeighborsClassifier()
    knn.fit(dat, lab)
    Kpred = knn.predict(D) #Makes prediction for dataset D (the development dataset)
    score = accuracy_score(Kpred, dLab) #Computes the accuracy metric of the prediction vs true labels
    print("KNN Accuracy for development set:", score)

class TitanicData():
    def __init__(self, fileName):
        self.fileName = fileName #File name as string
        
        self.import_clean_TitanicData() #Exectues uploading & cleaning of the data
        
    def import_clean_TitanicData(self):
        ti_file =pan.read_csv(self.fileName) #Import file
        
        """col = list(ti_file.columns) 
        ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 
        'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']"""
        
        self.tiDat = ti_file[['PassengerId', 'Survived', 'Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
        
        
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
    
        #Create datasets -----------------------------------------------------------------------------  
    
        train = [] #Initalize list that will store only a fragement of the training data

        train = wholeData.sample(frac=0.7)

        test = wholeData.loc[~wholeData.index.isin(train.index)] #Take the rest of wholeData that was Not used in train and uses those unuses values to create test
        
        #print(train.columns.values) #Prints the values as nparray format
        
        train_labels = train["Survived"] #Extract training labels and create new np.dataFrame train_labels
        
        train.drop("Survived", inplace=True, axis=1)
        
        test_labels = test["Survived"] #Extract testing labels and creates new np.dataFrame test_labels
        
        test.drop("Survived" , inplace=True, axis=1)

        dev = test.iloc[0:test.shape[0]//2, :] #dev is 50% of test data ... AKA 15% of original dataset
        
        dev_labels = test_labels.iloc[0:test_labels.shape[0]//2] #Get labels for dev from test data
        
        test = test.iloc[test.shape[0]//2:, :] #Update test data to be 15% of original dataset
        
        test_labels = test_labels.iloc[(test_labels.shape[0]//2):] #Update test data labels
        
        #Convert datsets into numpy arrays----------------------------------------------------------
        #Only KEEP Pclass and Sex features for all datasets 
        
        train = np.array(train['Pclass'])
        
        train = train.reshape(-1, 1) #Converts 1D np array into proper format for standardization methods & operations
        
        train_labels = np.array(train_labels)
        
        dev = np.array(dev['Pclass']) #Converts 1D np array into proper format for standardization methods & operations
        
        dev = dev.reshape(-1,1)
       
        dev_labels = np.array(dev_labels)
        
        test = np.array(test['Pclass']) #Converts 1D np array into proper format for standardization methods & operations
        
        test = test.reshape(-1, 1)
        
        test_labels = np.array(test_labels)

        #Standardize datasets --------------------------------------------------------------------
        stand = StandardScaler()
        
        stand.fit(train) #Computes mean and standard_deviation
        
        train_std = stand.transform(train)
     
        test_std = stand.transform(test)
        
        dev_std = stand.transform(dev)
        
        return(train_std, train_labels, dev_std, dev_labels, test_std, test_labels)
        
def main():
    Tipas = TitanicData("train.csv")
    passengers = Tipas.import_clean_TitanicData()
    print("")
    run_LinReg(passengers[0], passengers[1], passengers[2], passengers[3], passengers[4], passengers[5])
    run_SVC(passengers[0], passengers[1], passengers[2], passengers[3], passengers[4], passengers[5])
    run_KNN(passengers[0], passengers[1], passengers[2], passengers[3], passengers[4], passengers[5])   

main()
        