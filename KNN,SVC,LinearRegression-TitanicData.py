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
import math


def kNearestN(dat, lab, test, K):
    dis_lab = []
    test_predict = []
    
    #print(dat.shape,dat.size, test.shape, test.size)
    
    if len(dat) != len(lab): #If input arrays do not match in length cannot correlate dat w/ labs
        print("Length of dat and lab are not equal:", len(dat), len(lab))
        return None
    
    #Note len(test) < len(dat)
    for m in range(len(test)):
        dis_lab = [] #List of sample's distance form neighbor and the class label for the neighbor
        
        for i in range(len(dat)):
            neighbor_distance = EuclidDistance(dat[i], test[m]) #Measure distance between test[m] sample and its neighbors in dat
            dis_lab.append((neighbor_distance, lab[i])) #Add distance and class label to dis_lab
        
        #Sort list based on ascending order
        dis_lab.sort()
      
        lookAt = dis_lab[0:K] #Select only k neighbors to look at
        
        #Initialize count for classes count/appearance
        pred  = 0
        
        #Keep count of occurances of classes
        for k in (lookAt):
            if k[1] == 1:
                pred += 1 #Increment by 1
            elif k[1] == 0:
                pred -= 1 #Decrement by 1
                
        #Predication     
        #Makes predication based on most occured class label        
        if pred >= 0: #If predict is positive
            test_predict.append(1)
            
        else: #If predict is negative
            test_predict.append(0)
            
    return test_predict

def EuclidDistance(pt1, pt2):
    ssd = 0 #sum_squared_distance
    
    for k in range(len(pt1)):
        ssd += (pt1[k]-pt2[k])**2 #Sums the squared differences
        
    return math.sqrt(ssd) #Square roots the sum of squared differences

def selfKNNAccuracy(pred, true):
    #Computes accuracy metric for self kNearestN function
    
    if len(pred) != len(true): #Cannot compare accuracy of list of labels and predictions of which sizes are not the same
        
        print("Pred does not match length of true", len(true), len(pred))
        
        return None
    
    correct = 0 #Initialize the number of correct predictions
    
    for l in range(len(pred)):
        
        if pred[l] == true[l]:
            correct += 1 #Increment correct if the predication was correct
            
    print("self KNN Accuracy", correct/len(pred))   
    
    return correct/len(pred)
    
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
    print("skLearn KNN Accuracy for development set:", score)

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
        
        train = np.array(train[['Pclass', 'Sex']])
        
        train_labels = np.array(train_labels)
        
        dev = np.array(dev[['Pclass', 'Sex']])
       
        dev_labels = np.array(dev_labels)
        
        test = np.array(test[['Pclass', 'Sex']])
        
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
 
    devKKN = kNearestN(passengers[0], passengers[1], passengers[2], 18)
    print("Development set")
    selfKNNAccuracy(devKKN, passengers[3])
    
    testKKN = kNearestN(passengers[0], passengers[1], passengers[4], 18)
    print("Test set")
    selfKNNAccuracy(testKKN, passengers[5])
    

main()
        