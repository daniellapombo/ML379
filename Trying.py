# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 08:34:15 2019

@author: danie
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
#from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pan

#Train - fit() , partialfit() - partial training do not do all the training at once
def run_LinReg(dat, lab, D, dLab, tst, tstLab, graph_decision = False):
    #L = LogisticRegression(C = 10, random_state = 1)
    L = LogisticRegression()
    L.fit(dat, lab)
    if graph_decision:
        plot_decision_regions(dat, lab, clf = lr) #lr is Logistic regressions
        plt.xlabel("Passenger Sex & Pclass")
        plt.ylabel("Survived")
        plt.legend(loc = "upper right")
        plt.show()
    Lpred = L.predict(D)
    score = accuracy_score(Lpred, dLab)
    print("Logistic Regression Accruacy:", score)

class TitanicData():
    def __init__(self, fileName):
        self.fileName = fileName #File name as string
        
        self.import_clean_TitanicData() #Exectues uploading & cleaning of the data
        
    def import_clean_TitanicData(self):
        ti_file =pan.read_csv(self.fileName)
        
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
    
        train = [] #Initalize list that will store only a fragement of the training data

        train = wholeData.sample(frac=0.7)

        test = wholeData.loc[~wholeData.index.isin(train.index)] #Take the rest of wholeData that was Not used in train and uses those unuses values to create test
        
        
        #print(train.columns.values) #Prints the values as nparray format
        
        train_labels = train["Survived"] #Extract training labels and create new np.dataFrame train_labels
        
        train.drop("Survived", inplace=True, axis=1)
        
        test_labels = test["Survived"] #Extract testing labels and creates new np.dataFrame test_labels
        
        test.drop("Survived" , inplace=True, axis=1)
        
        
        dev = test.iloc[0:test.shape[0]//2, :]
        
        dev_labels = test_labels.iloc[0:test_labels.shape[0]//2]
        
        test = test.iloc[test.shape[0]//2:, :]
        
        test_labels = test_labels.iloc[(test_labels.shape[0]//2):]
        #print(train)
        
        train = np.array(train['Sex'])
        
        train_labels = np.array(train_labels)
        
        dev = np.array(dev['Sex'])
       
        dev_labels = np.array(dev_labels)
        
        test = np.array(test['Sex'])
        
        test_labels = np.array(test_labels)

        stand = StandardScaler()
        stand.fit(train.reshape(-1,1)) #JUSt make sure to reshape w/in or before the fit function
        
        train_std = stand.transform(train.reshape(-1,1))
      #Dont need to do standardization for 0s and 1s cuz its only 0 and 1
#     
#        stand.fit(test)
#        
#        test_std = stand.transform(test)
#
#
#        stand.fit(dev)
#        
#        dev_std = stand.transform(dev)
       
        
        #return(train_std, train_labels, dev_std, dev_labels, test_std, test_labels)
       
        
def main():
    Tipas = TitanicData("train.csv")
    passengers = Tipas.import_clean_TitanicData()
    #run_LinReg(passengers[0], passengers[1], passengers[2], passengers[3], passengers[4], passengers[5], False)

main()
        