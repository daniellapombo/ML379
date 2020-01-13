# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:36:02 2019

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
        print(self.tr_inpt.shape)
        if self.tr_inpt.ndim == 1:
            self.sz = 1
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
           return 1.0
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
                self.w[0] += self.Lr*error #Update the bias
                
                update_num += int(self.Lr*error != 0.0) #Increments the updates 
                
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
            
        
    def import_clean_TitanicData(self):
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

        test = wholeData.loc[~wholeData.index.isin(train.index)]
        
        print("Train samples", len(train))
        print("Test samples", len(test))
        
        #print(train.columns.values) #Prints the values as nparray format
        
        train_y = train["Survived"]
        
        train.drop("Survived", inplace=True, axis=1)
        
        test_y = test["Survived"]
        
        test.drop("Survived" , inplace=True, axis=1)
        
        return(np.array(train['Sex']), np.array(train_y), None , np.array(test['Sex']), np.array(test_y))
        
    def stats(self):
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
        
def main():
        
    learning_rate = 0.0003#float(input("Enter in learning rate:"))
    
    num_epochs = 500#int(input("Enter in max epochs:"))
    
    typeData = 'titanic' #input("Enter type of data would like to run program on: linear, nonlinear, titanic, TBadaline ").lower()
    if typeData == "linear":
        seperable = LinSep_data() #testD, testL, trainD, trainL
        
        print("Shape")
        print(seperable[0].shape)
        print("High")
        pDogs = Perceptron(seperable[3], seperable[4], num_epochs, learning_rate)  #self, tr_inpt, labels, epoch, Lr
        print("Accuracy", pDogs.testIt(seperable[0], seperable[1]), "%")
    
    elif typeData == "nonlinear":
        notSep = Not_linSep_data() #testD, testL, sz, trainD, trainL
        
        pets = Perceptron(notSep[3], notSep[4], num_epochs, learning_rate)
        print("Accuracy", pets.testIt(notSep[0], notSep[1]), "%")
        
    elif typeData == "titanic": 
        #"C:\\Users\\danie\\Documents\\School\\CS\\COMP 379\Hw\\train.csv"
        #tncFile = input("Enter in titanic file name")
        TD = TitanicData("C:\\Users\\danie\\Documents\\School\\CS\\COMP 379\Hw\\train.csv", False, False) 
        """Returns tuple of (trainingData, trainingLabels, size, testData, testDataLabels)"""
        
        titanic = TD.import_clean_TitanicData() #Returns tuple of (trainingData, trainingLabels, size, testData, testDataLabels)
    
        p = Perceptron(titanic[0], titanic[1], num_epochs, learning_rate)  #self, tr_inpt, labels, epoch, Lr
        print("Accuracy", p.testIt(titanic[3], titanic[4]), "%")
        
main()