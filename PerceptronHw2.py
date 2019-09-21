# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 12:36:45 2019

@author: danie
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pan


class Perceptron:
    def __init__(self, tr_inpt, labels, epoch, Lr, si):
        self.epoch = epoch
        self.tr_inpt = tr_inpt
        self.sz = self.tr_inpt.shape[1] #.shape returns a tuple and element at index 1 indicates the length of the row
        self.w = self.weights(self.sz)
        self.Lr = Lr
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
    TD = TitanicData("C:\\Users\\danie\\Documents\\School\\CS\\COMP 379\Hw\\train.csv", False, False) 
    """Returns tuple of (trainingData, trainingLabels, size, testData, testDataLabels)"""
    
    titanic = TD.import_clean_TitanicData()
    
    learning_rate = float(input("Enter in learning rate:"))
    num_epochs = int(input("Enter in max epochs:"))
   
    p = Perceptron(titanic[0], titanic[1], num_epochs, learning_rate, titanic[2])  #self, tr_inpt, labels, epoch, Lr, si
    #testIt(self, testDat, testLabels)
    
    print("Accuracy", p.testIt(titanic[3], titanic[4]), "%")
    
main()