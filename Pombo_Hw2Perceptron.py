# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 12:36:45 2019

@author: danie
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pan
import seaborn as sns

class AdalineSGD:
    def __init__(self, tr_inpt, labels, epoch, Lr):
        #Intialize parameters
        self.epoch = epoch
        self.tr_inpt = tr_inpt
       
        if self.tr_inpt.ndim == 1:#If self.tr_input.shape == (a, ) where a member of integer set
            self.sz = 1 #For 1D array there is 1 column = column length/width is 1 rather than nothing
        else:
            self.sz = self.tr_inpt.shape[1] #Length of row (number of features per sample)
        #.shape returns a tuple and element at index 1 indicates the length of the row
        
        self.w = self.weights() #weights 
        self.Lr = Lr #Learning rate
        self.labels = labels #Training data labels
        
        self.learning() #Learning algorithm execution
        self.plotErrors_Cost() #Generate log(cost) vs epoch graph
    
        
    def weights(self):
        #where sz is the size of x (number of x)
        self.w = np.random.random(self.sz+1)

        #random.randfl ? generate float of random wieght
        return self.w
    
    def z(self, x):
        #generate dot product between w and features x
        return np.dot(self.w[1:], x) + self.w[0]
    # return np.dot(np.transpose(self.w),x)
       
    def Id(self,z_): #Identity function - Activation function
        return z_
       
    def learning(self):
        self.cost = []
        for e in range(self.epoch):
            cst = 0
            for k in range(self.tr_inpt.shape[0]):
                
                X = self.tr_inpt[k] #Row k
                Z = self.z(X) #Net input
                
                error = (self.labels[k]-self.Id(Z)) #
                dw = self.Lr*error*X 
                
                self.w[1:] += dw
                self.w[0] += self.Lr*error
                
                cst += .5*(error**2)
               
            self.cost.append(cst)
        
    def quantizer(self, z):
        if z >= 0.0:
            return 1
        else:
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
       
    def plotErrors_Cost(self):
        costFig = plt.figure() #Initalizes new plot
        plt.title("Number of updates vs Epochs") 
        plt.plot(range(1,len(self.cost)+1), np.log10(self.cost)) #range(1,len(self.updates)+1) is the epochs
        #x = epochs, y = self.updates (number of updates per epoch)
        plt.xlabel('Epochs')
        plt.ylabel("Log(cost)")
        plt.show() #Generates/shows the plot
            

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
                
                update_num += int(self.Lr*error != 0.0) #Increments the updates, inspired by textbook "Python Machine Learning" by Sebastian Raschka
                
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
        errorUpdateFig = plt.figure() #Initalizes new plot
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
        
        return(np.array(train['Sex']), np.array(train_labels), None , np.array(test['Sex']), np.array(test_labels))
        
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
        
        
def LinSep_data():
    #sex, height, species
    #male 0 female 1, 15-110 cm, toy dog species 0 regular/largedog species 1
    #linearly sperable
    
    #Create array of dog data where index 0 indicates PID, index 1 is gender, index 2 is height, index 3 is species
    d = [[45, 1], [37, 1], [123, 1],[134,1],
         [48,1],[99, 1], [78, 1], [35, 1], 
         [88,1], [67,1], [40,1],[56,1], 
         [34, 1],[89, 1], [69,1], [18, 0], 
         [23, 0], [30, 0], [20, 0], [19, 0], 
         [18, 0], [16,0],[32, 0], [25,0], 
         [24, 0],[13,0], [12,0], [13,0]]
            
    dogs = pan.DataFrame(d, columns =["Height", "Type"]) #Create pandas data frame
    
    #Generates categorical scatter plot (looks like bar chart and scatter plot hybrid)
    dds = sns.catplot(x='Type', y="Height", hue = 'Type', data = dogs)
    dds
    
    #Generates categorical scatter plot w/ boundry lines
    sh = sns.lmplot(x = 'Type', y="Height", hue = 'Type', data = dogs)
    sh
    
    
    testD = []
    testL = []
    trainD =[]
    trainL =[]

    
    np.random.shuffle(d)
    
    #Creating test data
    for k in range(0,len(d), 3):
        testD.append(np.array(d[k][0]))
        testL.append(np.array(d[k][1]))
        #testnp = np.append(testnp, np.array(d[k][0]), axis = 1)
    rshp_test = len(testD)
    
    np.random.shuffle(d)
    #Creating training data
    for k in range(0,int(len(d)*.70)):
        trainD.append(np.array(d[k][0]))
        trainL.append(np.array(d[k][1]))

    rshp_train = len(trainD)
    
    sz = len(testD)

    #print(testDnp)
    return(np.array(testD).reshape(rshp_test,1), np.array(testL).reshape(rshp_test,1), sz, np.array(trainD).reshape(rshp_train,1), np.array(trainL).reshape(rshp_train,1))

def Not_linSep_data():
    #male 0 female 1, 15-110 cm, cats 0 dogs 1
    #NOT linearly sperable
    p = [[45, 1], [20, 1], [123, 1], [45, 1], 
    [78, 1], [35, 1], [34, 1], [89, 1], 
    [69,1], [18, 0], [23, 0], [30, 1], 
    [20, 0], [19, 0], [18, 1], [16,1], 
    [32, 0], [25,0], [24, 1], [13,1], 
    [12,1], [13,1], [134,1], [56,1], 
    [48,1], [23,0],[23,0], [25,0], 
    [34,0], [27,0], [16,0], [34, 0], 
    [13,1], [10, 1], [42,0], [21,0]]
    
    pets = pan.DataFrame(p, columns =["Height", "Type"]) #Create pandas data frame
    
    #Generates categorical scatter plot (looks like bar chart and scatter plot hybrid)
    ls = sns.catplot(x='Type', y="Height", hue = 'Type', data = pets)
    ls
    
 
    #Generates categorical scatter plot w/ boundry lines
    hs = sns.lmplot(x="Type", y="Height", hue = 'Type', data = pets)
    hs
    
    testD = []
    testL = []
    trainD =[]
    trainL =[]
    
    
    np.random.shuffle(p)
    
    #Creating training data sets
    for k in range(0, int(len(p)*.70)):
        trainD.append(np.array(p[k][0]))
        trainL.append(np.array(p[k][1]))
        
    rshp_train = len(trainD)
    
    #Creating test data sets
    for k in range(int(len(p)*.70),len(p)):
        testD.append(np.array(p[k][0]))
        testL.append(np.array(p[k][1]))
        
    rshp_test = len(testD)
    
    sz = len(trainD)
   
    return(np.array(testD).reshape(rshp_test,1), np.array(testL).reshape(rshp_test,1), sz, np.array(trainD).reshape(rshp_train,1), np.array(trainL).reshape(rshp_train,1))

def main():
    """learning_rate = 0.0003#float(input("Enter in learning rate:"))
    
    num_epochs = 500#int(input("Enter in max epochs:"))"""
    
    print("Titanic")
    TD = TitanicData("train.csv", False, False) 
    """Returns tuple of (trainingData, trainingLabels, size, testData, testDataLabels)"""
        
    titanic = TD.import_clean_TitanicData() #Returns tuple of (trainingData, trainingLabels, size, testData, testDataLabels)
    
    print("Perceptron")
    
    p = Perceptron(titanic[0], titanic[1], 1000, 0.003)  #self, tr_inpt, labels, epoch, Lr
    print("Titanic: Perceptron accuracy", p.testIt(titanic[3], titanic[4]), "%")
    
    print("Adaline Stochastic Gradient Descent")
    
    a = AdalineSGD(titanic[0], titanic[1], 1000, 0.0001)
    print("Titanic: AdalineSGD accuracy", a.testIt(titanic[3], titanic[4]), "%")
    
    print("Linearly Seperable")
    seperable = LinSep_data() #testD, testL, trainD, trainL

    pDogs = Perceptron(seperable[3], seperable[4], 1000, 0.001)  #self, tr_inpt, labels, epoch, Lr
    print("linear: Perceptron accuracy", pDogs.testIt(seperable[0], seperable[1]), "%")
    
    print("Nonlinearly Seperable") 
    notSep = Not_linSep_data() #testD, testL, sz, trainD, trainL
        
    pets = Perceptron(notSep[3], notSep[4], 250, 0.001)
    print("Nonlinear: Perceptron accuracy", pets.testIt(notSep[0], notSep[1]), "%")
    
main()