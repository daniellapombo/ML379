#https://www.kaggle.com/c/titanic
#import NumPy
#load dataset into NumPy array and then try to ID the attributes that will lead to the passengers to surive or die
# as long as make resonable attempt to solve it will do fine, will not be grades on completion of solving the problem
import numpy as np
import matplotlib.pyplot as plt
import pandas as pan

#conda update pandas


ti_file =pan.read_csv("C:\\Users\\danie\\Documents\\School\\CS\\COMP 379\Hw\\train.csv")
col = list(ti_file.columns)

tiDat = ti_file.iloc[:, [0,1,2,4,5,6,7,9]]
up_col = list(tiDat.columns)
#print(up_col)
#['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

def stats():
    dat = tiDat.groupby(['Survived', 'Pclass', 'Sex'])['Survived'].count()
    print(dat)
    
    
    tisum = tiDat['Survived'].sum()
    print("Total passengers survived", tisum)

#not Survived column is an integer
def scat():
    print("")
    print("Blue survival, red is death")

    x = tiDat['PassengerId']
    y = tiDat['Pclass']
    color = ('red', 'blue')
    groups = (int(0), int(1))
    plt.scatter(x, y, marker = 'o', label = groups, color = color)
    plt.xlabel("PassengerId")
    plt.ylabel("Pclass")
    plt.show()
    

    x = tiDat['Pclass']
    y = tiDat['Fare']
    color = ('red', 'blue')
    groups = (int(0), int(1))
    plt.scatter(x, y, marker = 'o', label = groups, color = color)
    plt.xlabel("Pclass")
    plt.ylabel("Fare")
    plt.show()

    x = tiDat['PassengerId']
    y = tiDat['Age']
    color = ('red', 'blue')
    groups = (int(0), int(1))
    plt.scatter(x, y, marker = 'o', label = groups, color = color)
    plt.xlabel("Passenger")
    plt.ylabel("Age")
    plt.show()
    
    
    x = tiDat['Age']
    y = tiDat['Fare']
    color = ('red', 'blue')
    groups = (int(0), int(1))
    plt.scatter(x, y, marker = 'o', label = groups, color = color)
    #plt.scatter(x, y, marker = 'o', label = 'Survived')
    #plt.title("Pclass", "vs. survived")
    plt.xlabel("Age")
    plt.ylabel("Fare")
    plt.show()
    
    x = tiDat['Sex']
    y = tiDat['Fare']
    color = ('red', 'blue')
    groups = (int(0), int(1))
    plt.scatter(x, y, marker = 'o', label = groups, color = color)
    #plt.scatter(x, y, marker = 'o', label = 'Survived')
    #plt.title("Pclass", "vs. survived")
    plt.xlabel("Sex")
    plt.ylabel("Fare")
    plt.show()
    
#Print head of the input data
#print(tiDat.head())

lbls = []
for t in tiDat.iloc[:,1]:
    lbls.append(np.array(t))

convert = tiDat.iloc[:,2:]
#FEMALE IS 0 and MALE IS 1
convert.loc[convert['Sex'] == 'male', 'Sex'] = 0
convert.loc[convert['Sex'] == 'female', 'Sex'] = 1
#print(convert.head())
colP = list(convert.columns)
#['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

sz = convert.count().max()

training = []
#['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
for k in range(sz):
    training.append(np.array(convert.iloc[k,:]))

class Perceptron:
    def __init__(self, tr_inpt,epoch, Lr, labels, si):
        self.epoch = epoch
        self.tr_inpt = tr_inpt
        self.sz = self.tr_inpt.shape[1]
            #.shape returns a tuple and I want only the element in the 0 position of that tuple
        self.w = self.weights(self.sz)
        self.Lr = 0.001
        self.labels = labels

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
    
    def fit(self, test):
        num_rt = []
        count = 0
        for m in range(self.epoch):
            right = 0
            for k in range(self.tr_inpt.shape[0]): #pick one the number of rows to be length of iteration
                self.z_input(self.tr_inpt[k])
                z = self.z_input(self.tr_inpt[k])
                prediction = self.predict(z)
                target = self.labels[k]
                #Is this interpretation of updating weights correct?
                error = target - prediction
                dw = self.Lr*error*self.tr_inpt[k]
                self.w[1:] += dw # Is this correct?
                self.w[0] += self.Lr*error
                #Check....
                count += 1
                if prediction == target:
                    right += 1
            num_rt.append(right)
        accurate_percent = (sum(num_rt)/count)*100
    
        if test:
           test_result = []
           for k in range(self.tr_inpt.shape[0]):
               self.z_input(self.tr_inpt[k])
               prediction = self.predict(z)
               test_result.append(prediction)
           return test_result
        else:
            return accurate_percent
    

tr1 = []
lb1 = []

#Create split training data
for j in range(1, sz, 3):
    tr1.append(training[j])
    lb1.append(lbls[j])

#Convert input data into numpy array that is 2d
#Make the training data into numpy arrays
t1 = np.array(tr1)
l1 = np.array(lb1)

def main():
    #Run the Graphs    
    stats()
    scat()
    #def __init__(self, tr_inpt,epoch, Lr, labels, si)
    p = Perceptron(t1, 100, 0.001, l1, sz)
    print("Accuracy", (p.fit(False)), "%")
    print(p.fit(True))


main()