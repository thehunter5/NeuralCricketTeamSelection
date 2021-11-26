# Group 23

# Code to implement the perceptron network for all the seasons combined
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
'''with open('C:/Users/user/Downloads/indian-premier-league-csv-dataset/data/season9.csv','w') as f:
    for l in lines:
         f.write('('+l+'),\n')'''
         #shuffle=True, random_state=None,
    
class linear_network:
    def __init__(self,eta=0.01,n_iter=1000):  # new
        self.eta = eta
        self.n_iter = n_iter
       
        
    def initializeWeights(self,data):
        self.w=np.zeros(1 + data.shape[1]) # +1 has been done to include bias weight

    def train(self,data,target):
        for epoch in range(self.n_iter):
            for x, y in zip(data, target):
                delta= self.eta * (y - self.predict(x))
                self.w[1:] += delta * x
                self.w[0] += delta
        return self


    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w[1:]) + self.w[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
    def test(self,data,target):
        y=np.dot(data,self.w[1:])+self.w[0]
        output=np.where(y>=0.0,1,-1)
        print(accuracy_score(target,output))
        print(confusion_matrix(target,output))
        print(classification_report(target,output))
        
        self.accuracy(target,output)
    
    def accuracy(self,target,output):
        correct=0
        n=len(target)
        for i in range(n):
            if(output[i]==target[i]):
                correct+=1
        print( correct / float(len(target)) * 100)

  
df=pd.read_csv('C:/Users/user/Downloads/indian-premier-league-csv-dataset/all9seasons.csv')


df['Average_runs']=pd.to_numeric(df['Average_runs'], errors='coerce').fillna(0.0).astype(np.float)
df['Average_wickets']=pd.to_numeric(df['Average_wickets'], errors='coerce').fillna(0.0).astype(np.float)
df['Performance_rating']=pd.to_numeric(df['Performance_rating'], errors='coerce').fillna(-1.0).astype(np.float)
df['Player_Id']=pd.to_numeric(df['Player_Id'], errors='coerce').fillna(0).astype(np.int64)
df['Runs_Scored']=pd.to_numeric(df['Runs_Scored'], errors='coerce').fillna(0).astype(np.int64)
df['Wickets_Taken']=pd.to_numeric(df['Wickets_Taken'], errors='coerce').fillna(0).astype(np.int64)
df['Outs']=pd.to_numeric(df['Outs'], errors='coerce').fillna(0).astype(np.int64)
df['Matches_Played']=pd.to_numeric(df['Matches_Played'], errors='coerce').fillna(0).astype(np.int64)


X = df.iloc[1:,1:-1].values
y=df.iloc[1:,-1].values
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)

#X1 = df_test.iloc[1:,1:-1].values
#y1=df_test.iloc[1:,-1].values

ppn = linear_network(0.01,2000)
ppn.initializeWeights(X)
ppn.train(X_train, Y_train)
ppn.test(X_validation,Y_validation)