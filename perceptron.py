# Group 23

# Code to implement the perceptron network on season by season

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


    def netvalue(self, X):
        """Calculate net input"""
        return np.dot(X, self.w[1:]) + self.w[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.netvalue(X) >= 0.0, 1, -1)
    
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

  
df_train=pd.read_csv('C:/Users/user/Downloads/indian-premier-league-csv-dataset/season1.csv')
df_test=pd.read_csv('C:/Users/user/Downloads/indian-premier-league-csv-dataset/season8.csv')

df_train['Average_runs']=pd.to_numeric(df_train['Average_runs'], errors='coerce').fillna(0.0).astype(np.float)
df_train['Average_wickets']=pd.to_numeric(df_train['Average_wickets'], errors='coerce').fillna(0.0).astype(np.float)
df_train['Performance_rating']=pd.to_numeric(df_train['Performance_rating'], errors='coerce').fillna(-1.0).astype(np.float)
df_train['Player_Id']=pd.to_numeric(df_train['Player_Id'], errors='coerce').fillna(0).astype(np.int64)
df_train['Runs_Scored']=pd.to_numeric(df_train['Runs_Scored'], errors='coerce').fillna(0).astype(np.int64)
df_train['Wickets_Taken']=pd.to_numeric(df_train['Wickets_Taken'], errors='coerce').fillna(0).astype(np.int64)
df_train['Outs']=pd.to_numeric(df_train['Outs'], errors='coerce').fillna(0).astype(np.int64)
df_train['Matches_Played']=pd.to_numeric(df_train['Matches_Played'], errors='coerce').fillna(0).astype(np.int64)

df_test['Average_runs']=pd.to_numeric(df_test['Average_runs'], errors='coerce').fillna(0.0).astype(np.float)
df_test['Average_wickets']=pd.to_numeric(df_test['Average_wickets'], errors='coerce').fillna(0.0).astype(np.float)
df_test['Performance_rating']=pd.to_numeric(df_test['Performance_rating'], errors='coerce').fillna(-1.0).astype(np.float)
df_test['Player_Id']=pd.to_numeric(df_test['Player_Id'], errors='coerce').fillna(0).astype(np.int64)
df_test['Runs_Scored']=pd.to_numeric(df_test['Runs_Scored'], errors='coerce').fillna(0).astype(np.int64)
df_test['Wickets_Taken']=pd.to_numeric(df_test['Wickets_Taken'], errors='coerce').fillna(0).astype(np.int64)
df_test['Outs']=pd.to_numeric(df_test['Outs'], errors='coerce').fillna(0).astype(np.int64)
df_test['Matches_Played']=pd.to_numeric(df_test['Matches_Played'], errors='coerce').fillna(0).astype(np.int64)

X = df_train.iloc[1:,1:-1].values
y=df_train.iloc[1:,-1].values


X1 = df_test.iloc[1:,1:-1].values
y1=df_test.iloc[1:,-1].values

ppn = linear_network(0.01,100)
ppn.initializeWeights(X)
ppn.train(X, y)

ppn.test(X1,y1)


