# Group 23

# Code for MLP network for season by season experiment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

class MultiLayer_Perceptron:
    def __init__(self,layers,eta):
        self.layers=layers;
        self.num_layers=len(layers);
        self.eta=eta
        self.initializeWeights();
        
    def initializeWeights(self):
        np.random.seed(0)
        self.W1 = np.random.rand(self.layers[0], self.layers[1]) # (no_inputs * no_hidden_neurons) weight matrix from input to hidden layer
        self.W2 = np.random.rand(self.layers[1], self.layers[2]) #(no_hiddenneurons * no_output_neurons) +1 for bias input

    def sigmoid(self,a):
        return ( 1.0 / (1.0 + np.exp(-a)))
    
    def softmax(self,a):
        exps=np.exp(a)
        return exps/np.sum(exps,axis=0)
        
    def derivative_softmax(self,a):
        return a*(1-a)
    
    def derivative_sigmoid(self,a):
        return a*(1-a)
    
    def forward(self,data):
        self.y1 = np.dot(data,self.W1)
        self.y2 = self.sigmoid(self.y1) # activation function
        self.y3 = np.dot(self.y2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
        output = self.softmax(self.y3) 
        return output
        
    def backwardPropagation(self,data,target,output):
        self.error1=target-output
        self.delta1=self.error1*self.derivative_softmax(output) 
        self.error2=self.delta1.dot(np.atleast_2d(self.W2).T) 
        self.delta2=self.error2*self.derivative_sigmoid(self.y2) 
        
        self.W1 += self.eta*((np.atleast_2d(data).T).dot(np.atleast_2d(self.delta2)))
        self.W2 +=(self.eta)*(np.atleast_2d(self.y2)).T.dot(np.atleast_2d(self.delta1))
        #print(self.cross_entropy_loss(data,target,output))
    
    def train(self,data,target):
        for x,y in zip(data,target):
            output=self.forward(x)
            self.backwardPropagation(x,y,output)
    
    def predict(self,data):
       return self.forward(data)
   
    def accuracy(self,target,output):
        correct=0
        n=len(target)
        for i in range(n):
            if(output[i]==target[i]):
                correct+=1
        print( correct / float(len(target)) * 100)

        
    def test(self,data,target):
        output1=self.forward(data)
        output2=np.argmax(output1,axis=1)-1
        #print(accuracy_score(target,output2))
        #print(confusion_matrix(target,output2))
        print(classification_report(target,output2))
        self.accuracy(target,output2)
        
    
    def cross_entropy_loss(self,data,target,output):
        m=target.shape[0]
        return (-(1.0/m) * np.sum(target*np.log(output) + (1-target)*np.log(1-output)))
        
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
df_train.insert(loc=7,column='bias_input',value=1)
df_train['bias_input']=pd.to_numeric(df_train['bias_input'], errors='coerce').astype(np.int64)

df_test['Average_runs']=pd.to_numeric(df_test['Average_runs'], errors='coerce').fillna(0.0).astype(np.float)
df_test['Average_wickets']=pd.to_numeric(df_test['Average_wickets'], errors='coerce').fillna(0.0).astype(np.float)
df_test['Performance_rating']=pd.to_numeric(df_test['Performance_rating'], errors='coerce').fillna(-1.0).astype(np.float)
df_test['Player_Id']=pd.to_numeric(df_test['Player_Id'], errors='coerce').fillna(0).astype(np.int64)
df_test['Runs_Scored']=pd.to_numeric(df_test['Runs_Scored'], errors='coerce').fillna(0).astype(np.int64)
df_test['Wickets_Taken']=pd.to_numeric(df_test['Wickets_Taken'], errors='coerce').fillna(0).astype(np.int64)
df_test['Outs']=pd.to_numeric(df_test['Outs'], errors='coerce').fillna(0).astype(np.int64)
df_test['Matches_Played']=pd.to_numeric(df_test['Matches_Played'], errors='coerce').fillna(0).astype(np.int64)

df_test.insert(loc=7,column='bias_input',value=1)
df_test['bias_input']=pd.to_numeric(df_test['bias_input'], errors='coerce').astype(np.int64)

X = df_train.iloc[1:,1:-1].values
y=df_train.iloc[1:,-1].values
Y=np.zeros((len(y),3))
for i in range(len(y)):
    if(y[i]==-1):
        Y[i][0]=1
    elif(y[i]==0):
        Y[i][1]=1
    else:
        Y[i][2]=1

X1 = df_test.iloc[1:,1:-1].values
y1=df_test.iloc[1:,-1].values
Y1=np.zeros((len(y1),3))
for i in range(len(y1)):
    if(y[i]==-1):
        Y1[i][0]=1
    elif(y[i]==0):
        Y1[i][1]=1
    else:
        Y1[i][2]=1
        
layers=np.array([7,7,3]) #using 7 neurons in the hidden layer gives the best accuracy
mppn=MultiLayer_Perceptron(layers,0.01)
for i in range(100):
    mppn.train(X,Y)
mppn.test(X1,y1)
   
    
    
    
    
    
    
   