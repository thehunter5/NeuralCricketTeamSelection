# Group 23

# Code to implement the RBF network for season by season

import numpy as np
import random
import pandas as pd
from sklearn.metrics import accuracy_score

# define radial basis function
def get_radial_basis(a,b,beta):
    euc_dist = np.linalg.norm(a-b)
    return np.exp(- beta*euc_dist**2);

# read the dataset
rd = pd.read_csv('C:/Users/user/Downloads/indian-premier-league-csv-dataset/season1.csv', sep =',' , header = 0)
input = rd.values
x = input[ : ,1:-1]
y = input[ : ,7]

rd1 = pd.read_csv('C:/Users/user/Downloads/indian-premier-league-csv-dataset/season8.csv', sep =',' , header = 0)
input = rd1.values
x2 = input[ : ,1:-1]
y2 = input[ : ,7]

#initializing the number of hidden neurons required.
number_of_hidden_neurons = 15


# x is the input and y is the corresponding label for each input.
number_of_inputs = x.shape[0];
number_of_features = x.shape[1];
random_nums = random.sample(range( number_of_inputs), number_of_hidden_neurons)
# setting the centers
centers = [x[arg] for arg in random_nums]
centers = np.asarray( centers)
Z =  centers
# finding the maximum distance between the centers
X,Y = np.atleast_2d(Z[:,0]), np.atleast_2d(Z[:,1])
D = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)
D_max = D.max();
# calculating the value of sigma and beta
sigma = D_max/np.sqrt(2* number_of_hidden_neurons)
beta = 1/(2* sigma**2)
# creating the radial basis matrix for pseudo-inversion
radial_basis_matrix = np.zeros(( number_of_inputs, number_of_hidden_neurons));
for i in range(0,number_of_inputs):
    for j in range(0,number_of_hidden_neurons):
         radial_basis_matrix[i,j] =  get_radial_basis(x[i],  centers[j],beta);
         
# employing the pseudo inversion technique which provides solution for method of least squares problem to
# train weights.
weights = np.dot(np.linalg.pinv( radial_basis_matrix),y)
#print(weights)
#print(weights.shape)
#testing data

number_of_inputs = x2.shape[0];
number_of_features = x2.shape[1];

radial_basis_matrix = np.zeros(( number_of_inputs, number_of_hidden_neurons));
for i in range(0,number_of_inputs):
    for j in range(0,number_of_hidden_neurons):
         radial_basis_matrix[i,j] =  get_radial_basis(x2[i],  centers[j], beta);

output = np.dot(radial_basis_matrix, weights)
#print(output)
for i in range(output.shape[0]):
	if output[i]>=0.5 :
		output[i]= 1
	elif output[i]<0.5 and output[i]>-0.5 :
		output[i]= 0
	else:
		output[i] = -1
# uncomment to print the outputs
# print(output)
# print(output.shape)
# print(y2.shape)
acc = accuracy_score(y2, output)
print(acc)

