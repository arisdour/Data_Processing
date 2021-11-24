from scipy.io import loadmat 
from sklearn.mixture import GaussianMixture
import numpy as np
from matplotlib import pyplot as plt
np.random.seed(1047398)
##################################### Import Data ##############################################

data = loadmat('hw3-1data.mat')   ##read matlab struct
data=data['X']
data=data.transpose()
gmm = GaussianMixture(n_components=2 )
##gmm = GaussianMixture(n_components=2 , init_params='random')                 #Model with random params init / Remove seed
gmm.fit(data)


print("Means :\n" ,gmm.means_)
print('\n')
print("Covariances :\n" ,gmm.covariances_)
print('\n')
print(" Weights :\n" ,gmm.weights_)
print('\n')
prob=gmm.predict_proba(data)
print(" Probabilities :\n" ,prob)

print('\n')
labels=gmm.predict(data)
Classes=np.split(labels,2)
Class0=Classes[0]
Class1=Classes[1]
##################################### Plots ##############################################
colors = ['purple', 'blue']

fig, ax = plt.subplots()

##Original Data
ax.scatter(data[:100,0],data[:100,1],s=80,color='violet' ,edgecolor='none')                      
ax.scatter(data[100:200,0],data[100:200,1],s=80,color='cyan',edgecolor='none')
##Predicted Data
ax.scatter(data[labels==0,0],data[labels==0,1],s=81,edgecolor='violet' , facecolors="none")
ax.scatter(data[labels==1,0],data[labels==1,1],s=81,edgecolor='cyan' , facecolors="none")

##################################### Error ##############################################
    
c0=np.count_nonzero(Class0 == 1)
c1=np.count_nonzero(Class1 == 0)
Error_rate1=c0/100
Error_rate2=c1/100
print("Error Class 1 : \n" , Error_rate1)
print("Error Class 2 : \n" , Error_rate2)
print("Total Error   : \n" , (Error_rate2+Error_rate1)*0.5)