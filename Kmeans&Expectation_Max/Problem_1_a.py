from scipy.io import loadmat 
from copy import deepcopy
import numpy
import numpy as np # linear algebra
from matplotlib import pyplot as plt


np.random.seed(1047398)
##################################### Import Data ##############################################

data = loadmat('hw3-1data.mat')   ##read matlab struct
data=data['X']
data=data.transpose()

##################################### K-Means ###################################################

K=2 #Number of Clusters
N = data.shape[0]                                                              # Length of data
M = data.shape[1]                                                              # Features of data


### Choose Centroids ###


centroids = np.random.uniform( (-4) , size=(2,2))

# Euclidean Distance 
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)



centroids_old = np.zeros(centroids.shape)
labels = np.zeros(N)                                                           # Cluster Labels
error = dist(centroids, centroids_old,None)                                    # If 0 terminates Alg.

while error != 0 :
    for i in range(N):
        distances = dist(data[i] , centroids)
        cluster = np.argmin(distances)                                         # Returns position of minimum in distances array 
        labels[i] = cluster
       
    centroids_old=deepcopy(centroids)

    for i in range(K):                                                         # Calculate new centroids    
                                                                          
            points = [data[j] for j in range(N) if labels[j] == i]

            centroids[i] = np.mean(points, axis=0)

            error = dist(centroids, centroids_old, None)


##################################### Plot Results ###################################################
            
colors = ['violet', 'cyan']
fig, ax = plt.subplots(dpi=150)
##Original Data
ax.scatter(data[:100,0],data[:100,1],s=80,color='violet' ,edgecolor='none')                      
ax.scatter(data[100:200,0],data[100:200,1],s=80,color='cyan',edgecolor='none')
##Predicted Data
ax.scatter(data[labels==0,0],data[labels==0,1],s=81,edgecolor='violet' , facecolors="none")
ax.scatter(data[labels==1,0],data[labels==1,1],s=81,edgecolor='cyan' , facecolors="none")
        
##Centroids
ax.scatter(centroids[0, 0], centroids[0, 1], marker='x', s=150, c='purple')
ax.scatter(centroids[1, 0], centroids[1, 1], marker='x', s=150, c='blue')

##################################### Calculate Errors ################################################
Classes=numpy.split(labels,2)
Class0=Classes[0]
Class1=Classes[1]

c0=np.count_nonzero(Class0 == 0)
c1=np.count_nonzero(Class1 == 1)
Error_rate1=c0/100
Error_rate2=c1/100
print("Error Class 1 : \n" , Error_rate1)
print("Error Class 2 : \n" , Error_rate2)
print("Total Error   : \n" , (Error_rate2+Error_rate1)*0.5)
