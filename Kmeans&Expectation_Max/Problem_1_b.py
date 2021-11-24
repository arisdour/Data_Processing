from scipy.io import loadmat 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(1047398)

##################################### Import Data ##############################################
N=200
K=2
data = loadmat('hw3-1data.mat')   ##read matlab struct
data=data['X']
data=data.transpose()
# Euclidean Distance 
def dist(a, b, ax=1):
    d=np.linalg.norm(a - b, axis=ax)
    return d*d


zero=a=np.zeros((1,2))
d_3 = np.zeros(len(data))
for  i in range(len(data)):
    d_3[i]=dist(data[i] , zero)

data=pd.DataFrame(data)
d_3=pd.DataFrame(d_3)
New_Data=pd.concat([data, d_3] , axis=1)  
New_Data=pd.DataFrame(New_Data).to_numpy()

kmeans = KMeans(n_clusters=2,init='random' ,random_state=0 , algorithm='full').fit(New_Data)

Labels=kmeans.labels_
centroids=kmeans.cluster_centers_


Classes=np.split(Labels,2)
Class0=Classes[0]
Class1=Classes[1]

c0=np.count_nonzero(Class0 == 1)
c1=np.count_nonzero(Class1 == 0)
Error_rate1=c0/100
Error_rate2=c1/100
print("Error Class 1 : \n" , Error_rate1)
print("Error Class 2 : \n" , Error_rate2)
print("Total Error   : \n" , (Error_rate2+Error_rate1)*0.5)





##################################### Plot Results ###################################################


fig=plt.figure(dpi=150)
ax=Axes3D(fig)
##Original Data
ax.scatter(New_Data[:100,0],New_Data[:100,1],New_Data[:100,2],color='violet',s=35)
ax.scatter(New_Data[100:200,0],New_Data[100:200,1],New_Data[100:200,2],color='cyan',s=35)
##Predicted Data
ax.scatter(New_Data[Labels==0,0],New_Data[Labels==0,1],New_Data[Labels==0,2],color='violet',s=36 ,linewidths=5)
ax.scatter(New_Data[Labels==1,0],New_Data[Labels==1,1],New_Data[Labels==1,2],color='cyan',s=36 ,linewidths=5)


##Centroids
ax.scatter(centroids[0,0],centroids[0,1],centroids[0,2],color='purple',marker='x',s=150)
ax.scatter(centroids[1,0],centroids[1,1],centroids[1,2],color='blue',marker='x',s=150)


