import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1047398)


############################################################################### Bayes Hypothesis Testing ############################################################################

mean=(0,0)
cov=[[1 , 0],[0 , 1]]
fi_0=np.random.multivariate_normal(mean , cov ,1000000)
mean=(-1,1)
cov=[[1 , 0],[0 , 1]]
fi_1=np.random.multivariate_normal(mean , cov ,1000000)


def pdf(array_1 ,array_2, i):   
    
    f0_x1 = 1/(1 * np.sqrt(2 * np.pi)) *np.exp( - (array_1[i][0])**2 / (2))
    f0_x2 = 1/(1 * np.sqrt(2 * np.pi)) *np.exp( - (array_1[i][1])**2 / (2))
    f1_x1 = 0.5/(1 * np.sqrt(2 * np.pi)) *np.exp( - (array_2[i][0] -1 )**2 / (2 )) + 0.5/(1 * np.sqrt(2 * np.pi)) *np.exp( - (array_2[i][0] +1 )**2 / (2))
    f1_x2 = 0.5/(1 * np.sqrt(2 * np.pi)) *np.exp( - (array_2[i][1] -1 )**2 / (2 )) + 0.5/(1 * np.sqrt(2 * np.pi)) *np.exp( - (array_2[i][1] +1 )**2 / (2))
    F0=f0_x1*f0_x2
    F1=f1_x1*f1_x2
    r=F1/F0
    return r


H1=0
H0=0
N=1000000
for j in range(N):

    r=pdf(fi_0,fi_0, j)
    
    if r>1 or r==1:
        H1=H1+1
    else:
        H0=H0+1

H0error= H1/N
print("H1: " , H1 , "H0",  H0 , "Error:", H0error ,"\n")

H1=0
H0=0
for j in range(N):

    r=pdf(fi_1,fi_1, j)
    if r>1 or r==1:
        H1=H1+1
    else:
        H0=H0+1

H1error= H0/N
print("H1: " , H1 , "H0",  H0 , "Error:", H1error ,"\n")

print("Error Sum:", 0.5*H1error+0.5*H0error)














