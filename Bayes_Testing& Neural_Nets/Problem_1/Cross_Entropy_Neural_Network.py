import keras 
import numpy as np
import tensorflow as tf 
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense 
import matplotlib.pyplot as plt
np.random.seed(1047398)




############################################################################### Test Sets ############################################################################
mean=(0,0)
cov=[[1 , 0],[0 , 1]]
fi_0=np.random.multivariate_normal(mean , cov ,1000000)
mean=(-1,1)
cov=[[1 , 0],[0 , 1]]
fi_1=np.random.multivariate_normal(mean , cov ,1000000)

test_x=np.concatenate((fi_0, fi_1), axis=0)



test_y1 = []
for i in range(1000000):
    test_y1.append(0)
    
test_y2 = []
for i in range(1000000):
    test_y2.append(1)
test_y=np.concatenate((test_y1, test_y2), axis=0) 

############################################################################### Train Sets ############################################################################
mean=(0,0)
cov=[[1 , 0],[0 , 1]]
f0=np.random.multivariate_normal(mean , cov ,200)
mean=(-1,1)
cov=[[1 , 0],[0 , 1]]
f1=np.random.multivariate_normal(mean , cov ,200)

train_x=np.concatenate((f0, f1), axis=0)


Train_Hypothesis_0 = []
for i in range(200):
    Train_Hypothesis_0.append(0)
    
Train_Hypothesis_1 = []
for i in range(200):
    Train_Hypothesis_0.append(1)

train_y=np.concatenate((Train_Hypothesis_0, Train_Hypothesis_1), axis=0)    

############################################################################### Neural Network Training ##################################################################




                                                                                    ## Neural Network ##

##Create and Compile Model
                                                                                    
                                                                                    
model = Sequential()
keras.initializers.RandomNormal(mean=0.0, stddev=1/(2+20), seed=1047398)
keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
model.add(Dense(20, input_dim=2, activation='relu', kernel_initializer='RandomNormal'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer="SGD", metrics=['accuracy'])


##Train Model

history=model.fit(train_x , train_y , epochs=5000, batch_size=10)


loss_history = history.history["loss"]  ## Calculate the average
data={"loss": loss_history}
df=df = pd.DataFrame(data)
df=df.rolling(window=20).mean()
print(df.tail(10) ,"\n")


############################################################################### Test Model ##################################################################

Results = model.evaluate(test_x, test_y)
print('Accuracy: ', Results[1] )
print('Loss ', Results[0])


