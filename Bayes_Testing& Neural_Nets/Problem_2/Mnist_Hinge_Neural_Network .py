import keras 
import pandas as pd
import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense 

np.random.seed(1047398)

############################################################################### Test Sets ############################################################################


(x_train, y_train), (x_test, y_test) = mnist.load_data()
train_mask = np.isin(y_train, [0,8])
test_mask = np.isin(y_test, [0,8])
x_train, y_train = x_train[train_mask], np.array(y_train[train_mask] == 8)
x_test, y_test = x_test[test_mask], np.array(y_test[test_mask] == 8)


train_images = (x_train / 255) -0.5
test_images = (x_test / 255) -0.5

train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

y_train=np.where(y_train == 0,-1 ,y_train)   ## HINGE REPLACE VAL 0 WITH -1
y_test=np.where(y_test == 0,-1, y_test)

############################################################################### Neural Network Training ##################################################################

model = Sequential()
keras.initializers.RandomNormal(mean=0.0, stddev=1/785, seed=1047398)
model.add(Dense(300, input_dim=784, activation='relu', kernel_initializer='RandomNormal'))
model.add(Dense(1, activation='tanh'))
model.compile(loss="hinge", optimizer=keras.optimizers.adam(lr=0.01), metrics=['accuracy'])



history=model.fit(train_images , y_train , epochs=100, batch_size=32)


_, accuracy = model.evaluate(test_images,y_test)
print('Accuracy: %.2f' % (accuracy*100))

loss_history = history.history["loss"]
data={"loss": loss_history}
df=df = pd.DataFrame(data)
df=df.rolling(window=20).mean()
print(df.tail(10) ,"\n")



############################################################################### Test Model ##################################################################


Results = model.evaluate(test_images, y_test)
print('Hinge Accuracy :' , (Results[1]))
print('Hinge Loss :' , (Results[0]))

y_new = model.predict_classes(test_images)


a=np.count_nonzero(y_new == 1,axis = 0)
b=np.count_nonzero(y_test == 1,axis = 0)
c=np.count_nonzero(y_new == 0,axis = 0)
d=np.count_nonzero(y_test == -1,axis = 0)

#print("Predicted 1",a)
#print("Known 1",b)
#print("Predicted 0",c)
#print("Known 0",d)

print("H1 error = " ,  (abs(b-a))/b)
print("H0 error = " ,  (abs(d-c))/d)
