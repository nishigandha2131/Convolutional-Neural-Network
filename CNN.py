#importing libraries
#In CNN we have various functions like the convolution, Pooling
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical

import matplotlib.pyplot as plt

#Load the data
(X_train, Y_train), (X_test,Y_test) = mnist.load_data()

#Data Preprocessing
#Mnisht images have depth of 1 so that should be declared
num_classes = 10
epocs = 3
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)
X_train.astype('float32')
X_test.astype('float32')
X_train =X_train/255.0
X_test = X_test/255.0
Y_train=to_categorical(Y_train,num_classes)
Y_test=to_categorical(Y_test,num_classes)

#Creating a CNN model
cnn = Sequential()
#Add the convolution layers
cnn.add(Conv2D(32,kernel_size=(5,5), input_shape=(28,28,1), padding='same', activation='relu'))
cnn.add(MaxPooling2D())
cnn.add((Conv2D(64,kernel_size=(5,5),padding='same',activation='relu')))
cnn.add(MaxPooling2D())
cnn.add(Flatten())
cnn.add(Dense(1024,activation='relu'))
cnn.add(Dense(10,activation='softmax'))

#Compile the model
cnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
cnn.summary()

#Train the model
history=model.fit(X_train,Y_train,epochs=20,validation_data=(X_test,Y_test))

#Checking the accuracy of the model using evaluate method
score = model.evaluate(X_test,Y_test)
print(score)