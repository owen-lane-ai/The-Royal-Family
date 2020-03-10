# import statements
from __future__ import print_function
import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras import backend as K
from keras.datasets import cifar10           # this data set will be used for baseline implementation only
import numpy as np
import matplotlib.pyplot as plot


# set up train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# preprocessing of data, image pixels must be between 0-255
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


num_classes = y_test.shape[1]     # this will be changed when using our own data
batch_size = 128                  # can be changed
epochs = 12                       # can be changed
optimizer = 'adam'                # can be changed

# creating the model
model = Sequential()
# useful link for Conv2D: https://keras.io/layers/convolutional/
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=X_train.shape[1:]))
model.add(Dropout(0.2))                                            # randomly eliminates 20% of existing connections
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))       # NOTE: keep filter size a multiple of 2  (eg 32 64)
model.add(MaxPooling2D(pool_size=(2, 2)))             # useful link for pooling layers https://keras.io/layers/pooling/
model.add(Dropout(0.2))                               # randomly eliminates 20% of existing connections
model.add(BatchNormalization())
model.add(Flatten())                           # Matrix to vector conversion
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

# compile the model
model.compile(loss=categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

# train and fit the model
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

