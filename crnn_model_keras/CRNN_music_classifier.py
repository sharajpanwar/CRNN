
# CRNN model
# this code takes highly from https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/music_tagger_crnn.py


from __future__ import print_function

import numpy as np
import h5py


import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Dropout,  Activation, ZeroPadding2D,Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.layers import Input
from keras.layers.recurrent import GRU

import os

os.environ["CUDA_VISIBLE_DEVICES"]= '0'
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# input image dimensions

batch = 16
classes = 50
epochs = 15
rows, cols = 96, 1366



########################################################################################################################
# uploading training_data

X_train = np.load('input_data/X_train.npy')
print('X_train_shape', X_train.shape)

Y_train = np.load('input_data/Y_train.npy')
print('Y_train_shape', Y_train.shape)
# let's shuffle the data

from random import shuffle

index = [i for i in range(X_train.shape[0])]
shuffle(index)
X_train = X_train[index, :,:]
Y_train = Y_train[index,]

# reshape the data to feed in the neural network
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, rows, cols)
    input_shape = (1, rows, cols)
else:
    X_train = X_train.reshape(X_train.shape[0], rows, cols, 1)
    input_shape = (rows, cols, 1)
print(X_train.shape)

########################################################################################################################
# uploading Testing data

X_test = np.load('input_data/X_test.npy')
print('X_test_shape', X_test.shape)
Y_test = np.load('input_data/Y_test.npy')
print('X_test_shape', Y_test.shape)

# reshape the data to feed in the neural network
if K.image_data_format() == 'channels_first':
    X_test = X_test.reshape(X_test.shape[0], 1, rows, cols)
    input_shape = (1, rows, cols)
else:
    X_test = X_test.reshape(X_test.shape[0], rows, cols, 1)
    input_shape = (rows, cols, 1)
print(X_test.shape)


########################################################################################################################
# uploading validation data

X_val = np.load('input_data/X_val.npy')
print('X_val_shape', X_val.shape)
Y_val = np.load('input_data/Y_val.npy')
print('Y_val_shape', Y_val.shape)

# reshape the data to feed in the neural network
if K.image_data_format() == 'channels_first':
    X_val = X_val.reshape(X_val.shape[0], 1, rows, cols)
    input_shape = (1, rows, cols)
else:
    X_val = X_val.reshape(X_val.shape[0], rows, cols, 1)
    input_shape = (rows, cols, 1)
print(X_val.shape)


########################################################################################################################

# one hot encoding of labels
Y_train = keras.utils.to_categorical(Y_train, classes)
print(Y_train.shape)
Y_test = keras.utils.to_categorical(Y_test, classes)
print(Y_test.shape)

# Model Definition

def music_classifier():

    input= Input(shape=input_shape)
    input_zero_padded = ZeroPadding2D(padding=(0, 37))(input)
    input_batch = BatchNormalization()(input_zero_padded)

    out_put_conv1 = Conv2D(64, (3, 3), padding='same')(input_batch)
    out_put_conv1 = BatchNormalization()(out_put_conv1)
    out_put_conv1 = Activation('elu')(out_put_conv1)
    out_put_conv1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(out_put_conv1)
    out_put_conv1 = Dropout(0.1)(out_put_conv1)

    out_put_conv2 = Conv2D(128, (3, 3), padding='same')(out_put_conv1)
    out_put_conv2 = BatchNormalization()(out_put_conv2)
    out_put_conv2 = Activation('elu')(out_put_conv2)
    out_put_conv2 = MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(out_put_conv2)
    out_put_conv2 = Dropout(0.1)(out_put_conv2)

    out_put_conv3 = Conv2D(128, (3, 3), padding='same')(out_put_conv2)
    out_put_conv3 = BatchNormalization()(out_put_conv3)
    out_put_conv3 = Activation('elu')(out_put_conv3)
    out_put_conv3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(out_put_conv3)
    out_put_conv3 = Dropout(0.1)(out_put_conv3)

    out_put_conv3 = Conv2D(128, (3, 3), padding='same')(out_put_conv3)
    out_put_conv3 = BatchNormalization()(out_put_conv3)
    out_put_conv3 = Activation('elu')(out_put_conv3)
    out_put_conv3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(out_put_conv3)
    out_put_conv3 = Dropout(0.1)(out_put_conv3)

    input_rnn = Reshape((15, 128))(out_put_conv3)

    out_put_rnn1 = GRU(32, return_sequences=True)(input_rnn)
    out_put_rnn2 = GRU(32, return_sequences=False)(out_put_rnn1)
    out_put = Dropout(0.3)(out_put_rnn2)

    out_put = Dense(classes, activation='sigmoid')(out_put)

    model = Model(inputs=input, outputs=out_put)

    return model

print("Starting to train.....\n\n")

model = music_classifier()

# Compile
print("Compiling model architecture.....\n\n")
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
              metrics=['accuracy'])

print (model.count_params())

print (model.summary())

# Fit the model
print("Fitting the model with train data.....\n\n")

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch, verbose=1, validation_data=(X_val, Y_val))

# Save model
print("Saving the model.....\n\n")
model.save('/saved_model/music_classifier.h5')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save('./classifier_model/cnn3_aug_model.h5')

# score, acc = model.evaluate(X_test, Y_test, batch_size=64)
# print('Test accuracy:', acc)
# target = model.predict(X_test, batch_size=32)
# auc = roc_auc_score(Y_test, target)
# print("auc_roc:", auc)
