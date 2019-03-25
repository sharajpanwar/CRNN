
# CRNN model for 'A deep learning approach for mapping music geners'
# We are using default settings of keras 2.2.4 
# we can check the setting with  $HOME/.keras/keras.json
# You should see:    
# {
#     "image_data_format": "channels_last",
#     "epsilon": 1e-07,
#     "floatx": "float32",
#     "backend": "tensorflow"
# }
# if you see different please change with above settings 

#Acknowledgement:
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
import matplotlib.pyplot as plt

# input image dimensions
batch = 32
classes = 50
epochs = 100
mel_bin, time = 96, 1366
########################################################################################################################

# function for uploading the data
def data_upload(x, y):
    x = np.load('/home/sharaj/PycharmProjects/music/input_data/'+x+'.npy')
    y = np.load('/home/sharaj/PycharmProjects/music/input_data/'+y+'.npy')
    # let's shuffle the data
    ind_list = [i for i in range(x.shape[0])]
    shuffle(ind_list)
    x = x[ind_list, :, :]
    y = y[ind_list,]
    # reshape the data to feed in the neural network
    if K.image_data_format() == 'channels_first':
        x = x.reshape(x.shape[0], 1, mel_bin, time)
        input_shape = (1, mel_bin, time)
    else:
        x = x.reshape(x.shape[0], mel_bin, time, 1)
        input_shape = (mel_bin, time, 1)
    return x, y, input_shape

# uploading training data
X_train, Y_train, input_shape = data_upload('X_train', 'Y_train')
print ('training data dimensions', X_train.shape, Y_train.shape)

# uploading Testing data
X_test, Y_test, input_shape  = data_upload('X_test', 'Y_test')
print ('testing data dimensions', X_test.shape, Y_test.shape)

# uploading validation data
X_train, Y_train, input_shape = data_upload('X_val', 'Y_val')
print ('validation data dimensions', X_val.shape, Y_val.shape)
########################################################################################################################

# Model Architecture
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

    out_put_conv4 = Conv2D(128, (3, 3), padding='same')(out_put_conv3)
    out_put_conv4 = BatchNormalization()(out_put_conv4)
    out_put_conv4 = Activation('elu')(out_put_conv4)
    out_put_conv4 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(out_put_conv4)
    out_put_conv4 = Dropout(0.1)(out_put_conv4)    
    
    # we are using default keras 2.2.4 i.e channel_last (mel_bin, time, channel), so we don't need to add 
    # permute block before reshaping,for channel_last add permute block as:
    # out_put_conv3 = Permute((3, 1, 2))(out_put_conv3)
        
    input_rnn = Reshape((15, 128))(out_put_conv4) # cnn output is reshaped to feed to RNN-GRU
    out_put_rnn1 = GRU(32, return_sequences=True)(input_rnn)
    out_put_rnn2 = GRU(32, return_sequences=False)(out_put_rnn1)
    out_put_rnn2 = Dropout(0.3)(out_put_rnn2)

    out_put = Dense(classes, activation='sigmoid')(out_put_rnn2)
    model = Model(inputs=input, outputs=out_put)
    return model

print("Starting to train.....\n\n")
model = music_classifier()

# Compile
print("Compiling model architecture.....\n\n")
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
              metrics=['accuracy'])
#count the total number of trainable parameters 
print (model.count_params())
#print the model
print (model.summary())

# Fit the model
print("Fitting the model with train data.....\n\n")
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch, verbose=1, validation_data=(X_val, Y_val))

# Save model
print("Saving the model.....\n\n")
model.save('/saved_model/music_classifier.h5')

#plotting the learning curve
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
