import matplotlib.pyplot as plt
import numpy as np
import random
import re
import string
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

characters = string.digits + string.ascii_uppercase
print(characters)
width, height, n_len, n_class = 200, 80, 4, len(characters)+1

from keras import backend as K
# ctc loss
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
# network structrue
from keras.models import *
from keras.layers import *
rnn_size = 128

input_tensor = Input((width, height, 3))
x = input_tensor
for i in range(3):
    x = Convolution2D(32, 3, 3, activation='relu')(x)
    x = Convolution2D(32, 3, 3, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

conv_shape = x.get_shape()
x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2]*conv_shape[3])))(x)

x = Dense(32, activation='relu')(x)

gru_1 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru1')(x)
gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru1_b')(x)
gru1_merged = merge([gru_1, gru_1b], mode='sum')

gru_2 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru2')(gru1_merged)
gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru2_b')(gru1_merged)
x = merge([gru_2, gru_2b], mode='concat')
x = Dropout(0.25)(x)
x = Dense(n_class, init='he_normal', activation='softmax')(x)
base_model = Model(input=input_tensor, output=x)

labels = Input(name='the_labels', shape=[n_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])

model = Model(input=[input_tensor, labels, input_length, label_length], output=[loss_out])
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')
# load weights
model.load_weights('model.h5')

def evaluate(model, batch_num):
	X = np.zeros((128, width, height, 3), dtype=np.uint8)
	num = batch_num + 9000
	for i in range(1):
		im = plt.imread('/home/alex/Desktop/data-3/train/'+ str(num) + '.jpg')
		X[i] = np.array(im).transpose(1, 0, 2)
		X_test = X
		y_pred = base_model.predict(X_test)
		shape = y_pred[:,2:,:].shape
		out = K.get_value(K.ctc_decode(y_pred[:,2:,:], input_length=np.ones(shape[0])*shape[1])[0][0])[:, :4]
		out = ''.join([characters[x] for x in out[0]])
		return num, out
f = open('mappings_test_clean.txt', 'w')
for i in range(1000):
	num, out = evaluate(base_model, i)
	print(str(num) + ',' + out)
	f.write(str(num) + ',' + out)
	f.write('\n')
f.close()
