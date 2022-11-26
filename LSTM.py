import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
df = pd.read_csv("boat.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.set_index(['date'], drop=True)
# print(df.head(5))
print(list(df.loc['2016-10-17']))

#### 省事一点全部拆开写了，其实应该合在一起写的 但是不知道为啥搞了半天没搞好。。。


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# define input sequence
in_seq1 = array([7,4,7,5,4,5,1,6])
in_seq2 = array([7,5,3,9,3,4,6,5])
in_seq3 = array([7,7,11,5,7,6,13,12])
in_seq4 = array([5,5,12,6,8,8,9,9])
in_seq5 = array([15,6,11,7,3,9,4,11])
in_seq6 = array([11,14,12,5,5,11,7,7])
in_seq7 = array([18,13,7,8,4,16,10,14])
in_seq8 = array([10,8,7,15,14,11,10,20])
in_seq9 = array([14,13,16,19,16,7,12,17])
in_seq10 = array([16,9,8,14,16,14,8,18])
in_seq11 = array([11,10,6,10,15,12,12,8])
in_seq12 = array([15,10,6,16,8,10,10,10])
in_seq13 = array([12,7,7,11,10,12,5,8])
in_seq14 = array([13,11,6,7,11,16,9,10])
in_seq15 = array([14,8,7,7,8,12,11,10])
in_seq16 = array([11,7,9,6,8,7,8,11])
in_seq17 = array([8,8,10,4,6,5,9,9])
in_seq18 = array([11,7,4,3,4,5,4,7])
in_seq19 = array([5,5,4,4,5,8,8,7])
in_seq20 = array([0,5,4,4,6,6,5,9])
in_seq21 = array([0,2,2,3,6,8,6,6])
in_seq22 = array([0,6,3,2,1,6,6,7])
in_seq23 = array([0,3,3,1,7,3,6,5])
in_seq24 = array([4,4,2,4,3,2,8,5])


out_seq = array([in_seq1[i] + in_seq2[i] + in_seq3[i] + in_seq4[i] + in_seq5[i] + in_seq6[i] + in_seq7[i]
                 + in_seq8[i] + in_seq9[i] + in_seq10[i] + in_seq11[i] + in_seq12[i] + in_seq13[i] + in_seq14[i]
                 + in_seq15[i] + in_seq16[i] + in_seq17[i] + in_seq18[i] + in_seq19[i] + in_seq20[i] + in_seq21[i]
                 + in_seq22[i] + in_seq23[i] +in_seq24[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
in_seq3 = in_seq3.reshape((len(in_seq3), 1))
in_seq4 = in_seq4.reshape((len(in_seq4), 1))
in_seq5 = in_seq5.reshape((len(in_seq5), 1))
in_seq6 = in_seq6.reshape((len(in_seq6), 1))
in_seq7 = in_seq7.reshape((len(in_seq7), 1))
in_seq8 = in_seq8.reshape((len(in_seq8), 1))
in_seq9 = in_seq9.reshape((len(in_seq9), 1))
in_seq10 = in_seq10.reshape((len(in_seq10), 1))
in_seq11 = in_seq11.reshape((len(in_seq11), 1))
in_seq12 = in_seq12.reshape((len(in_seq12), 1))
in_seq13 = in_seq13.reshape((len(in_seq13), 1))
in_seq14 = in_seq14.reshape((len(in_seq14), 1))
in_seq15 = in_seq15.reshape((len(in_seq15), 1))
in_seq16 = in_seq16.reshape((len(in_seq16), 1))
in_seq17 = in_seq17.reshape((len(in_seq17), 1))
in_seq18 = in_seq18.reshape((len(in_seq18), 1))
in_seq19 = in_seq19.reshape((len(in_seq19), 1))
in_seq20 = in_seq20.reshape((len(in_seq20), 1))
in_seq21 = in_seq21.reshape((len(in_seq21), 1))
in_seq22 = in_seq22.reshape((len(in_seq22), 1))
in_seq23 = in_seq23.reshape((len(in_seq23), 1))
in_seq24 = in_seq24.reshape((len(in_seq24), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2,in_seq3,in_seq4,in_seq5,in_seq6,in_seq7,in_seq8,in_seq9,in_seq10,in_seq11,
                  in_seq12,in_seq13,in_seq14,in_seq15,in_seq16,in_seq17,in_seq18,in_seq19,in_seq20,in_seq21,in_seq22,in_seq23,in_seq24,
                  out_seq))
# choose a number of time steps
n_steps_in, n_steps_out = 3, 1
# covert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]
# define model
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
'''
这里输入3D为（样本量，return_sequences数量（=n_steps_out）,200）
输出为（样本量，return_sequences数量（=n_steps_out），n_features）
就是每个输出是（3,2)维度的
'''
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=300, verbose=0)
# demonstrate prediction
x_input = array([[5,4,6,8,9,11,16,11,7,14,12,10,12,16,12,7,5,5,8,6,8,6,3,2,203],
                 [1,6,13,9,4,7,10,10,12,8,12,10,5,9,11,8,9,4,8,5,6,6,6,8,187],
                 [6,5,12,9,11,7,14,20,17,18,8,10,8,10,10,11,9,7,7,9,6,7,5,5,231]])
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat[0][0])
