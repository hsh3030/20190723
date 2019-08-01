import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

dataend = read_csv('D:\kospi200test.csv', usecols=[4])
print(dataend.shape)

data_trainX = dataend[0 : 500]
data_testX = dataend[500:600]

data_trainX = data_trainX.values.reshape(data_trainX.shape[0], data_trainX.shape[1])
print(data_trainX.shape)

