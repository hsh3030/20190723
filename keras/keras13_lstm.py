from numpy import array # as np 대신 바로 array 가져와 쓴다
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터 만들기
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = array([4,5,6,7])

print("x.shape : ", x.shape)
print("y.shape : ", y.shape)

# reshape 작업
x = x.reshape((x.shape[0], x.shape[1],1))
print("x.shape : ", x.shape)

# 2. Model 구성
model = sequential()
model.add(LSTM(50, activation='relu', input_shape=(3,1)))
model.add(Dense(1))
