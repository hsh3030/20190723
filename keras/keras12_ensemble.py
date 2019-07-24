# keras Ensemble

#1. 데이터 구성
import numpy as np 

x1 = np.array([range(100), range(311,411), range(100)]) 
y1 = np.array([range(501,601), range(711,811), range(100)]) 
x2 = np.array([range(100,200), range(311,411), range(100,200)]) 
y2 = np.array([range(501,601), range(711,811), range(100)]) 

# 행과 열 바꾸기 (transpose)
# ValueError: Error when checking input: expected dense_1_input to have shape (3,) but got array with shape (100,) => 열이 틀리다
x1 = np.transpose(x1)
y1 = np.transpose(y1)
x2 = np.transpose(x2)
y2 = np.transpose(y2)

print(x1.shape)
print(x2.shape)
print(y1.shape)
print(y2.shape)


from sklearn.model_selection import train_test_split  # 함수를 test_size=0.4로 train =0.6 , test = 0.4 로 나눈다
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, random_state=66, test_size=0.4
)
# 함수를 test_size=0.4로 train =0.6 , test = 0.2 val = 0.2 로 나눈다
x1_val, x1_test, y1_val, y1_test = train_test_split( 
    x1_test, y1_test, random_state=66, test_size=0.5
)

from sklearn.model_selection import train_test_split  # 함수를 test_size=0.4로 train =0.6 , test = 0.4 로 나눈다
x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, random_state=66, test_size=0.4
)
# 함수를 test_size=0.4로 train =0.6 , test = 0.2 val = 0.2 로 나눈다
x2_val, x2_test, y2_val, y2_test = train_test_split( 
    x2_test, y2_test, random_state=66, test_size=0.5
)

print(x2_test.shape)


#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input 
# model = Sequential() # Sequential = 순차적인 모델

# 함수형 model (input1,2 는 각각의 모델이다)
input1 = Input(shape=(3,))
dense1 = Dense(100, activation='relu')(input1)
dense1_2 = Dense(30)(dense1) # 추가 Dense를 만든다
dense1_3 = Dense(7)(dense1_2)
#두번째 모델
input2 = Input(shape=(3,))
dense2 = Dense(50, activation='relu')(input2)
dense2_2 = Dense(7)(dense2)

# 두 model 합치기 (concatenate)
from keras.layers.merge import concatenate 
merge1 = concatenate([dense1_3, dense2_2])
#3번째 모델 
middle1 = Dense(10)(merge1)
middle2 = Dense(5)(middle1)
middle3 = Dense(7)(middle2)


############################ output model #####################

output1 = Dense(30)(middle3)
output1_2 = Dense(7)(output1)
output1_3 = Dense(3)(output1_2)

output2 = Dense(20)(middle3)
output2_2 = Dense(70)(output2)
output2_3 = Dense(3)(output2_2)

model = Model(inputs =[input1, input2], outputs = [output1_3, output2_3])

model.summary()

#3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) #loss : 손실율 / optimizer : 적용함수 
model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) #loss : 손실율 / optimizer : 적용함수 

# model.fit(x, y, epochs = 100, batch_size=3) # epochs : 반복 횟수 / batch_size : 몇개씩 잘라서 할 것인가 / batch_size defalt = 32
model.fit([x1_train, x2_train],[y1_train, y2_train], epochs = 10, batch_size=1, validation_data=([x1_val, x2_val], [y1_val, y2_val])) # model.fit : 훈련 / validation_data를 추가하면 훈련이 더 잘됨.

#4. 평가 예측
a = list(model.evaluate([x1_test, x2_test],[y1_test, y2_test], batch_size=1)) # evaluate : 평가 [x,y 값으로]
#loss, acc = model.evaluate(x2_test, y2_test, batch_size=1)
print(a)

#print("acc : ", acc) # acc = 분류 모델에 적용
'''
y_predict = model.predict([x1_test, y1_test]) # predict : 예측치 확인
print(y_predict)
'''
'''
# RMSE 구하기 (RMSE: 낮을수록 좋다.)
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict): # y_test 와 y_predict 비교하기 위한 함수 (원래의 값과 예측값을 비교)
    return np.sqrt(mean_squared_error(y_test, y_predict)) # 비교하여 그 차이를 빼준다
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기 (1에 가까울 수록 좋다.)
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)
'''