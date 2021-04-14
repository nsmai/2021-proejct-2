import tensorflow as tf
from tensorflow import keras

### Coefficient of Determination (결정계수) 
### 결정계수 위키(영문) https://en.wikipedia.org/wiki/Coefficient_of_determination
### 결정계수 위키(국문) https://ko.wikipedia.org/wiki/%EA%B2%B0%EC%A0%95%EA%B3%84%EC%88%98
### 결정계수는 0~1 사이로 정의되나, 
### 0미만의 값이 보인다면 해당 모델은 x_data의 값들과 y_data의 평균값의 차이보다 큰경우 음수가 나옴 
### (다시 말하면 그냥 y_data 평균값으로 예측해도 그것보다 못하다는 말로, 매우 안좋은 모델이라는 말임)
def coeff_determination(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    r2 =  1 - SS_res/(SS_tot + K.epsilon())
    return r2

x_data = [1,2,3,4,5]
y_data = [4,7,10,13,16]

model = keras.Sequential([keras.layers.Dense(1, input_shape=[1])])

optimizer = keras.optimizers.RMSprop(lr=0.2)

### 위에서 결정한 optimizer를 이용한 예측모델 방법 생성
model.compile(loss='mse', optimizer=optimizer, metrics=[coeff_determination])

### 모델 피팅 epochs만큼 수행
model.fit(x_data, y_data, epochs=100)

### 만들어진 모델에 x_data를 입력하여 예측된 y_data의 값을 y_pred_data에 저장
y_pred_data =  model.predict(x_data)

### x_data, y_data, 예측된 y_pred_data, 오차값 을 출력함.
for idx, xi in enumerate(x_data):
    yi = y_data[idx]
    y_pred_i = y_pred_data[idx][0]
    error_i = abs(yi-y_pred_i)
    
    print('x: {0}, y: {1}, y_pred: {2:.4f}, Mean_squared_error: {3:.4f}'.format(xi, yi, y_pred_i, error_i))
    pass
