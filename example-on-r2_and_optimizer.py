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

### 아래 optimizer 방법을 구현하고 주석처리 하였음.
### 원하는 optimizer를 활성화 하여 학습을 진행하면 좋을 것임.
### 주석 제거는 앞의 #을 제거하면 됩니다.

### 확률적 경사 하강법(Stochastic Gradient Descent) 
#optimizer = 'sgd' ### 또는 아래 표현도 같음
optimizer = keras.optimizers.SGD()

### 확률적 경사 하강법(Stochastic Gradient Descent) + Learning rate (학습률) 적용
#optimizer = keras.optimizers.SGD(lr=0.1)

### 확률적 경사 하강법(Stochastic Gradient Descent) + Learning rate (학습률) + Momentum(모멘텀) 적용
### learning rate과 momemtum의 합이 1이 되는 경우의 근처에서 좋은 최적화 찾아지는 경험적 규칙이 있습니다. 
### 정답은 아니나, learning rate와 momentum을 변화 시킬때 경우의 수를 줄일때 참고하세요. ex) [ lr=0.1, momentum =0.9], [ lr=0.2, momentum = 0.8], ... [ lr=0.9, momentum =0.1] 으로 grid search
### 합이 1이라는 경험적 규칙은 정답은 아니므로, 다양한 learning rate와 momentum을 넣어 학습 결과 최적화 방법에 대한 경험을 쌓아가면 좋습니다.
#optimizer = keras.optimizers.SGD(lr=0.1, momentum=0.9)

### Nesterov Accelrated Gradient(NAG, 네스테로프 모멘텀)
#optimizer = keras.optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)

### Adagrad(Adaptive Gradient, 아다그라드)
#optimizer = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-6)

### RMSprop (learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False, name='RMSprop', **kwargs)
### 자율탐구예제의 0.2는 learning_rate = 0.2를 입력한 것입니다.
#optimizer = keras.optimizers.RMSprop(lr=0.2)
#optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

### Adam(Adaptive Moment Estimation, 아담)
### 함수형태: Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam', **kwargs)
#optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

### AdaDelta(Adaptive Delta, 아다델타)
### 함수형태: AdaDelta(learning_rate=0.001, rho=0.95, epsilon=1e-07, name='Adadelta', **kwargs)
#optimizer = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)


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
