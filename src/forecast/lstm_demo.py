import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# 生成正弦波数据集
def generate_sine_wave(seq_length, num_samples):
    x = np.linspace(0, num_samples*2*np.pi, num_samples)
    y = np.sin(x)
    data = []
    for i in range(len(y) - seq_length):
        data.append(y[i:i+seq_length])
    data = np.array(data)
    data = data.reshape((data.shape[0], data.shape[1], 1))
    return data, y[seq_length:]

# 参数设置
SEQ_LENGTH = 50
NUM_SAMPLES = 1000

# 生成数据
data, target = generate_sine_wave(SEQ_LENGTH, NUM_SAMPLES)

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)
target = scaler.fit_transform(target.reshape(-1, 1)).flatten()

# 划分训练集和测试集
train_size = int(len(data) * 0.8)
test_size = len(data) - train_size
train_data, test_data = data[0:train_size, :], data[train_size:len(data), :]
train_target, test_target = target[0:train_size], target[train_size:len(target)]

# 构建LSTM模型
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(train_data, train_target, epochs=20, batch_size=32, validation_split=0.1, verbose=1)

# 预测
train_predict = model.predict(train_data)
test_predict = model.predict(test_data)

# 反归一化
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
target = scaler.inverse_transform(target.reshape(-1, 1))

# 绘图
plt.figure(figsize=(12, 6))
plt.plot(target, label='True')
train_plot = np.empty_like(target)
train_plot[:, :] = np.nan
train_plot[SEQ_LENGTH:len(train_predict)+SEQ_LENGTH] = train_predict
plt.plot(train_plot, label='Train Predict')

test_plot = np.empty_like(target)
test_plot[:, :] = np.nan
test_plot[len(train_predict)+(SEQ_LENGTH*2)+1:len(target)-1] = test_predict
plt.plot(test_plot, label='Test Predict')

plt.legend()
plt.show()