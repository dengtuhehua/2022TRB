import pandas as pd
from pandas import concat
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import os
import seaborn as sns
import time
import numpy as np


os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
plt.rcParams["font.sans-serif"] = ["SimHei"]   # 解决中文乱码问题
plt.rcParams["axes.unicode_minus"] = False    # 该语句解决图像中的“-”负号的乱码问题

df = pd.read_excel(r'D:\WORK\生态驾驶\项目\LMEBEG1R8HE000054_20210428171832179_4239-52\处理后数据\带标签\下行完整_异常值处理后.xlsx')

data = df.loc[:, ['瞬时能耗/kwh', '车辆加速度/m/s2', '车辆速度/m/s', '油门踏板状态', '制动踏板状态']]

# 相关性分析
data_corr = df.loc[:, ['瞬时能耗/kwh', '加速度变化率', '车辆加速度/m/s2', '车辆速度/m/s', '油门踏板状态', '制动踏板状态']]
data_corr.columns = ['Energy consumption', 'Jerk', 'Acceleration', 'Speed', 'Accelerator pedal status', 'Brake pedal status']
correlation = data_corr.corr(method='spearman')
plt.figure()
# 绘制热力图
sns.heatmap(correlation, linewidths=0.2, vmax=1, vmin=-1, linecolor='w', annot=True, annot_kws={'size': 10}, square=True)
plt.show()
plt.close()


# 将无监督转化为有监督
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]              # 变量个数
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(axis=0, inplace=True)
    return agg


values = data.values
# 确保所有变量都是实数型
values = values.astype('float32')
# 将数据进行归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# 将时间序列数据转换成有监督学习数据
reframed = series_to_supervised(scaled, 3, 1)
reframed.drop(columns=['var1(t-3)', 'var1(t-2)', 'var1(t-1)'], inplace=True)
# 将数据分割为训练集和测试集
values = reframed.values
n_train_num = int(len(values[:, 0]) * 0.8)       # 训练集数据个数
train = values[:n_train_num, :]
test = values[n_train_num:, :]
# 分离出特征集与标签
train_X, train_y = train[:, :-5], train[:, -5]
test_X, test_y = test[:, :-5], test[:, -5]
# 转换成3维数组 [样本数, 时间步长, 特征数]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, type(test_y))

# 创建模型
model_y = Sequential()
model_y.add(LSTM(128, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2]), kernel_initializer='he_uniform'))
model_y.add(Dropout(0.25))
model_y.add(Dense(1, activation='tanh'))
model_y.compile(loss='mae', optimizer='adam')
model_y.summary()
# 训练模型
history_y = model_y.fit(train_X, train_y, epochs=200, batch_size=64, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# 对损失进行可视化
plt.plot(history_y.history['loss'], label='Train')
plt.plot(history_y.history['val_loss'], label='Test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Train Loss')
plt.legend(loc='upper right')
plt.show()
# 模型训练结果展示
time_begin = time.time()
prediction_y = model_y.predict(test_X)
prediction_y = prediction_y * (df.loc[:, "瞬时能耗/kwh"].max() - df.loc[:, "瞬时能耗/kwh"].min()) + df.loc[:, "瞬时能耗/kwh"].min()
time_end = time.time()
print('预测花费的时间为：', time_end-time_begin)
test_y = test_y * (df.loc[:, "瞬时能耗/kwh"].max() - df.loc[:, "瞬时能耗/kwh"].min()) + df.loc[:, "瞬时能耗/kwh"].min()
plt.figure(figsize=(10, 5), dpi=100)
x = range(0, len(prediction_y))
plt.plot(x, test_y, label="Actual Consumption")
plt.plot(x, prediction_y, label="Predictive Consumption")
plt.xlabel("Time(s)", fontdict={'size': 15})
plt.ylabel("Energy Consumption(kwh)", fontdict={'size': 15})
plt.legend(loc='upper right')
plt.rcParams.update({'font.size': 12})     # 设置图例字体大小
# 显示网格
plt.grid(True)
# 限制横轴显示刻度的范围
plt.xlim(0, 1001)
plt.show()
plt.close()

# 预测误差
MSE = metrics.mean_squared_error(test_y, prediction_y)
RMSE = metrics.mean_squared_error(test_y, prediction_y)**0.5
MAE = metrics.mean_absolute_error(test_y, prediction_y)
# MAPE = metrics.mean_absolute_percentage_error(test_y, prediction_y)    # 因为真实值有0，所以存在0除问题，该公式不可用
print(MSE, RMSE, MAE)
