import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import *
import matplotlib.ticker as mticker
import matplotlib
from matplotlib import cm
import sklearn
from xgboost import XGBClassifier
import shap
import pylab

plt.rcParams["font.sans-serif"] = ["SimHei"]   # 解决中文乱码问题
plt.rcParams["axes.unicode_minus"] = False    # 该语句解决图像中的“-”负号的乱码问题

df_ebs = pd.read_excel(r'D:\WORK\生态驾驶\项目\LMEBEG1R8HE000054_20210428171832179_4239-52\处理后数据\带标签\下行完整_异常值处理后.xlsx')
df_bus = pd.read_csv(r'D:\WORK\生态驾驶\项目\燃油车数据\2013年1月100路公交实验数据\1.5\1.5原始数据.csv', encoding='gbk')


# 对齐y1和y2的0刻度的函数
def align_yaxis(ax1, ax2):
    y_lims = np.array([ax.get_ylim() for ax in [ax1, ax2]])
    # force 0 to appear on both axes, comment if don't need
    y_lims[:, 0] = y_lims[:, 0].clip(None, 0)
    y_lims[:, 1] = y_lims[:, 1].clip(0, None)
    # normalize both axes
    y_mags = (y_lims[:, 1] - y_lims[:, 0]).reshape(len(y_lims), 1)
    y_lims_normalized = y_lims / y_mags
    # find combined range
    y_new_lims_normalized = np.array([np.min(y_lims_normalized), np.max(y_lims_normalized)])
    # denormalize combined range to get new axes
    new_lim1, new_lim2 = y_new_lims_normalized * y_mags
    ax1.set_ylim(new_lim1)
    ax2.set_ylim(new_lim2)


# 峰值识别
def thresholding_algo(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])      # 初始化
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter[i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1       # -1改成了0

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

    return dict(signals=np.asarray(signals),
                avgFilter=np.asarray(avgFilter),
                stdFilter=np.asarray(stdFilter))


# 电车
energy_consum = df_ebs.loc[:, '瞬时能耗/kwh']
print(energy_consum.describe())
# 参数寻优，找到最优lag, threshold, influence
# Obj_Func = []
# for i in range(1, 5):
#     for j in range(1, 9):
#         for k in range(1, 6):
#             lag = 10*i
#             threshold = j
#             influence = 0.1*k
#             result = thresholding_algo(energy_consum, lag, threshold, influence)
#             signals = pd.DataFrame(result['signals'], columns=['signals'])
#             df_ = pd.concat([df_ebs, signals], axis=1)
#             spike_points = len(df_[df_['signals'] == 1]['signals'])
#             sum_spikeenergy = df_[df_['signals'] == 1]['瞬时能耗/kwh'].sum()
#             obj_func = 1 - (spike_points / len(df_['signals'])) + (sum_spikeenergy / df_['瞬时能耗/kwh'].sum())
#             Obj_Func.append(obj_func)
# outcome = pd.DataFrame(Obj_Func).values.reshape((4, 8, 5))
# max_outcome = np.max(outcome)
# max_outcome_index = np.where(outcome == np.max(outcome))
# print(max_outcome, max_outcome_index)

# 输入最优参数
# lag1 = (max_outcome_index[0][0]+1)*10
# threshold1 = max_outcome_index[1][0]+1
# influence1 = max_outcome_index[2][0]*0.1
lag1 = 30
threshold1 = 1
influence1 = 0.1
print(lag1, threshold1, influence1)
y1 = np.array(energy_consum)
result1 = thresholding_algo(energy_consum, lag1, threshold1, influence1)
signals1 = pd.DataFrame(result1['signals'], columns=['signals'])

signals1 = signals1.replace([-1], [0])    # 将signals=-1替换为0
df_ebs_ = pd.concat([df_ebs, signals1], axis=1)
spike_points1 = len(df_ebs_[df_ebs_['signals'] == 1]['signals'])
sum_spikeenergy1 = df_ebs_[df_ebs_['signals'] == 1]['瞬时能耗/kwh'].sum()
obj_func1 = 1 - (spike_points1 / len(df_ebs_['signals'])) + (sum_spikeenergy1 / df_ebs_['瞬时能耗/kwh'].sum())
print('obj_func:', obj_func1, '\n')

# Plot result
fig1, ax1 = plt.subplots(figsize=(14, 6))
ax1.tick_params(labelsize=15)
ax1.plot(np.arange(1, len(y1)+1), y1, label="Energy consumption")
plt.xlabel('Time(s)', size=15)
plt.ylabel('Energy consumption(kwh)', size=15)
plt.ylim(-0.015, 0.025)
plt.rcParams.update({'font.size': 15})     # 设置图例字体大小
ax1.legend(loc='upper left')
# 第二纵轴的设置和绘图
ax2 = ax1.twinx()
plt.step(np.arange(1, len(y1)+1), result1["signals"], color="red", lw=2, label="Spike signal")
ax2.tick_params(labelsize=15)
ax2.set_ylabel("Spike signal", size=15)
plt.rcParams.update({'font.size': 15})     # 设置图例字体大小
plt.legend(loc='upper right')
plt.ylim(-1.25, 1.25)
plt.xlim(19400, 19600)
# 对齐y轴0刻度线
align_yaxis(ax1, ax2)
plt.show()
plt.close()

# 分析高能耗和一般能耗的特征
df_ebs_spike = df_ebs_[df_ebs_['signals'] == 1]
df_ebs_smooth = df_ebs_[df_ebs_['signals'] == 0]
print(df_ebs_spike.shape, df_ebs_smooth.shape)
# 绘制电车不同能耗等级下的加速度箱型图
df_ebs_spike_acc = df_ebs_spike['车辆加速度/m/s2']
df_ebs_smooth_acc = df_ebs_smooth['车辆加速度/m/s2']
plt.figure(figsize=(6, 5), dpi=100)
plt.boxplot([df_ebs_spike_acc, df_ebs_smooth_acc])
plt.xticks([1, 2], ['Spike points', 'Non-spike points'])
plt.ylabel("Acceleration(m/${s^2}$)", fontdict={'size': 15})
plt.grid(linestyle="--", alpha=0.3)
plt.ylim(-4.5, 3.5)
plt.show()
plt.close()
# 绘制电车不同能耗等级下的速度箱型图
df_ebs_spike_v = df_ebs_spike['车辆速度/km/h']
df_ebs_smooth_v = df_ebs_smooth['车辆速度/km/h']
plt.figure(figsize=(6, 5), dpi=100)
plt.boxplot([df_ebs_spike_v, df_ebs_smooth_v])
plt.xticks([1, 2], ['Spike points', 'Non-spike points'])
# plt.title("电车不同能耗等级下的速度箱型图", fontdict={'size': 14})
plt.ylabel("Speed(km/h)", fontdict={'size': 15})
plt.grid(linestyle="--", alpha=0.3)
plt.show()
plt.close()

# SHAP图分析
# XGBoost分类
# 归一化
dataXGBoost_ebs = df_ebs_[['车辆加速度/m/s2', '车辆速度/m/s', '瞬时能耗/kwh', 'signals']]
scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
scaler.fit(dataXGBoost_ebs)    # 找出每列最大、小值，并存储
datascaled_ebs = scaler.transform(dataXGBoost_ebs)
datascaled_ebs = pd.DataFrame(datascaled_ebs)
datascaled_ebs.columns = ('车辆加速度/m/s2', '车辆速度/m/s', '瞬时能耗/kwh', 'signals')
# 将数据分割为训练集和测试集
ebs_values = datascaled_ebs.values
np.random.shuffle(ebs_values)
n_train = int(len(ebs_values[:, 0]) * 0.8)       # 训练集数据个数
train = ebs_values[:n_train, :]
test = ebs_values[n_train:, :]
train_data, test_data, train_label, test_label = train[:, 0:2], test[:, 0:2], train[:, 3], test[:, 3]
# 建模
Classifier_XGBoost_ebs = XGBClassifier(learning_rate=0.1)
eval_set = [(test_data, test_label)]
Classifier_XGBoost_ebs.fit(train_data, train_label, early_stopping_rounds=10, eval_metric='logloss', eval_set=eval_set, verbose=True)
# XGBoost准确率
print("XGBoost训练集：", Classifier_XGBoost_ebs.score(train_data, train_label))
print("XGBoost测试集：", Classifier_XGBoost_ebs.score(test_data, test_label))
explainer = shap.TreeExplainer(Classifier_XGBoost_ebs)
shap_values = explainer.shap_values(train_data)  # 传入特征矩阵X，计算SHAP值
# summarize the effects of all the features
shap.summary_plot(shap_values, train_data, plot_size=(10, 4))


# 油车
diesel_consum = df_bus.loc[:, 'Fuel Rate(gal/s)']
print(diesel_consum.describe())
# 参数寻优，找到最优lag, threshold, influence
# Obj_Func = []
# for i in range(1, 5):
#     for j in range(1, 9):
#         for k in range(1, 6):
#             lag = 10*i
#             threshold = j
#             influence = 0.1*k
#             result = thresholding_algo(diesel_consum, lag, threshold, influence)
#             signals = pd.DataFrame(result['signals'], columns=['signals'])
#             df_ = pd.concat([df_bus, signals], axis=1)
#             spike_points = len(df_[df_['signals'] == 1]['signals'])
#             sum_spikeenergy = df_[df_['signals'] == 1]['Fuel Rate(gal/s)'].sum()
#             obj_func = 1 - (spike_points / len(df_['signals'])) + (sum_spikeenergy / df_['Fuel Rate(gal/s)'].sum())
#             Obj_Func.append(obj_func)
# outcome = pd.DataFrame(Obj_Func).values.reshape((4, 8, 5))
# max_outcome = np.max(outcome)
# max_outcome_index = np.where(outcome == np.max(outcome))
# print(max_outcome, max_outcome_index)

# 输入最优参数
# lag2 = (max_outcome_index[0][0]+1)*10
# threshold2 = max_outcome_index[1][0]+1
# influence2 = max_outcome_index[2][0]*0.1
lag2 = 30
threshold2 = 1
influence2 = 0.2
print(lag2, threshold2, influence2)
y2 = np.array(diesel_consum)
result2 = thresholding_algo(diesel_consum, lag2, threshold2, influence2)
signals2 = pd.DataFrame(result2['signals'], columns=['signals'])

signals2 = signals2.replace([-1], [0])    # 将signals=-1替换为0
df_bus_ = pd.concat([df_bus, signals2], axis=1)
spike_points2 = len(df_bus_[df_bus_['signals'] == 1]['signals'])
sum_spikeenergy2 = df_bus_[df_bus_['signals'] == 1]['Fuel Rate(gal/s)'].sum()
obj_func2 = 1 - (spike_points2 / len(df_bus_['signals'])) + (sum_spikeenergy2 / df_bus_['Fuel Rate(gal/s)'].sum())
print('obj_func:', obj_func2, '\n')

# Plot result
fig2, ax1 = plt.subplots(figsize=(14, 6))
ax1.tick_params(labelsize=15)
ax1.plot(np.arange(1, len(y2)+1), y2, label="Diesel consumption")
plt.xlabel('Time(s)', size=15)
plt.ylabel('Diesel consumption(gal)', size=15)
plt.ylim(-0.0015, 0.002)
plt.rcParams.update({'font.size': 15})     # 设置图例字体大小
ax1.legend(loc='upper left')
# 第二纵轴的设置和绘图
ax2 = ax1.twinx()
plt.step(np.arange(1, len(y2)+1), result2["signals"], color="red", lw=2, label="Spike signal")
ax2.tick_params(labelsize=15)
ax2.set_ylabel("Spike signal", size=15)
plt.rcParams.update({'font.size': 15})     # 设置图例字体大小
plt.legend(loc='upper right')
plt.ylim(-1.25, 1.25)
plt.xlim(4800, 5000)
# 对齐y轴0刻度线
align_yaxis(ax1, ax2)
plt.show()
plt.close()

# 分析高能耗和一般能耗的特征
df_bus_spike = df_bus_[df_bus_['signals'] == 1]
df_bus_smooth = df_bus_[df_bus_['signals'] != 1]
print(df_bus_spike.shape, df_bus_smooth.shape)
# 绘制电车不同能耗等级下的加速度箱型图
df_bus_spike_acc = df_bus_spike['acc(m/s2)']
df_bus_smooth_acc = df_bus_smooth['acc(m/s2)']
plt.figure(figsize=(6, 5), dpi=100)
plt.boxplot([df_bus_spike_acc, df_bus_smooth_acc])
plt.xticks([1, 2], ['Spike points', 'Non-spike points'])
plt.ylabel("Acceleration(m/${s^2}$)", fontdict={'size': 15})
plt.grid(linestyle="--", alpha=0.3)
plt.ylim(-4.5, 3.5)
plt.show()
plt.close()
# 绘制电车不同能耗等级下的速度箱型图
df_bus_spike_v = df_bus_spike['speed/km/h']
df_bus_smooth_v = df_bus_smooth['speed/km/h']
plt.figure(figsize=(6, 5), dpi=100)
plt.boxplot([df_bus_spike_v, df_bus_smooth_v])
plt.xticks([1, 2], ['Spike points', 'Non-spike points'])
plt.ylabel("Speed(km/h)", fontdict={'size': 15})
plt.grid(linestyle="--", alpha=0.3)
plt.show()
plt.close()

# SHAP图分析
# XGBoost分类
# 归一化
dataXGBoost_bus = df_bus_[['acc(m/s2)', 'speed/km/h', 'Fuel Rate(gal/s)', 'signals']]
scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
scaler.fit(dataXGBoost_bus)    # 找出每列最大、小值，并存储
datascaled_bus = scaler.transform(dataXGBoost_bus)
datascaled_bus = pd.DataFrame(datascaled_bus)
datascaled_bus.columns = ('acc(m/s2)', 'speed/km/h', 'Fuel Rate(gal/s)', 'signals')
# 将数据分割为训练集和测试集
bus_values = datascaled_bus.values
np.random.shuffle(bus_values)
n_train = int(len(bus_values[:, 0]) * 0.8)       # 训练集数据个数
train = bus_values[:n_train, :]
test = bus_values[n_train:, :]
train_data, test_data, train_label, test_label = train[:, 0:2], test[:, 0:2], train[:, 3], test[:, 3]
# 建模
Classifier_XGBoost_bus = XGBClassifier(learning_rate=0.1)
eval_set = [(test_data, test_label)]
Classifier_XGBoost_bus.fit(train_data, train_label, early_stopping_rounds=10, eval_metric='logloss', eval_set=eval_set, verbose=True)
# XGBoost准确率
print("XGBoost训练集：", Classifier_XGBoost_bus.score(train_data, train_label))
print("XGBoost测试集：", Classifier_XGBoost_bus.score(test_data, test_label))
explainer = shap.TreeExplainer(Classifier_XGBoost_bus)
shap_values = explainer.shap_values(train_data)  # 传入特征矩阵X，计算SHAP值
# summarize the effects of all the features
shap.summary_plot(shap_values, train_data, plot_size=(10, 4))

