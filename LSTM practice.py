import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from tensorflow. keras.layers import LSTM

#一些基础的设置
feanum=1#一共有多少特征
window=5#时间窗设置
df1=pd.read_csv('C:/Users/hanxu/Downloads/813+(1).csv') #读取数据
df1=df1.iloc[:,4:]#删除前四列没用的
df1.tail()

#归一化
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
df0=min_max_scaler.fit_transform(df1)
df = pd.DataFrame(df0, columns=df1.columns)
df.tail()

#这一部分在处理数据 将原始数据改造为LSTM网络的输入
stock=df
seq_len=window
amount_of_features = len(stock.columns)#有几列
pd.DataFrame(stock)
data = stock.values#表格转化为矩阵
sequence_length = seq_len + 1#序列长度+1
result = []
for index in range(len(data) - sequence_length):#循环 数据长度-时间窗长度 次
    result.append(data[index: index + sequence_length])#第i行到i+5
result = np.array(result)#得到样本，样本形式为 window*feanum
cut=150#分训练集测试集 最后cut个样本为测试集
train = result[:-cut, :]
x_train = train[:, :-1]
y_train = train[:, -1][:,-1]
x_test = result[-cut:, :-1]
y_test = result[-cut:, -1][:,-1]
X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))

#展示下训练集测试集的形状 看有没有问题
#print("X_train", X_train.shape)
#print("y_train", y_train.shape)
#print("X_test", X_test.shape)
#print("y_test", y_test.shape)

#更改数组shape
#X_train=X_train.reshape(len(X_train),window)
#y_train=y_train.reshape(len(X_train))
#X_test=X_test.reshape(cut,window)
#y_test=y_test.reshape(cut)
#print("X_train", X_train.shape)
#print("y_train", y_train.shape)
#print("X_test", X_test.shape)
#print("y_test", y_test.shape)

#建立、训练模型过程
d = 0.01
model = Sequential()#建立层次模型
model.add(LSTM(32, input_shape=(window, feanum), return_sequences=True))#建立LSTM层
model.add(Dropout(d))#建立的遗忘层
model.add(LSTM(16, input_shape=(window, feanum), return_sequences=False))#建立LSTM层
model.add(Dropout(d))#建立的遗忘层
model.add(Dense(2,kernel_initializer='uniform',activation='relu'))   #建立全连接层
model.add(Dense(1,kernel_initializer='uniform',activation='relu'))
model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
model.fit(X_train, y_train, epochs =100, batch_size = 128) #训练模型epoch次

