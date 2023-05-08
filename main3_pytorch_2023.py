'''
参考URL: https://zenn.dev/dumbled0re/articles/744c783a8b3992
背景:
- テニスのインプレー判定を行うに当たり時系列を考慮したかった
- まずは、同じような株価分析を時系列で行うべく上記URLより実施
'''

import os
import datetime
import math
import numpy as np
import pandas as pd
import pandas.tseries.offsets as offsets
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.simplefilter('ignore')
plt.style.use("ggplot")

csv_data_path = "data/bitcoin_data.csv"

df = pd.read_csv(csv_data_path, engine="python", encoding="utf-8")
df = df.drop(["始値", "高値", "安値", "出来高", "変化率 %"], axis=1)
df = df.rename(columns={"終値": "Close", "日付け": "Date"})
df["Date"] = pd.to_datetime(df["Date"], format="%Y年%m月%d日")
df["Close"] = df["Close"].str.replace(',', '').astype(float)
df.set_index("Date", inplace=True)
df = df.sort_index()

plt.figure(figsize=(16, 8))
plt.title("Close Price History")
plt.plot(df["Close"])
plt.xlabel("Date", fontsize=18)
plt.ylabel("Close Price USD ($)", fontsize=18)
plt.show()

train = df.loc[:"2020-12-31"].values
test = df.loc["2021-01-01"::].values
scaler = MinMaxScaler(feature_range=(0,1))
scaler = scaler.fit(train)
scaled_train_data = scaler.transform(train)
scaled_test_data = scaler.transform(test)

from typing import Tuple

def make_sequence_data(data: np.ndarray, sequence_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """データをsequence_sizeに指定したサイズのシーケンスに分けてシーケンスとその答えをarrayで返す
    Args:
        data (np.ndarray): 入力データ
        sequence_size (int): シーケンスサイズ
    Returns:
        seq_arr: sequence_sizeに指定した数のシーケンスを格納するarray
        target_arr: シーケンスに対応する答えを格納するarray
    """

    num_data = len(data)
    seq_data = []
    target_data = []
    for i in range(num_data - sequence_size):
        seq_data.append(data[i:i+sequence_size])
        target_data.append(data[i+sequence_size:i+sequence_size+1])
    seq_arr = np.array(seq_data)
    target_arr = np.array(target_data)

    return seq_arr, target_arr

seq_length = 30
train_X, train_Y = make_sequence_data(scaled_train_data, seq_length)
test_X, test_Y = make_sequence_data(scaled_test_data, seq_length)
# テンソル変換してLSTMに入力するために軸を変更(シーケンス、バッチサイズ、入力次元)
tensor_train_X = torch.FloatTensor(train_X).permute(1, 0, 2)
tensor_train_Y = torch.FloatTensor(train_Y).permute(1, 0, 2)
tensor_test_X = torch.FloatTensor(test_X).permute(1, 0, 2)
# rmseを計算する時の形を合わせる
test_Y = test_Y.reshape(len(test_Y), 1)

class LSTM(nn.Module):
    def __init__(self, hidden_size=100):
        super().__init__()
        self.hidden_size = hidden_size
        # input_sizeは入力する次元数
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_size) # batch_first=True
        self.linear = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x_last = x[-1]
        x = self.linear(x_last)

        return x

model = LSTM(100)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
losses = []
with tqdm(total=epochs, desc='Training', position=0) as pbar:
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(tensor_train_X)
        loss = criterion(output, tensor_train_Y)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        # if epoch % 10 == 0:
        #     pbar.write(f"epoch: {epoch}, loss: {loss.item()}")
        pbar.update()

plt.figure(figsize=(12, 8))
plt.title("model loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(losses)
plt.legend(["Train"])
plt.show()

predictions = model(tensor_test_X).detach().numpy()
rmse = mean_squared_error(test_Y, predictions, squared=False)
print(rmse)

train = df.loc["2018-12-31":"2020-12-31"]
valid = df.loc["2021-01-31"::]
valid["Predictions"] = scaler.inverse_transform(predictions)
# 可視化
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

# print(df.head())