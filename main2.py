'''
参考URL: https://supersoftware.jp/tech/20220907/17599/
背景:
- テニスのインプレー判定を行うに当たり時系列を考慮したかった
- まずは、同じような株価分析を時系列で行うべく上記URLより実施
'''

import numpy as np
from sklearn.metrics import r2_score
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

def get_stock_data(s_target, start_date, end_date):
    df = pdr.get_data_yahoo(s_target, start=start_date, end=end_date)
    return df

def plot_stock_data(df, s_target):
    plt.figure(figsize=(16,6))
    plt.title(s_target + ' Close Price History')
    plt.plot(df['Close'])
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Close Price USD ($)', fontsize=14)
    plt.show()

def add_moving_averages(df, ma_days):
    for ma in ma_days:
        column_name = f"MA for {ma} days"
        df[column_name] = df['Adj Close'].rolling(ma).mean()
    return df

def get_scaled_data(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def split_train_test_data(scaled_data, training_data_len, window_size):
    train_data = scaled_data[0:int(training_data_len), :]
    x_train, y_train = [], []
    for i in range(window_size, len(train_data)):
        x_train.append(train_data[i-window_size:i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    test_data = scaled_data[training_data_len - window_size: , :]
    x_test = []
    y_test = []
    for i in range(window_size, len(test_data)):
        x_test.append(test_data[i-window_size:i, 0])
        y_test.append(test_data[i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
    y_test = np.array(y_test)
    return x_train, y_train, x_test, y_test

def build_lstm_model(x_train):
    model = Sequential()
    model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(model, x_train, y_train, batch_size, epochs):
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    return history

def predict_stock_price(model, x_test, scaler, y_test):
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    r2s = r2_score(y_test, predictions)
    return predictions, rmse, r2s

def plot_predictions(data, training_data_len, predictions):
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid.loc[:, 'Predictions'] = predictions
    plt.figure(figsize=(16,6))
    plt.title('Model')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Close Price USD ($)', fontsize=14)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()

def main():
    s_target = 'GOOG'
    start_date = '2014-01-01'
    end_date = datetime.now()
    df = get_stock_data(s_target, start_date, end_date)
    plot_stock_data(df, s_target)
    ma_days = [10, 20, 50]
    df = add_moving_averages(df, ma_days)
    data = df.filter(['Close']).values
    scaled_data, scaler = get_scaled_data(data)
    training_data_len = int(np.ceil(len(data) * 0.8))
    window_size = 60
    x_train, y_train, x_test, y_test = split_train_test_data(scaled_data, training_data_len, window_size)
    model = build_lstm_model(x_train)
    batch_size = 32
    epochs = 100
    history = train_lstm_model(model, x_train, y_train, batch_size, epochs)
    predictions, rmse, r2s = predict_stock_price(model, x_test, scaler, y_test)
    print('RMSE:', rmse)
    print('R2 Score:', r2s)
    plot_predictions(data, training_data_len, predictions)

if __name__ == '__main__':
    main()
