import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

data = pd.read_csv("sphist.csv")

data['Date'] = pd.to_datetime(data['Date'])

data = data.sort_values(by='Date', ascending=True)

data['day_5'] = data['Close'].rolling(5).mean()
data['day_5'] = data['day_5'].shift()
data['day_30'] = data['Close'].rolling(30).mean()
data['day_30'] = data['day_30'].shift()
data['day_ratio'] = data['day_5'] / data['day_30']
data['vol_5'] = data['Volume'].rolling(5).mean()
data['vol_5'] = data['vol_5'].shift()
data['vol_30'] = data['Volume'].rolling(30).mean()
data['vol_30'] = data['vol_30'].shift()
data['vol_ratio'] = data['vol_5'] / data['vol_30']
data['day_diff'] = data['Close'] - data['Open']
data['day_diff'] = data['day_diff'].shift()
data['high_low'] = data['High'] - data['Low']
data['high_low'] = data['high_low'].shift()

data = data.dropna(axis=0)

train_filter = data['Date'] < datetime(year=2013, month=1, day=1)
test_filter = data['Date'] >= datetime(year=2013, month=1, day=1)

train = data[train_filter]
test = data[test_filter]

model = LinearRegression()
features = ['day_5', 'day_30', 'vol_5', 'vol_30', 'vol_ratio', 'day_diff', 'high_low']
target = 'Close'
model.fit(train[features], train[target])
predictions = model.predict(test[features])
MAE = mean_absolute_error(test[target], predictions)
MSE = mean_squared_error(test[target], predictions)
RMSE = MSE ** (1/2)

print("Mean Absolute Error: " + str(MAE))
print("Mean Squared Error: " + str(MSE))
print("Root Mean Squared Error: " + str(RMSE))


    