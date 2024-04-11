from datetime import datetime
import calendar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from sklearn.metrics import *

df = pd.read_csv('CleanData_PM10.csv')
print(df.info())
df.head(2)

# conversione del dato numerico
#df['datetime'] = pd.to_datetime(df['datetime']) # automatically recognize format
df['datetime'] = pd.to_datetime(df['datetime'], format = '%Y-%m-%d %H:%M:%S')

print(df.info())
df.head(2)

plt.figure(figsize = (20, 4))
plt.plot(df['datetime'], df['PM10'])
plt.show()

# standard pandas operations apply

start = datetime.strptime('2020-03-25 00:00:00','%Y-%m-%d %H:%M:%S')
end = datetime.strptime('2020-04-2 00:00:00','%Y-%m-%d %H:%M:%S')

df2 = df[(df['datetime']>start)&(df['datetime']<end)]
plt.figure(figsize = (20, 4))
plt.plot(df2['datetime'], df2['PM10'])
plt.show()

# features based on time

df['month'] = df['datetime'].dt.month
df['weekday'] = df['datetime'].apply(lambda t: calendar.day_name[t.weekday()])
df['weekend01'] = df['weekday'].apply(lambda w: 1  if (w == 'Saturday' or w == 'Sunday') else 0)
df.head(2)

# features based on lag
df['PM10-1'] = df['PM10'].shift(1)
df['PM10-2'] = df['PM10'].shift(2)
df['PM10-3'] = df['PM10'].shift(3)
df.head(5)

# fix NaN based on initial shifts
# either remove initial columns
df = df.iloc[3:,:]
# or bfill
#df = df.bfill()
df.head()

# A step beyond adding raw lagged values is to add a summary of the values at previous time
# steps. We can calculate summary statistics across the values in the sliding window and include
# these as features in our dataset.


# function over a rolling window (e.g., rolling mean)
df['roll3mean'] = df['PM10'].rolling(3).mean()
df.head()

# function over all the values up to the current one (e.g., minimum value so far)
df['min_so_far'] = df['PM10'].expanding().min()
df.head(10)

# difference between current value i-th and a previous value (e.g., diff(2) --> v[i] - v[i-2])
df['diff2'] = df['PM10'].diff(2)
df.head(10)

#Forecasting as a Regression Problem
print('min time', df['datetime'].min())
print('max time', df['datetime'].max())
print(df.shape)

# split between train and test
tt_split = datetime.strptime('2020-05-02 00:00:00','%Y-%m-%d %H:%M:%S')
train = df[df['datetime'] <= tt_split]
test = df[df['datetime'] > tt_split]
train.head(4)

#Linear models on lags, a.k.a. autoregression (AR model)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(train.loc[:,['PM10-1','PM10-2','PM10-3','weekend01']], train['PM10'])

yp_train = model.predict(train.loc[:,['PM10-1','PM10-2','PM10-3','weekend01']])
yp = model.predict(test.loc[:,['PM10-1','PM10-2','PM10-3','weekend01']])

# note. This is not a pure AR model beccause of the 'weekend01' covariate

plt.figure(figsize = (20, 4))
plt.plot(df['datetime'],df['PM10'],label='original')
plt.plot(test['datetime'],yp,label='forecast')
plt.legend(loc="upper left")
plt.show()

print('MAE = ',mean_absolute_error(test['PM10'],yp))
MAE =  6.784168473816068

from joblib import dump, load
dump(model, 'model.joblib')

model = load('model.joblib')

yp = model.predict(test.loc[:,['PM10-1','PM10-2','PM10-3','weekend01']])

plt.figure(figsize = (20, 4))
plt.plot(df['datetime'],df['PM10'],label='original')
plt.plot(test['datetime'],yp,label='forecast')
plt.legend(loc="upper left")
plt.show()

