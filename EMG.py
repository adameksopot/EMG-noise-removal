from statsmodels.graphics.tsaplots import plot_pacf
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
import numpy as np

# load EMG
column = ['t', 'emg']
dataset = pd.read_csv('emg_healthy.txt', sep='\s', names=column, engine='python')
data = pd.DataFrame(dataset)
data['t'] = data['t'].astype(float)
data['emg'] = pd.to_numeric(data['emg'], errors='coerce')

# check signal
data.dropna(inplace=True)
data.isnull().sum().sum()

# Autoregressive
Y = data["emg"].values
plot_pacf(Y, lags=10)
model = AutoReg(Y, lags=(2), trend='t', seasonal=False, old_names=True)
result = model.fit()
y_pred = result.fittedvalues
# plot filtered signal and its noisy version
ax = pd.Series(Y).plot(color='lightgray')
pd.Series(y_pred).plot(color='black', ax=ax, figsize=(12, 10))
Y = np.delete(Y, 0)
Y = np.delete(Y, 0)
noise = Y - y_pred
Psignal = y_pred.var()
Pnoise = noise.var()
print("\nsignaltonoise ratio for y_pred : ", 10 * np.log10(Psignal / Pnoise))
