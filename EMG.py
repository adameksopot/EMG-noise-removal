from statsmodels.graphics.tsaplots import plot_pacf
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

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
Y1 = np.delete(Y, 0)
Y2 = np.delete(Y1, 0)
noise = Y2 - y_pred
Psignal = y_pred.var()
Pnoise = noise.var()
print("\nSignal to noise ratio for y_pred : ", 10 * np.log10(Psignal / Pnoise))

# Savitzky-Golay
X = data["t"].values
y_pred2 = savgol_filter(Y, window_length=51, polyorder=3)
noise2 = Y - y_pred2
Psignal2 = y_pred2.var()
Pnoise2 = noise2.var()
print("\nSignal to noise ratio for Savitzky-Golay : ", 10 * np.log10(Psignal2 / Pnoise2))

# count peak to peak amplitude
print(np.ptp(noise))
print(np.ptp(y_pred2))

# KNN Regressor model

from sklearn.neighbors import KNeighborsRegressor

# X is time in ms
# Y is emg signal [mV]
X_KNN = data.iloc[:, :-1].values
Y_KNN = data.iloc[:, 1].values
clf = KNeighborsRegressor(n_neighbors=50, weights='uniform')
clf.fit(X_KNN, Y_KNN)
y_pred3 = clf.predict(X_KNN)
# calc noise
noise3 = Y - y_pred3
Psignal3 = y_pred3.var()
Pnoise3 = noise3.var()
print("\nSignal to noise ratio for KNN : ", 10 * np.log10(Psignal3 / Pnoise3))
