from statsmodels.graphics.tsaplots import plot_pacf
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from numpy import log10
import pywt
from statsmodels.robust import mad
import statistics

# load EMG
column = ['t', 'emg']
dataset = pd.read_csv('emg_healthy.txt', sep='\s', names=column, engine='python')
data = pd.DataFrame(dataset)[:2001]
data['t'] = data['t'].astype(float)
data['emg'] = pd.to_numeric(data['emg'], errors='coerce')

# check signal
data.dropna(inplace=True)
data.isnull().sum().sum()

# generate noise
gaussian_noise = np.random.normal(0, 0.01, 2000)  # 0.01


# snr function
def signaltonoiseratio(noise, emg_denoised):
    Psignal = emg_denoised.var()
    Pnoise = noise.var()
    return 10 * log10(Psignal / Pnoise)


# mse function
def meansquareerror(emg, emg_denoised):
    return np.square(np.subtract(emg, emg_denoised)).mean()


# cross correlation
def crosscorelation(emg, emg_denoised):
    stdev_emg = statistics.stdev(emg)
    stdev_emg_de = statistics.stdev(emg_denoised)
    return ((emg - emg.mean()) * (emg_denoised - emg_denoised.mean())).mean() / (stdev_emg * stdev_emg_de)


# signals; one is orignal and the second is artificially noised
Y = data["emg"].values  # orginal
YwithNoise = data["emg"].values + gaussian_noise  # with noise

# Autoregressive

plot_pacf(Y, lags=10)
model = AutoReg(Y, lags=2, trend='t', seasonal=False, old_names=True)
result = model.fit()
y_pred = result.fittedvalues
# plot filtered signal and its noisy version
ax = pd.Series(Y).plot(color='lightgray')
pd.Series(y_pred).plot(color='black', ax=ax, figsize=(12, 10))
Y1 = np.delete(Y, 0)
Y2 = np.delete(Y1, 0)
noise = Y2 - y_pred
# this is for the Y and denoised - here the noise is from the subtraction
print("\nMetrics without Gaussian noise")
print("SNR for Autoregressive : ", signaltonoiseratio(noise, y_pred))
print("MSE for Autoregressive : ", meansquareerror(Y2, y_pred))
print("Cross correlation for Autoregressive : ", crosscorelation(Y2, y_pred))
# AR with Gaussian noise
model = AutoReg(YwithNoise, lags=3, trend='t', seasonal=False, old_names=True)
result = model.fit()
y_pred = result.fittedvalues
# plot filtered signal and its noisy version
ax = pd.Series(YwithNoise).plot(color='lightgray')
pd.Series(y_pred).plot(color='black', ax=ax, figsize=(12, 10))
Y1 = np.delete(YwithNoise, 0)
Y2 = np.delete(Y1, 0)
Y3 = np.delete(Y2, 0)
noise = Y3 - y_pred

# this is for YwithNoise and denoised - here the noise is gaussian noise
print("\nMetrics with Gaussian noise")
print("SNR for Autoregressive : ", signaltonoiseratio(gaussian_noise, y_pred))
print("MSE for Autoregressive : ", meansquareerror(Y3, y_pred))
print("Cross correlation for Autoregressive : ", crosscorelation(Y3, y_pred))

# Savitzky-Golay
X = data["t"].values
y_pred2 = savgol_filter(Y, window_length=51, polyorder=3)
noise = Y - y_pred2
# this is for the Y and denoised - here the noise is from the subtraction
print("\nMetrics without Gaussian noise")
print("SNR for Savitzky-Golay : ", signaltonoiseratio(noise, y_pred2))  # count noise from SGfilter
print("MSE for Savitzky-Golay : ", meansquareerror(Y, y_pred2))
print("Cross correlation for Savitzky-Golay : ", crosscorelation(Y, y_pred2))

# this is for YwithNoise and denoised - here the noise is gaussian noise
y_pred2 = savgol_filter(YwithNoise, window_length=51, polyorder=3)
noise = YwithNoise - y_pred2
print("\nMetrics with Gaussian noise")
print("SNR for Savitzky-Golay : ", signaltonoiseratio(gaussian_noise, y_pred2))
print("MSE for Savitzky-Golay : ", meansquareerror(YwithNoise, y_pred2))
print("Cross correlation for Savitzky-Golay : ", crosscorelation(YwithNoise, y_pred2))
# count peak to peak amplitude
print("\nCount peak to peak amplitude to determine if a process was succesfull")
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
noise = Y_KNN - y_pred3
# this is for the Y and denoised - here the noise is from the subtraction
print("\nMetrics without Gaussian noise")
print("SNR for KNN Regressor : ", signaltonoiseratio(noise, y_pred3))  # count noise from KNN
print("MSE for KNN Regressor : ", meansquareerror(Y, y_pred3))
print("Cross correlation for KNN Regressor : ", crosscorelation(Y, y_pred3))

# this is for YwithNoise and denoised - here the noise is gaussian noise

Y_KNN = data.iloc[:, 1].values + gaussian_noise
clf.fit(X_KNN, Y_KNN)
y_pred3 = clf.predict(X_KNN)
print("\nMetrics with Gaussian noise")
print("SNR for KNN Regressor : ", signaltonoiseratio(gaussian_noise, y_pred3))
print("MSE for KNN Regressor : ", meansquareerror(Y, y_pred3))
print("Cross correlation for KNN Regressor : ", crosscorelation(Y, y_pred3))


# wavlet transform

def wavlet_transform(Y, wavelet, level):
    signal_emg = Y
    # wavelet="db4"
    # level=2

    # calculate the wavelet coefficients
    coeff = pywt.wavedec(signal_emg, wavelet, mode="per")
    # levels  = floor(log2(signal_emg.shape[0]))

    # calculate a threshold
    sigma = mad(coeff[-level])

    uthresh = sigma * np.sqrt(2 * np.log(len(signal_emg)))
    # print(uthresh)

    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in coeff[1:])

    # reconstruct the signal using the thresholded coefficients
    y_denoised = pywt.waverec(coeff, wavelet, mode="per")

    f, ax = plt.subplots(figsize=(20, 10))
    plt.plot(signal_emg, color="b", alpha=0.5)
    plt.plot(data["emg"].values, color="g", alpha=0.5)
    plt.plot(y_denoised, color="r")
    ax.set_xlim((0, len(y_denoised)))
    return y_denoised


Y_denoised_WT = wavlet_transform(Y, "db4", 2)
Y_denoised2_WT = wavlet_transform(YwithNoise, "db4", 2)

noiseWT = Y - Y_denoised_WT

# this is for the Y and denoised - here the noise is from the subtraction
print("\nMetrics without Gaussian noise")
print("SNR for wavlet transform : ", signaltonoiseratio(noiseWT, Y_denoised_WT))
print("MSE for wavlet transform : ", meansquareerror(Y, Y_denoised_WT))
print("Cross correlation for wavlet transform : ", crosscorelation(Y, Y_denoised_WT))
# this is for YwithNoise and denoised - here the noise is gaussian noise
print("\nMetrics with Gaussian noise")
print("SNR for wavlet transform : ", signaltonoiseratio(gaussian_noise, Y_denoised2_WT))
print("MSE for wavlet transform : ", meansquareerror(Y, Y_denoised2_WT))
print("Cross correlation for wavlet transform : ", crosscorelation(Y, Y_denoised2_WT))
