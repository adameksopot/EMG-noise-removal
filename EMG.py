from statsmodels.graphics.tsaplots import plot_pacf
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from numpy import floor, log2, log10
import pywt
from statsmodels.robust import mad
import statistics

# load EMG
column = ['t', 'emg']
dataset = pd.read_csv('emg_healthy.txt', sep='\s', names=column, engine='python')
data = pd.DataFrame(dataset)
data['t'] = data['t'].astype(float)
data['emg'] = pd.to_numeric(data['emg'], errors='coerce')

# check signal
data.dropna(inplace=True)
data.isnull().sum().sum()

#snr function
def signaltonoiseratio(emg,emg_denoised):
    noise = emg_denoised-emg
    Psignal = emg_denoised.var()
    Pnoise = noise.var()
    return 10*log10(Psignal/Pnoise)

#mse function
def meansquareerror(emg,emg_denoised):
    return np.square(np.subtract(emg, emg_denoised)).mean()
    
#cross correlation
def crosscorelation(emg,emg_denoised):
    stdev_emg = statistics.stdev(emg)
    stdev_emg_de = statistics.stdev(emg_denoised)
    return ((emg-emg.mean())*(emg_denoised-emg_denoised.mean())).mean()/(stdev_emg*stdev_emg_de)


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

print("SNR for Autoregressive : ",signaltonoiseratio(Y2,y_pred))

print("MSE for Autoregressive : ", meansquareerror(Y2,y_pred))

print("Cross correlation for Autoregressive : ", crosscorelation(Y2,y_pred))

# Savitzky-Golay
X = data["t"].values
y_pred2 = savgol_filter(Y, window_length=51, polyorder=3)
noise2 = Y - y_pred2
Psignal2 = y_pred2.var()
Pnoise2 = noise2.var()
print("\nSignal to noise ratio for Savitzky-Golay : ", 10 * np.log10(Psignal2 / Pnoise2))

print("SNR for Savitzky-Golay : ",signaltonoiseratio(Y,y_pred2))

print("MSE for Savitzky-Golay : ", meansquareerror(Y,y_pred2))

print("Cross correlation for Savitzky-Golay : ", crosscorelation(Y,y_pred2))

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

print("SNR for KNN Regressor : ",signaltonoiseratio(Y,y_pred3))

print("MSE for KNN Regressor : ", meansquareerror(Y,y_pred3))

print("Cross correlation for KNN Regressor : ", crosscorelation(Y,y_pred3))
#wavlet transform

signal_emg = data["emg"].values
wavelet="db4" 
level=6

# calculate the wavelet coefficients
coeff = pywt.wavedec(signal_emg, wavelet, mode="per" )
#levels  = floor(log2(signal_emg.shape[0])) 

# calculate a threshold
sigma = mad( coeff[-level] )

uthresh = sigma * np.sqrt( 2*np.log( len( signal_emg ) ) )
print(uthresh)

coeff[1:] = ( pywt.threshold( i, value=uthresh, mode="soft" ) for i in coeff[1:] )

# reconstruct the signal using the thresholded coefficients
y_denoised = pywt.waverec( coeff, wavelet, mode="per" )
    
f, ax = plt.subplots(figsize=(20, 10))
plt.plot( signal_emg, color="b", alpha=0.5 )
plt.plot( y_denoised, color="r" )
ax.set_xlim((0,len(y_denoised)))

print("SNR for wavlet transform : ",signaltonoiseratio(signal_emg,y_denoised))

print("MSE for wavlet transform : ", meansquareerror(signal_emg,y_denoised))

print("Cross correlation for wavlet transform : ", crosscorelation(signal_emg,y_denoised))