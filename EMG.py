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

