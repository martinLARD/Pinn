import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.cluster import KMeans

df = np.loadtxt('loss.dat').T
plt.plot(np.linspace(0,len(df[1])*100,len(df[1])),df[1],label="loss phy")
plt.plot(np.linspace(0,len(df[1])*100,len(df[1])),df[0],label="loss data")
plt.xlabel('Epoch')
plt.legend()
plt.show()


plt.plot(np.linspace(0,len(df[1])*100,len(df[1])),df[2])
plt.show()


