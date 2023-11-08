import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.cluster import KMeans


df = np.loadtxt('loss.dat').T
plt.title('losses')
plt.plot(np.linspace(0,len(df[1]),len(df[1])),df[1],label="loss phy")
plt.plot(np.linspace(0,len(df[1]),len(df[1])),df[0],label="loss data")
plt.xlabel('Epoch (e3)')
plt.legend()
plt.ylim(0,2*10**-5)
plt.show()

plt.title('$\lambda_1$')
plt.plot(np.ones(len(df[1]))*0.5,linestyle='dashed',label='real value')
plt.plot(np.linspace(0,len(df[1]),len(df[1])),df[2],label='predict')
plt.xlabel('Epoch (e3)')
plt.ylim(0.45,0.55)
plt.show()


