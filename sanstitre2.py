import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



df = np.loadtxt('loss.dat').T
plt.title('losses')
#plt.plot(np.linspace(0,len(df[1]),len(df[1])),df[1],label="loss phy")
plt.plot(np.linspace(0,len(df[1]),len(df[1])),df[2],label="loss tot")
plt.xlabel('Epoch (e1)')
plt.legend()
plt.show()

df = np.loadtxt('lbd.dat').T
plt.title('$\lambda_1$')
#plt.plot(np.linspace(0,len(df[1]),len(df[1])),df[1],label="loss phy")
plt.plot(np.linspace(0,len(df),len(df)),df,label="loss data")
plt.xlabel('Epoch (e3)')
plt.legend()
plt.yscale("log")
plt.show()