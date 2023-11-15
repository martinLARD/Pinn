import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



df = np.loadtxt('loss1.dat').T
plt.title('losses')
plt.plot(np.linspace(0,len(df[1]),len(df[1])),df[1],label="loss phy")
plt.plot(np.linspace(0,len(df[1]),len(df[1])),df[0],label="loss data")
plt.xlabel('Epoch')
plt.legend()
plt.yscale("log")

plt.show()

df = np.loadtxt('lbd1.dat').T
plt.title('$\lambda_1$')
#plt.plot(np.linspace(0,len(df[1]),len(df[1])),df[1],label="loss phy")
plt.plot(np.linspace(0,len(df),len(df)),df,label="loss data")
plt.xlabel('Epoch')
plt.legend()
plt.show()