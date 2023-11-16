import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



df = np.loadtxt('output/loss600.dat').T
plt.title('losses')
plt.plot(np.linspace(0,len(df[1]),len(df[1])),df[2],label="loss phy")
#plt.plot(np.linspace(0,len(df[1]),len(df[1])),df[0],label="loss data")
plt.xlabel('Epoch')
plt.legend()
plt.yscale("log")
plt.show()

df = np.loadtxt('output/lbd600.dat').T
plt.title('$\lambda_1$')
#plt.plot(np.linspace(0,len(df[1]),len(df[1])),df[1],label="loss phy")
plt.plot(np.linspace(0,len(df),len(df)),df,label="loss data")
plt.xlabel('Epoch')
plt.legend()
plt.show()


List=[50,200,400]

for i in List:
    df=np.loadtxt(f'output/loss{i}.dat').T
    plt.plot(np.linspace(0,len(df[1]),len(df[1])),df[2]*1/i,label=f"Ntrain {i}")

plt.title('losses')
plt.xlabel('Epoch')
plt.legend()
plt.yscale("log")
plt.show()



for i in List:
    df = np.loadtxt(f'output/lbd{i}.dat').T
    plt.plot(np.linspace(0,len(df),len(df)),df,label=f"Ntrain {i}")

plt.title('$\lambda_1$')
plt.plot(np.ones(len(df))*0.5,'--',label="$\lambda^*$")
plt.xlabel('Epoch')
plt.legend()
plt.show()
