import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# df = np.loadtxt('output/loss600.dat').T
# plt.title('losses')
# plt.plot(np.linspace(0,len(df[1]),len(df[1])),df[2],label="loss phy")
# #plt.plot(np.linspace(0,len(df[1]),len(df[1])),df[0],label="loss data")
# plt.xlabel('Epoch')
# plt.legend()
# plt.yscale("log")
# plt.show()

# df = np.loadtxt('output/lbd600.dat').T
# plt.title('$\lambda_1$')
# #plt.plot(np.linspace(0,len(df[1]),len(df[1])),df[1],label="loss phy")
# plt.plot(np.linspace(0,len(df),len(df)),df,label="loss data")
# plt.xlabel('Epoch')
# plt.legend()
# plt.show()


List=[50,200,600,400]

for i in List:
    df=np.loadtxt(f'output/loss{i}.dat').T
    plt.plot(np.linspace(0,len(df[1]),len(df[1])),df[2]*1/i,label=f"Ntrain {i}")

plt.title('losses tot')
plt.xlabel('Epoch')
plt.legend()
plt.yscale("log")
plt.show()

for i in List:
    df=np.loadtxt(f'output/loss{i}.dat').T
    plt.plot(np.linspace(0,len(df[1]),len(df[1])),df[0]*1/i,label=f"Ntrain {i}")

plt.title('losses data')
plt.xlabel('Epoch')
plt.legend()
plt.yscale("log")
plt.show()

for i in List:
    df=np.loadtxt(f'output/loss{i}.dat').T
    plt.plot(np.linspace(0,len(df[1]),len(df[1])),df[1]*1/i,label=f"Ntrain {i}")

plt.title('losses phy')
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


for i in List:
    dlbd = np.loadtxt(f'output/lbd{i}.dat').T
    dloss = np.loadtxt(f'output/loss{i}.dat').T
    plt.scatter(dlbd[-1],dloss[2][-1],label=f"Ntrain {i}")
plt.scatter(0.5,0,label='$\lambda^*$',marker="*")
plt.title('scatter')
plt.xlabel('$\lambda$')
plt.legend()
plt.ylabel('loss')
plt.show()

dlbd1 = np.loadtxt(f'output/lbd400datadown.dat').T
dlbd2 = np.loadtxt(f'output/lbd400phyup.dat').T
dlbd3 = np.loadtxt(f'output/lbd400.dat').T
plt.plot(np.linspace(0,len(dlbd1),len(dlbd1)),dlbd1,label="f_data*0.001")
plt.plot(np.linspace(0,len(dlbd1),len(dlbd1)),dlbd2,label="f_phy same f_data")
plt.plot(np.linspace(0,len(dlbd1),len(dlbd1)),dlbd3,label="f_phy*0.001")
plt.title('$\lambda$ with different weights for loss fct')
plt.legend()
plt.show()

dlbd1 = np.loadtxt(f'output/loss400datadown.dat').T
dlbd2 = np.loadtxt(f'output/loss400phyup.dat').T
dlbd3 = np.loadtxt(f'output/loss400.dat').T

plt.plot(np.linspace(0,len(dlbd1[0]),len(dlbd1[0])),dlbd1[2],label="f_data*0.001")
plt.plot(np.linspace(0,len(dlbd1[2]),len(dlbd1[2])),dlbd2[2],label="f_phy same f_data")
plt.plot(np.linspace(0,len(dlbd1[2]),len(dlbd1[2])),dlbd3[2],label="f_phy*0.001")
plt.title('loss with different weights for loss fct')
plt.yscale('log')
plt.legend()
plt.show()


dlbd1 = np.loadtxt(f'output/lbd4004L.dat').T
dlbd2 = np.loadtxt(f'output/lbd4008L.dat').T
dlbd3 = np.loadtxt(f'output/lbd400.dat').T
plt.plot(np.linspace(0,len(dlbd1),len(dlbd1)),dlbd1,label="8 Layers")
plt.plot(np.linspace(0,len(dlbd1),len(dlbd1)),dlbd2,label="4 Layers")
plt.plot(np.linspace(0,len(dlbd1),len(dlbd1)),dlbd3,label="6 Layers (Default)")
plt.title('$\lambda$ with different nbr of layers')
plt.legend()
plt.show()

dlbd1 = np.loadtxt(f'output/loss4008L.dat').T
dlbd2 = np.loadtxt(f'output/loss4004L.dat').T
dlbd3 = np.loadtxt(f'output/loss400.dat').T

plt.plot(np.linspace(0,len(dlbd1[0]),len(dlbd1[0])),dlbd1[2],label="8 Layers")
plt.plot(np.linspace(0,len(dlbd1[2]),len(dlbd1[2])),dlbd2[2],label="4 Layers")
plt.plot(np.linspace(0,len(dlbd1[2]),len(dlbd1[2])),dlbd3[2],label="6 Layers (Defaults")
plt.title('loss with different nbr of Layers')
plt.yscale('log')
plt.legend()
plt.show()

df1=np.loadtxt(f'output/lbd{i}.dat').T
