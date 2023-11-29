
import numpy as np
import matplotlib.pyplot as plt



df2=np.loadtxt('output/loss400bfgs.dat').T
plt.title('losses')


plt.plot(np.linspace(0,len(df2[1]),len(df2[1])),df2[2],label="loss evolve cst")
#plt.vlines(x=26390,ymin=0,ymax=max(df2[2]),ls=':',colors='r')

plt.xlabel('Epoch')
plt.legend()
plt.yscale("log")
plt.savefig("testloss0.png")

plt.show()
plt.close()



df2=np.loadtxt('output/lbd400bfgs.dat').T

plt.title('$\lambda_1$')
plt.plot(np.ones(len(df2))*0.75,'--',label="$\lambda^*$")

plt.plot(np.linspace(0,len(df2),len(df2)),df2,label="lambda value")
#plt.vlines(x=26390,ymin=0,ymax=1,ls=':',colors='r')
plt.xlabel('Epoch')
plt.legend()
plt.savefig("testlbd.png")
plt.show()


# df=np.loadtxt('/home/mlardy2/Documents/work/PINN/data_premiertest/loss400_20neur.dat').T

# df1=np.loadtxt('/home/mlardy2/Documents/work/PINN/data_premiertest/loss400_10neur.dat').T
# df2=np.loadtxt('/home/mlardy2/Documents/work/PINN/data_premiertest/loss400.dat').T
# plt.title('losses')

# plt.plot(np.linspace(0,len(df2[1]),len(df2[1])),df[0],label="20 neurones")
# plt.plot(np.linspace(0,len(df2[1]),len(df2[1])),df1[0],label="10 neurones")
# plt.plot(np.linspace(0,len(df2[1]),len(df2[1])),df2[0],label="loss evolve cst")
# plt.xlabel('Epoch')
# plt.legend()
# plt.yscale("log")
# plt.show()
# plt.close()


# df=np.loadtxt('/home/mlardy2/Documents/work/PINN/data_premiertest/lbd400_20neur.dat').T
# df1=np.loadtxt('/home/mlardy2/Documents/work/PINN/data_premiertest/lbd400_10neur.dat').T
# df2=np.loadtxt('/home/mlardy2/Documents/work/PINN/data_premiertest/lbd400.dat').T

# plt.title('$\lambda_1$')
# plt.plot(np.ones(len(df2))*0.5,'--',label="$\lambda^*$")
# plt.plot(np.linspace(0,len(df2),len(df2)),df,label="20 neurones")
# plt.plot(np.linspace(0,len(df2),len(df2)),df1,label="10 neurones")
# plt.plot(np.linspace(0,len(df2),len(df2)),df2,label="cst")

# plt.xlabel('Epoch')
# plt.legend()
# plt.show()
# plt.savefig("lbddiff.png")


# List=[50,200,600,400]

# for i in List:
#     df=np.loadtxt(f'output/loss{i}.dat').T
#     plt.plot(np.linspace(0,len(df[1]),len(df[1])),df[2]*1/i,label=f"Ntrain {i}")

# plt.title('losses tot')
# plt.xlabel('Epoch')
# plt.legend()
# plt.yscale("log")
# plt.show()

# for i in List:
#     df=np.loadtxt(f'output/loss{i}.dat').T
#     plt.plot(np.linspace(0,len(df[1]),len(df[1])),df[0]*1/i,label=f"Ntrain {i}")

# plt.title('losses data')
# plt.xlabel('Epoch')
# plt.legend()
# plt.yscale("log")
# plt.show()

# for i in List:
#     df=np.loadtxt(f'output/loss{i}.dat').T
#     plt.plot(np.linspace(0,len(df[1]),len(df[1])),df[1]*1/i,label=f"Ntrain {i}")

# plt.title('losses phy')
# plt.xlabel('Epoch')
# plt.legend()
# plt.yscale("log")
# plt.show()

# for i in List:
#     df = np.loadtxt(f'output/lbd{i}.dat').T
#     plt.plot(np.linspace(0,len(df),len(df)),df,label=f"Ntrain {i}")

# plt.title('$\lambda_1$')
# plt.plot(np.ones(len(df))*0.5,'--',label="$\lambda^*$")
# plt.xlabel('Epoch')
# plt.legend()
# plt.show()


# for i in List:
#     dlbd = np.loadtxt(f'output/lbd{i}.dat').T
#     dloss = np.loadtxt(f'output/loss{i}.dat').T
#     plt.scatter(dlbd[-1],dloss[2][-1],label=f"Ntrain {i}")
# plt.scatter(0.5,0,label='$\lambda^*$',marker="*")
# plt.title('scatter')
# plt.xlabel('$\lambda$')
# plt.legend()
# plt.ylabel('loss')
# plt.show()

# dlbd1 = np.loadtxt(f'output/lbd400datadown.dat').T
# dlbd2 = np.loadtxt(f'output/lbd400phyup.dat').T
# dlbd3 = np.loadtxt(f'output/lbd400.dat').T
# plt.plot(np.linspace(0,len(dlbd1),len(dlbd1)),dlbd1,label="f_data*0.001")
# plt.plot(np.linspace(0,len(dlbd1),len(dlbd1)),dlbd2,label="f_phy same f_data")
# plt.plot(np.linspace(0,len(dlbd1),len(dlbd1)),dlbd3,label="f_phy*0.001")
# plt.title('$\lambda$ with different weights for loss fct')
# plt.legend()
# plt.xlabel("Epochs")
# plt.show()

# dlbd1 = np.loadtxt(f'output/loss400datadown.dat').T
# dlbd2 = np.loadtxt(f'output/loss400phyup.dat').T
# dlbd3 = np.loadtxt(f'output/loss400.dat').T

# plt.plot(np.linspace(0,len(dlbd1[0]),len(dlbd1[0])),dlbd1[2],label="f_data*0.001")
# plt.plot(np.linspace(0,len(dlbd1[2]),len(dlbd1[2])),dlbd2[2],label="f_phy same f_data")
# plt.plot(np.linspace(0,len(dlbd1[2]),len(dlbd1[2])),dlbd3[2],label="f_phy*0.001")
# plt.title('loss with different weights for loss fct')
# plt.yscale('log')
# plt.legend()
# plt.xlabel("Epochs")
# plt.close()
# plt.show()


# dlbd1 = np.loadtxt(f'output/lbd4004L.dat').T
# dlbd2 = np.loadtxt(f'output/lbd4008L.dat').T
# dlbd3 = np.loadtxt(f'output/lbd400.dat').T
# plt.plot(np.linspace(0,len(dlbd1),len(dlbd1)),dlbd1,label="8 Layers")
# plt.plot(np.linspace(0,len(dlbd1),len(dlbd1)),dlbd2,label="4 Layers")
# plt.plot(np.linspace(0,len(dlbd1),len(dlbd1)),dlbd3,label="6 Layers (Default)")
# plt.plot(np.ones(len(dlbd1))*0.5,'--',label="$\lambda^*$")
# plt.title('$\lambda$ with different nbr of layers')
# plt.legend()
# plt.savefig("CV_lbd_layers.png")
# plt.xlabel('Epoch')
# plt.close()
# plt.show()


# dlbd1 = np.loadtxt(f'output/loss4008L.dat').T
# dlbd2 = np.loadtxt(f'output/loss4004L.dat').T
# dlbd3 = np.loadtxt(f'output/loss400.dat').T

# plt.plot(np.linspace(0,len(dlbd1[0]),len(dlbd1[0])),dlbd1[2],label="8 Layers")
# plt.plot(np.linspace(0,len(dlbd1[2]),len(dlbd1[2])),dlbd3[2],label="6 Layers (Default)")
# plt.plot(np.linspace(0,len(dlbd1[2]),len(dlbd1[2])),dlbd2[2],label="4 Layers")

# plt.title('loss with different nbr of Layers')
# plt.yscale('log')
# plt.legend()
# plt.xlabel('Epoch')
# plt.savefig("loss_layers.png")
# plt.close()
# plt.show()



# dlbd = np.loadtxt(f'output/lbd400.dat').T
# dloss = np.loadtxt(f'output/loss400.dat').T
# dlbd1 = np.loadtxt(f'output/lbd4004L.dat').T
# dloss1 = np.loadtxt(f'output/loss4004L.dat').T
# dlbd2 = np.loadtxt(f'output/lbd4008L.dat').T
# dloss2 = np.loadtxt(f'output/loss4008L.dat').T

# plt.scatter(dlbd[-1],dloss[2][-1],label="6 Layers (Default)")
# plt.scatter(dlbd1[-1],dloss1[2][-1],label="4 Layers")
# plt.scatter(dlbd2[-1],dloss2[2][-1],label="8 Layers")
# plt.scatter(0.5,0,label='$\lambda^*$',marker="*")
# plt.title('scatter')
# plt.xlabel('$\lambda$')
# plt.legend()
# plt.ylabel('loss')
# plt.savefig("scatter2.png")
# plt.title("scatter")

# plt.show()

