import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("/home/mlardy2/Documents/work/PINN/Pinn/output/output400pytorch3.dat") # i, j, rho, u, v


U = data[:,[3,4]] # shape = (N,2)
P = data[:,2] / 3. # shape = (N)
X = data[:,[0,1]] # shape = (N,2)


colors = plt.cm.rainbow(U)
#plt.plot(X[:,0],X[:,1],color=colors[i])
plt.scatter(X[:,0], X[:,1], c=U[:,0], s=100)
cbar=plt.colorbar()
plt.xlabel('i')
plt.ylabel('j')
plt.show()

