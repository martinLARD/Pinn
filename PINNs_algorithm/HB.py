import torch
import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import exists


path_data='/home/mlardy2/Documents/work/simulation_constric2/snaps/'
path_data = "/home/mlardy2/Documents/work/simulation/snaps/"
namedata='select'
data=f'Macro_{namedata}.dat'

#Set default dtype to float32
torch.set_default_dtype(torch.float32)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

if device == 'cuda': 
    print(torch.cuda.get_device_name()) 

mainpath=f'/home/mlardy2/Documents/work/PINN/Pinn/output/HB_new'   

#save the output in files
nbr=np.random.randint(0,1000)

lossfile=f'loss_{nbr}'
lbdfile=f'lbd_{nbr}'


file_exists = exists(f'{mainpath}/{lossfile}.dat')
while file_exists==True:
    print(nbr)
    nbr+=1
    lossfile=f'loss_{nbr}'
    lbdfile=f'lbd_{nbr}'
    file_exists = exists(f'{mainpath}/{lossfile}.dat')
print('output save in files', nbr)
loss_file = open(f"{mainpath}/{lossfile}.dat",'w')
loss_file.close()
loss_file = open(f"{mainpath}/{lbdfile}.dat","w")
loss_file.close()




# Size of the NN
N_train = 800

layers = [2, 20, 20, 20, 20, 2]

# Load Data
data = np.loadtxt(f"{path_data}{data}") # i, j, rho, u, v

U = data[:,[3,4]] # shape = (N,2)
P = data[:,2] / 3. # shape = (N)
X = data[:,[0,1]] # shape = (N,2)
Eta=data[:,-2]
Shearate = data[:,-1]
N = X.shape[0]
Nx = int(np.sqrt(N))
Ny = int(np.sqrt(N))

# Rearrange Data
XX = X[:,0]
YY = X[:,1]

UU = U[:,0]
VV = U[:,1]
PP = P[:]
EE=Eta[:]
SS =Shearate[:]

s = SS
x = XX # This forms a rank-2 array with a single vector component, shape=(N,1)
y = YY
u = UU
v = VV
p = PP
eta = EE
N=len(x)

plt.scatter(x,y,c=u)
plt.show()
######################################################################
######################## Data Preprocessing ##########################
######################################################################
# Training Data
walls=False
idx = np.random.choice(N, N_train, replace=False)

if walls==True: #take into account the walls in the data
    print(' /!\ WAVY /!\ ')
    wall = np.loadtxt("/home/mlardy2/Documents/work/simulation_constric2/snaps/Markers_on_live.dat")
    wall_inf_y=wall[400:,1]
    wall_sup_y=wall[:400,1]
    wall_inf_x=wall[400:,0]
    wall_sup_x=wall[:400,0]
    ysorti=[]
    xsorti=[]
    usorti=[]
    vsorti=[]
    etasorti=[]
    sorti=[]
    leps=3
    deb=int(min(x))
    fin=int(max(x))
    for i in range(deb,fin):
        close=np.argmin(abs(i-wall_inf_x))
        temp=y[x==i]
        #aa=np.logical_and(temp>=55,temp<=73)

        aa=np.logical_and(temp>wall_inf_y[close]+leps,temp<wall_sup_y[close]-leps)
        #aa=(i-62.4)**2+(temp-62.4)**2>28**2
        #aa=np.logical_and(temp>0,temp<50)
        #aa=s[x==i]<0.8e-6

        ytemp=y[x==i][aa]
        xtemp=x[x==i][aa]
        utemp=u[x==i][aa]
        vtemp=v[x==i][aa]
        etatemp=eta[x==i][aa]
        stemp=s[x==i][aa]
        ysorti=np.concatenate((ysorti,ytemp))
        xsorti=np.concatenate((xsorti,xtemp))
        usorti=np.concatenate((usorti,utemp))
        vsorti=np.concatenate((vsorti,vtemp))
        etasorti=np.concatenate((etasorti,etatemp))
        sorti=np.concatenate((sorti,stemp))
        dist=(abs(xsorti-np.mean(x))+abs(ysorti-np.mean(y)))
        distnorm=dist/max(dist)
        #proba=(1-sorti/sum(sorti))**2/sum(1-(sorti/sum(sorti))**2)#(1-(distnorm))**2/sum((1-(distnorm))**2)
        stest=np.digitize(sorti,np.linspace(min(sorti),max(sorti),int(len(xsorti)/10)))
        proba=(stest**4)/sum(stest**(4))#/sum(stest/sum(stest))#/sum(1-stest**2/sum(stest**2))
    print(len(xsorti))
    N=len(ysorti)
    idx = np.random.choice(N, N_train, replace=False,p=proba)


x_train = x[idx]
y_train = y[idx]
u_train = u[idx]
v_train = v[idx]
s_train = s[idx]
eta_train = eta[idx]
if walls==True:
    x_train = xsorti[idx]
    y_train = ysorti[idx]
    u_train = usorti[idx]
    v_train = vsorti[idx]
    eta_train = etasorti[idx]
    s_train = sorti[idx]

plt.scatter(x_train,y_train,c=s_train)
if walls==True:
   plt.scatter(wall[:,0],wall[:,1])
plt.show()

plt.scatter(s_train,eta_train)
plt.show()

# Normalization

uM=min(u_train)
vM=min(v_train)

ubar=np.mean(u_train)
vbar=np.mean(v_train)
# Fixing D and U0 defining the Bingham number
# Arbitrary definition: with experimental data, we can use D and U0 from the data
D = 64#np.mean(x_train)
U0 = 1e-4#/50#vbar*10**1

u_train = (u_train-ubar) / (U0)#(psi*u_train-ubar) / U0
v_train = (v_train-vbar) / (U0)#(psi*v_train-vbar) / U0

X_train = np.zeros((N_train,2))
for l in range(0, N_train) :
    X_train[l,0] = x_train[l]#(x_train[l]-xmin)/(xmax-xmin)
    X_train[l,1] = y_train[l]#(y_train[l]-ymin)/(ymax-ymin)

# Simon
X_train[:,0] = X_train[:,0] / D
X_train[:,1] = X_train[:,1] / D
x = x / D
y = y / D
# Simon

Xmin=X_train.min()
Xmax=X_train.max()

u_train=torch.from_numpy(u_train).to(device)
v_train=torch.from_numpy(v_train).to(device)

print(r"$\gamma_c_nn",U0/D)

# plt.scatter(x,y,c=u)
# plt.show()
######################################################################
######################## Neural Network###############################
######################################################################


class Sequentialmodel(nn.Module):
    
    def __init__(self,layers):
        super().__init__() #call __init__ from parent class 
              
        'activation function'
        self.activation = nn.Tanh()
    
        'Initialise neural network as a list using nn.Modulelist'  
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        
        self.iter = 0
    
        'Xavier Normal Initialization'
        # std = gain * sqrt(2/(input_dim+output_dim))
        for i in range(len(layers)-1):
            
            # weights from a normal distribution with 
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)
            
        self.alpha_1 = nn.Parameter(torch.ones([1], dtype=torch.float32)*1.2)

        self.alpha_2 = nn.Parameter(torch.ones([1], dtype=torch.float32)*4.)
        
        

        
    'foward pass'
    def forward(self,x,y):
        
        
        X=torch.stack([x,y],axis=1)
        if torch.is_tensor(x) != True:         
            X = torch.tensor(X).to(device)                

        #preprocessing input 
        X_normed = (X - Xmin)/(Xmax - Xmin)+1e-16 #feature scaling
        #convert to float
        a = X_normed.float()
        #print(a.size())

        for i in range(len(layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
            
        a = self.linears[-1](a)
        return a
    
    def weight(self, X):

        x_train=X[:,0]
        y_train=X[:,1]
        
        alpha_1 = torch.abs(self.alpha_1)
        n=torch.abs(self.alpha_2)

        x = torch.from_numpy(x_train).to(device)
        y = torch.from_numpy(y_train).to(device)
        
        x.requires_grad = True
        y.requires_grad = True
        psi_and_p = self.forward(x,y)
        psi = psi_and_p[:,0:1].T[0]
        p = psi_and_p[:,1:2].T[0]
        u = autograd.grad(psi,y,torch.ones(x.shape).to(device), retain_graph=True, create_graph=True)[0]
        v = -autograd.grad(psi,x,torch.ones(x.shape).to(device), retain_graph=True, create_graph=True)[0]
        
        u_x = autograd.grad(u,x,torch.ones(x.shape).to(device), create_graph=True)[0]
        u_y = autograd.grad(u,y,torch.ones(x.shape).to(device), create_graph=True)[0]
    
        v_x = autograd.grad(v,x,torch.ones(x.shape).to(device), create_graph=True)[0]
        v_y = autograd.grad(v,y,torch.ones(x.shape).to(device), create_graph=True)[0]
           
        p_x = autograd.grad(p,x,torch.ones(x.shape).to(device), create_graph=True)[0]
        p_y = autograd.grad(p,y,torch.ones(x.shape).to(device), create_graph=True)[0]
        S11 = u_x
        S22 = v_y
        S12 = 0.5 * (u_y + v_x)

        gammap = (2.*(S11**2. + 2.*S12**2. + S22**2.))**(0.5)


        eta_star = alpha_1*(gammap)**(-1.) + (gammap)**(n-1)
        
        S11 = S11
        S22 = S22
        S12 = S12

        sig11 = 2. * eta_star * S11 
        sig12 = 2. * eta_star * S12  
        sig22 = 2. * eta_star * S22 
        
        sig11_x = autograd.grad(sig11,x,torch.ones(x.shape).to(device),create_graph=True)[0]
        sig12_x = autograd.grad(sig12,x,torch.ones(x.shape).to(device),create_graph=True)[0]
        sig12_y = autograd.grad(sig12,y,torch.ones(x.shape).to(device),create_graph=True)[0]
        sig22_y = autograd.grad(sig22,y,torch.ones(x.shape).to(device),create_graph=True)[0] 

        kappa = 1*(U0/D) * (U0/D)**(-n)

        f_u = (- p_x * kappa + sig11_x + sig12_y) 
        f_v = (- p_y * kappa + sig12_x + sig22_y)
         
        
        loss_phy =  (torch.sum(torch.square(f_u)) + torch.sum(torch.square(f_v)))
        loss_u=torch.sum(torch.square(u_train - u)) + torch.sum(torch.square(v_train - v))
        w=torch.floor(torch.log10(loss_u/loss_phy))
        return w

    def loss_PDE(self, X, u_train, v_train, w):
        
        x_train=X[:,0]
        y_train=X[:,1]
        
        alpha_1 = self.alpha_1
        n=self.alpha_2

        x = torch.from_numpy(x_train).to(device)
        y = torch.from_numpy(y_train).to(device)
        x.requires_grad = True
        y.requires_grad = True
 
        psi_and_p = self.forward(x,y)
        psi = psi_and_p[:,0:1].T[0]
        p = psi_and_p[:,1:2].T[0]
        u = autograd.grad(psi,y,torch.ones(x.shape).to(device), retain_graph=True, create_graph=True)[0]
        v = -autograd.grad(psi,x,torch.ones(x.shape).to(device), retain_graph=True, create_graph=True)[0]
        
        u_x = autograd.grad(u,x,torch.ones(x.shape).to(device), create_graph=True)[0]
        u_y = autograd.grad(u,y,torch.ones(x.shape).to(device), create_graph=True)[0]
    
        v_x = autograd.grad(v,x,torch.ones(x.shape).to(device), create_graph=True)[0]
        v_y = autograd.grad(v,y,torch.ones(x.shape).to(device), create_graph=True)[0]
           
        p_x = autograd.grad(p,x,torch.ones(x.shape).to(device), create_graph=True)[0]
        p_y = autograd.grad(p,y,torch.ones(x.shape).to(device), create_graph=True)[0]
        S11 = u_x
        S22 = v_y
        S12 = 0.5 * (u_y + v_x)

        gammap = (2.*(S11**2. + 2.*S12**2. + S22**2.))**(0.5)


        eta_star = (alpha_1*(gammap)**(-1.) + (gammap)**(n-1))
        S11 = S11
        S22 = S22
        S12 = S12

        sig11 = 2. * eta_star * S11 
        sig12 = 2. * eta_star * S12 
        sig22 = 2. * eta_star * S22  
        
        sig11_x = autograd.grad(sig11,x,torch.ones(x.shape).to(device),create_graph=True)[0]
        sig12_x = autograd.grad(sig12,x,torch.ones(x.shape).to(device),create_graph=True)[0]
        sig12_y = autograd.grad(sig12,y,torch.ones(x.shape).to(device),create_graph=True)[0]
        sig22_y = autograd.grad(sig22,y,torch.ones(x.shape).to(device),create_graph=True)[0] 

        kappa = 1*(U0/D)**(-n)*(U0/D)

        f_u = (- p_x * kappa + sig11_x + sig12_y) 
        f_v = (- p_y * kappa + sig12_x + sig22_y)

        loss_phy =  10**(w-1)*(torch.sum(torch.square(f_u)) + torch.sum(torch.square(f_v)))
        loss_u=torch.sum(torch.square(u_train - u)) + torch.sum(torch.square(v_train - v))
        loss_pos=10*(torch.maximum(torch.as_tensor(0),(-alpha_1))+torch.maximum(torch.as_tensor(0),-n))
        
        loss_file = open(f"{mainpath}/{lossfile}.dat","a")
        loss_file.write(f'{loss_u:.3e}'+" "+\
                            f'{loss_phy:.3e}'+" "+\
                            f'{loss_phy+loss_u:.3e}'+"\n")
        loss_file.close()
        
        lbd_file = open(f"{mainpath}/{lbdfile}.dat","a") 
        lbd_file.write(f'{self.alpha_1.item()}'+" "+\
                        f'{self.alpha_2.item()}'+"\n") 
        lbd_file.close()


        return loss_u+loss_phy+loss_pos, loss_u, loss_phy

    'callable for optimizer'                                       
    def closure(self):
        
        optimizer.zero_grad()
        
        loss = self.loss_PDE(X_train)
        
        loss.backward()
                
        self.iter += 1

        return loss   

    def eval(self,  x, y):
     
        alpha_1 = self.alpha_1
        n=self.alpha_2

        x = torch.from_numpy(x_train).to(device)
        y = torch.from_numpy(y_train).to(device)
        x.requires_grad = True
        y.requires_grad = True
 
        psi_and_p = self.forward(x,y)
        psi = psi_and_p[:,0:1].T[0]
        p = psi_and_p[:,1:2].T[0]
        u = autograd.grad(psi,y,torch.ones(x.shape).to(device), retain_graph=True, create_graph=True)[0]
        v = -autograd.grad(psi,x,torch.ones(x.shape).to(device), retain_graph=True, create_graph=True)[0]
        
        u_x = autograd.grad(u,x,torch.ones(x.shape).to(device), create_graph=True)[0]
        u_y = autograd.grad(u,y,torch.ones(x.shape).to(device), create_graph=True)[0]
    
        v_x = autograd.grad(v,x,torch.ones(x.shape).to(device), create_graph=True)[0]
        v_y = autograd.grad(v,y,torch.ones(x.shape).to(device), create_graph=True)[0]
           
        p_x = autograd.grad(p,x,torch.ones(x.shape).to(device), create_graph=True)[0]
        p_y = autograd.grad(p,y,torch.ones(x.shape).to(device), create_graph=True)[0]
        S11 = u_x
        S22 = v_y
        S12 = 0.5 * (u_y + v_x)

        gammap = (2.*(S11**2. + 2.*S12**2. + S22**2.))**(0.5)


        eta_star = (alpha_1*(gammap)**(-1.) + (gammap)**(n-1))

        return u, v, p, gammap, eta_star#, sig12, sig11, sig22, S12, S11, S22, f_u, f_v
        # return u, v, p, gammap, eta_star, S12, S11, S22
        

PINN = Sequentialmodel(layers)
       
PINN.to(device)

'Neural Network Summary'
print(PINN)

params = list(PINN.parameters())

######################################################################
######################## Optimization ################################
######################################################################

np.savetxt("Xtrain.dat", X_train)

start_time = time.time()

param = list(PINN.parameters())
'Adam Optimizer'
optimizer = optim.Adam(param, lr=0.0025)#, amsgrad=False)


epoch =800000

start_time = time.time()
saveoutputnn = epoch
w=PINN.weight(X_train).detach().item()
print('weight:',w)
for i in range(epoch):

    optimizer.zero_grad()

    loss, loss_u, loss_phy = PINN.loss_PDE(X_train,u_train, v_train,w)
    
    if i % 200 == 0:
         print('#########',i,'/','loss:',loss.item(),'/','a1',PINN.alpha_1.item(),'/','n',PINN.alpha_2.item(),'#########')
    
    # zeroes the gradient buffers of all parameters

    loss.backward()

    if i == 700_000:
        print('Save of the final output')
        u,v,p,gammap, eta =PINN.eval(XX/D, YY/D)
        loss_file = open(f"{mainpath}/Macro_final_save{nbr}.dat","w")
        for j in range(len(XX)):
                loss_file.write(
                            f'{XX[j]}'+" "+\
                            f'{YY[j]}'+" "+\
                            f'{p[j]}'+" "+\
                            f'{u[j]}'+" "+\
                            f'{v[j]}'+" "+\
                            f'{eta[j]}'+" "+\
                            f'{gammap[j]}'+"\n")
        loss_file.close()



