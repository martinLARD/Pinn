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


#path_data='/home/mlardy2/Documents/work/Carreau/snaps/'
path_data = "/home/mlardy2/Documents/work/simulation/snaps/"
namedata='lbd_0_4_n_0_9'
data=f'Macro_{namedata}.dat'

#Set default dtype to float32
# torch.set_default_dtype(torch.float)
torch.set_default_dtype(torch.float32)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

if device == 'cuda': 
    print(torch.cuda.get_device_name()) 

mainpath=f'/home/mlardy2/Documents/work/PINN/Pinn/output/full_model'   

#save the output in files
nbr=np.random.randint(0,1000)

lossfileHB=f'loss_HB_right{nbr}'
lossfilecar=f'loss_car_right{nbr}'

alphafileHB=f'alpha_HB_right{nbr}'
alphafilecar=f'alpha_car_right{nbr}'



file_exists = exists(f'{mainpath}/{lossfileHB}.dat')
while file_exists==True:
    print(nbr)
    nbr+=1
    lossfileHB=f'loss_HB_right{nbr}'
    lossfilecar=f'loss_car_right{nbr}'

    alphafileHB=f'alpha_HB_right{nbr}'
    alphafilecar=f'alpha_car_right{nbr}'
    file_exists = exists(f'{mainpath}/{lossfileHB}.dat')

print(nbr)
loss_file = open(f"{mainpath}/{lossfileHB}.dat",'w')
loss_file.close()
loss_file = open(f"{mainpath}/{lossfilecar}.dat",'w')
loss_file.close()
loss_file = open(f"{mainpath}/{alphafileHB}.dat","w")
loss_file.close()
loss_file = open(f"{mainpath}/{alphafilecar}.dat","w")
loss_file.close()



# Size of the NN
N_train = 800

layers = [2, 20, 20, 20, 20, 2]

# Load Data
data = np.loadtxt(f"{path_data}{data}") # i, j, rho, u, v
#data= np.loadtxt('/home/mlardy2/Documents/work/PINN/Pinn/Macro_select_0_75.dat')


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
#u=u+np.random.normal(0,np.mean(u)*0.05,len(u))
plt.scatter(x,y,c=u)
plt.show()
######################################################################
######################## Data Preprocessing ##########################
######################################################################
# Training Data
wavy=False
idx = np.random.choice(N, N_train, replace=False)
if wavy==True: #sample only inside walls
    print(' /!\ WAVY /!\ ')
    wall = np.loadtxt("/home/mlardy2/Documents/work/simulation_wavy/simulation/snaps/Markers_on_live.dat")
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
    deb=55#int(min(x))
    fin=75#int(max(x))
    for i in range(deb,fin):
        close=np.argmin(abs(i-wall_inf_x))
        temp=y[x==i]
        #aa=np.logical_and(temp>=55,temp<=73)

        #aa=np.logical_and(temp>wall_inf_y[close]+leps,temp<wall_sup_y[close]-leps)
        aa=(i-62.4)**2+(temp-62.4)**2>28**2
        aa=np.logical_and(temp>0,temp<50)

        #c=np.logical_or(temp<wall_inf_y[close]+20,temp>wall_sup_y[close]-20)

        #c=np.logical_and(temp<1e8,eta[x==i]>)
        #b=np.logical_and(s[x==i]<3e-7,s[x==i]>0.7e-7)
        #aa=np.logical_and(a,eta[x==i]<1e5)
        #aa=np.logical_and(ab,c)
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
        proba=(stest**(1)/sum(stest**(1)))#/sum(stest/sum(stest))#/sum(1-stest**2/sum(stest**2))
    print(len(xsorti))
    N=len(ysorti)
    idx = np.random.choice(N, N_train, replace=False)


x_train = x[idx]
y_train = y[idx]
u_train = u[idx]
v_train = v[idx]
s_train = s[idx]
eta_train = eta[idx]
if wavy==True:
    x_train = xsorti[idx]
    y_train = ysorti[idx]
    u_train = usorti[idx]
    v_train = vsorti[idx]
    eta_train = etasorti[idx]
    s_train = sorti[idx]

plt.scatter(x_train,y_train,c=s_train)
if wavy==True:
   plt.scatter(wall[:,0],wall[:,1])
plt.show()

print(max(s_train))
plt.scatter(s_train,eta_train)
#plt.hist(s_train,color='red',density=True)
plt.show()
# Normalization
tau=1.
# tau=1.


uM=min(u_train)
vM=min(v_train)
#max_u = max(np.max(abs(u_train-ubar)),np.max(abs(v_train-vbar)))

#u_train = (u_train +2*abs(uM))#*1e2
#v_train = (v_train + 2*abs(vM))#*1e2

ubar=np.mean(u_train)
vbar=np.mean(v_train)
# Fixing D and U0 defining the Bingham number
# Arbitrary definition: with experimental data, we can use D and U0 from the data
D = 50#np.mean(x_train)
U0 = 1e-4#/50#vbar*10**1

alphareval=1e-4/U0
u_train = (u_train-ubar) / (U0)#(psi*u_train-ubar) / U0
v_train = (v_train-vbar) / (U0)#(psi*v_train-vbar) / U0
print(np.mean(u_train))

print(min(u_train),max(u_train))

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

# gama_c = ( U0nn/Dnn )
gama_c = 1.
print(r"$\gamma_c_nn",U0/D,alphareval)

# plt.scatter(x,y,c=u)
# plt.show()
######################################################################
######################## Neural Network###############################
######################################################################


class HB(nn.Module):
    
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
            # Recommended gain value for tanh = 5/3?
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)
            
        self.alpha_1_HB = nn.Parameter(torch.ones([1], dtype=torch.float32)*0.2)

        self.alpha_2_HB = nn.Parameter(torch.ones([1], dtype=torch.float32)*0.8)  
    'foward pass'
    def forward_HB(self,x,y):
        
        
        X=torch.stack([x,y],axis=1)
        if torch.is_tensor(x) != True:         
            X = torch.tensor(X).to(device)                

        #preprocessing input 
        X_normed = (X - Xmin)/(Xmax - Xmin)+1e-16 #feature scaling
        #convert to float
        a = X_normed.float()
        
        for i in range(len(layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
            
        a = self.linears[-1](a)
        return a
    
    def weight_HB(self, X, u_train, v_train):

        x_train=X[:,0]
        y_train=X[:,1]
        
        alpha_1 = torch.abs(self.alpha_1_HB)
        n=torch.abs(self.alpha_2_HB)

        x = torch.from_numpy(x_train).to(device)
        y = torch.from_numpy(y_train).to(device)
        
        x.requires_grad = True
        y.requires_grad = True
        psi_and_p = self.forward_HB(x,y)
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

        gammap = (2.*(S11**2. + 2.*S12**2. + S22**2.))**(0.5)#/gama_c


        eta_star = alpha_1*(gammap/gama_c)**(-1.) + (gammap/gama_c)**(n-1)
        
        S11 = S11#/gama_c #/ gammapstar_mean
        S22 = S22#/gama_c #/ gammapstar_mean
        S12 = S12#/gama_c #/ gammapstar_mean

        sig11 = 2. * eta_star * S11 / gama_c #- S11 / torch.abs(S11) * torch.abs(alpha_1)
        sig12 = 2. * eta_star * S12 / gama_c #- S12 / torch.abs(S12) * torch.abs(alpha_1)
        sig22 = 2. * eta_star * S22 / gama_c #- S22 / torch.abs(S22) * torch.abs(alpha_1)
        
        sig11_x = autograd.grad(sig11,x,torch.ones(x.shape).to(device),create_graph=True)[0]
        sig12_x = autograd.grad(sig12,x,torch.ones(x.shape).to(device),create_graph=True)[0]
        sig12_y = autograd.grad(sig12,y,torch.ones(x.shape).to(device),create_graph=True)[0]
        sig22_y = autograd.grad(sig22,y,torch.ones(x.shape).to(device),create_graph=True)[0] ##bug here for low nbr of points##

        kappa = 1*(U0/D) * (U0/D)**(-n)

        f_u = (- p_x * kappa + sig11_x + sig12_y) #/ kappa
        f_v = (- p_y * kappa + sig12_x + sig22_y) #/ kappa #0.0001
         
        
        loss_phy =  (torch.sum(torch.square(f_u)) + torch.sum(torch.square(f_v))) #0.001
        loss_u=torch.sum(torch.square(u_train - u)) + torch.sum(torch.square(v_train - v))
        w=torch.floor(torch.log10(loss_u/loss_phy))
        #print(loss_phy,loss_u)
        return w

    def loss_PDE_HB(self, X, u_train, v_train, w):
        
        x_train=X[:,0]
        y_train=X[:,1]
        
        alpha_1 = self.alpha_1_HB
        n=self.alpha_2_HB
        # alpha_2 = self.alpha_2
        # alpha_1 = torch.as_tensor(0.4)
        # n = torch.as_tensor(0.9)

        x = torch.from_numpy(x_train).to(device)
        y = torch.from_numpy(y_train).to(device)
        x.requires_grad = True
        y.requires_grad = True
 
        psi_and_p = self.forward_HB(x,y)
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

        gammap = (2.*(S11**2. + 2.*S12**2. + S22**2.))**(0.5)#/gama_c


        eta_star = (alpha_1*(gammap/gama_c)**(-1.) + (gammap/gama_c)**(n-1))
        S11 = S11#/gama_c #/ gammapstar_mean
        S22 = S22#/gama_c #/ gammapstar_mean
        S12 = S12#/gama_c #/ gammapstar_mean

        sig11 = 2. * eta_star * S11 / gama_c #- S11 / torch.abs(S11) * torch.abs(alpha_1)
        sig12 = 2. * eta_star * S12 / gama_c #- S12 / torch.abs(S12) * torch.abs(alpha_1)
        sig22 = 2. * eta_star * S22 / gama_c #- S22 / torch.abs(S22) * torch.abs(alpha_1)
        
        sig11_x = autograd.grad(sig11,x,torch.ones(x.shape).to(device),create_graph=True)[0]
        sig12_x = autograd.grad(sig12,x,torch.ones(x.shape).to(device),create_graph=True)[0]
        sig12_y = autograd.grad(sig12,y,torch.ones(x.shape).to(device),create_graph=True)[0]
        sig22_y = autograd.grad(sig22,y,torch.ones(x.shape).to(device),create_graph=True)[0] ##bug here for low nbr of points##

        kappa = 1*(U0/D)**(-n)*(U0/D)# * (U0/D)**(-n)

        f_u = (- p_x * kappa + sig11_x + sig12_y) #/ kappa
        f_v = (- p_y * kappa + sig12_x + sig22_y) #/ kappa #0.0001

        loss_phy =  10**(w-1)*(torch.sum(torch.square(f_u)) + torch.sum(torch.square(f_v))) #0.001
        loss_u=torch.sum(torch.square(u_train - u)) + torch.sum(torch.square(v_train - v))
        loss_pos=10*(torch.maximum(torch.as_tensor(0),(-alpha_1))+torch.maximum(torch.as_tensor(0),-n))# +torch.maximum(torch.as_tensor(0),-(1.5-n)))
        
        loss_file = open(f"{mainpath}/{lossfileHB}.dat","a")
        loss_file.write(f'{loss_u:.3e}'+" "+\
                            f'{loss_phy:.3e}'+" "+\
                            f'{loss_phy+loss_u:.3e}'+"\n")
                            # f'{loss_phy+loss_u:.3e}'+"\n")
        loss_file.close()
        
        lbd_file = open(f"{mainpath}/{alphafileHB}.dat","a") 
        lbd_file.write(f'{self.alpha_1_HB.item()}'+" "+\
                        f'{self.alpha_2_HB.item()}'+"\n") 


        # return loss_phy+loss_u+loss_pos, p_x , (sig11_x + sig12_y)
        # return loss_phy+loss_u, p_x , (sig11_x + sig12_y)
        return loss_u+loss_phy+loss_pos, torch.min(gammap), torch.max(gammap)


class Carreau(nn.Module):
    
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
            # Recommended gain value for tanh = 5/3?
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)
        
        self.alpha_1_car = nn.Parameter(torch.ones([1], dtype=torch.float32)*0.2)

        self.alpha_2_car = nn.Parameter(torch.ones([1], dtype=torch.float32)*0.5)

        self.alpha_3_car = nn.Parameter(torch.ones([1], dtype=torch.float32)*1)     
    'foward pass'
    def forward_car(self,x,y):
    
    
        X=torch.stack([x,y],axis=1)
        if torch.is_tensor(x) != True:         
            X = torch.tensor(X).to(device)                

        #preprocessing input 
        X_normed = (X - Xmin)/(Xmax - Xmin)+1e-16 #feature scaling
        #convert to float
        a = X_normed.float()
        
        for i in range(len(layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
            
        a = self.linears[-1](a)
        return a
           
    def weight_car(self, X, u_train, v_train):

        x_train=X[:,0]
        y_train=X[:,1]
        
        alpha_1 = torch.abs(self.alpha_1_car)
        n=torch.abs(self.alpha_2_car)
        alpha_2 = self.alpha_3_car

        x = torch.from_numpy(x_train).to(device)
        y = torch.from_numpy(y_train).to(device)
        
        x.requires_grad = True
        y.requires_grad = True
        psi_and_p = self.forward_car(x,y)
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

        gammap = (2.*(S11**2. + 2.*S12**2. + S22**2.))**(0.5)#/gama_c


        eta_star=alpha_1+(1-alpha_1)*(1+alpha_2*(gammap/(gama_c))**2)**((n-1)/2) #1e-6
        
        S11 = S11#/gama_c #/ gammapstar_mean
        S22 = S22#/gama_c #/ gammapstar_mean
        S12 = S12#/gama_c #/ gammapstar_mean

        sig11 = 2. * eta_star * S11 / gama_c #- S11 / torch.abs(S11) * torch.abs(alpha_1)
        sig12 = 2. * eta_star * S12 / gama_c #- S12 / torch.abs(S12) * torch.abs(alpha_1)
        sig22 = 2. * eta_star * S22 / gama_c #- S22 / torch.abs(S22) * torch.abs(alpha_1)
        
        sig11_x = autograd.grad(sig11,x,torch.ones(x.shape).to(device),create_graph=True)[0]
        sig12_x = autograd.grad(sig12,x,torch.ones(x.shape).to(device),create_graph=True)[0]
        sig12_y = autograd.grad(sig12,y,torch.ones(x.shape).to(device),create_graph=True)[0]
        sig22_y = autograd.grad(sig22,y,torch.ones(x.shape).to(device),create_graph=True)[0] ##bug here for low nbr of points##

        kappa = 1# * (U0/D)**(-n)

        f_u = (- p_x * kappa + sig11_x + sig12_y) #/ kappa
        f_v = (- p_y * kappa + sig12_x + sig22_y) #/ kappa #0.0001
         
        
        loss_phy =  (torch.sum(torch.square(f_u)) + torch.sum(torch.square(f_v))) #0.001
        loss_u=torch.sum(torch.square(u_train - u)) + torch.sum(torch.square(v_train - v))
        
        w=torch.floor(torch.log10(loss_u/loss_phy))
        print(loss_phy,loss_u)
        return w

    def loss_PDE_car(self, X,u_train, v_train, w):
        
        x_train=X[:,0]
        y_train=X[:,1]
        
        alpha_1 = self.alpha_1_car
        n=self.alpha_2_car
        alpha_2 = self.alpha_3_car


        x = torch.from_numpy(x_train).to(device)
        y = torch.from_numpy(y_train).to(device)
        x.requires_grad = True
        y.requires_grad = True
 
        psi_and_p = self.forward_car(x,y)
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

        gammap = (2.*(S11**2. + 2.*S12**2. + S22**2.))**(0.5)#/gama_c


        eta_star=alpha_1+(1-alpha_1)*(1+alpha_2*(gammap/(gama_c))**2)**((n-1)/2) #1e-6
        
        S11 = S11#/gama_c #/ gammapstar_mean
        S22 = S22#/gama_c #/ gammapstar_mean
        S12 = S12#/gama_c #/ gammapstar_mean

        sig11 = 2. * eta_star * S11 / gama_c #- S11 / torch.abs(S11) * torch.abs(alpha_1)
        sig12 = 2. * eta_star * S12 / gama_c #- S12 / torch.abs(S12) * torch.abs(alpha_1)
        sig22 = 2. * eta_star * S22 / gama_c #- S22 / torch.abs(S22) * torch.abs(alpha_1)
        
        sig11_x = autograd.grad(sig11,x,torch.ones(x.shape).to(device),create_graph=True)[0]
        sig12_x = autograd.grad(sig12,x,torch.ones(x.shape).to(device),create_graph=True)[0]
        sig12_y = autograd.grad(sig12,y,torch.ones(x.shape).to(device),create_graph=True)[0]
        sig22_y = autograd.grad(sig22,y,torch.ones(x.shape).to(device),create_graph=True)[0] ##bug here for low nbr of points##

        kappa = 1#(U0/D)#**(-n)*(U0/D)# * (U0/D)**(-n)

        f_u = (- p_x * kappa + sig11_x + sig12_y) #/ kappa
        f_v = (- p_y * kappa + sig12_x + sig22_y) #/ kappa #0.0001
        
        loss_phy =  10**(w-1)*(torch.sum(torch.square(f_u)) + torch.sum(torch.square(f_v))) #0.001
        loss_u=torch.sum(torch.square(u_train - u)) + torch.sum(torch.square(v_train - v))
        loss_pos=1000*(torch.maximum(torch.as_tensor(0),(0.005-alpha_2))+torch.maximum(torch.as_tensor(0),(0.005-alpha_1))+torch.maximum(torch.as_tensor(0),-n) )#+torch.maximum(torch.as_tensor(0),-(2-alpha_1)) +torch.maximum(torch.as_tensor(0),-(2-n)))
        
        loss_file = open(f"{mainpath}/{lossfilecar}.dat","a")
        loss_file.write(f'{loss_u:.3e}'+" "+\
                            f'{loss_phy:.3e}'+" "+\
                            f'{loss_phy+loss_u:.3e}'+"\n")
                            # f'{loss_phy+loss_u:.3e}'+"\n")
        loss_file.close()
        
        lbd_file = open(f"{mainpath}/{alphafilecar}.dat","a") 
        lbd_file.write(f'{self.alpha_1_car.item()}'+" "+\
                       f'{self.alpha_2_car.item()}'+" "+\
                        f'{self.alpha_3_car.item()}'+"\n") 
        lbd_file.close()

        # return loss_phy+loss_u+loss_pos, p_x , (sig11_x + sig12_y)
        # return loss_phy+loss_u, p_x , (sig11_x + sig12_y)
        return loss_u+loss_phy+loss_pos, torch.min(gammap),torch.max(gammap)

        

PINN_HB = HB(layers)
       
PINN_Car = Carreau(layers)

PINN_HB.to(device)

'Neural Network Summary'
print(PINN_HB)

######################################################################
######################## Optimization ################################
######################################################################


start_time = time.time()

param_HB = list(PINN_HB.parameters())
param_Car = list(PINN_Car.parameters())

'Adam Optimizer'
optimizer_HB = optim.Adam(param_HB, lr=0.0025)#, amsgrad=False)
optimizer_carreau = optim.Adam(param_Car, lr=0.0025)#, amsgrad=False)


epoch =800000
# epoch = 1
eps=5e-6
start_time = time.time()
w_HB=PINN_HB.weight_HB(X_train,u_train, v_train).detach().item()
w_car=PINN_Car.weight_car(X_train,u_train, v_train).detach().item()

loss_car, gradp, sig = PINN_Car.loss_PDE_car(X_train,u_train, v_train,w_car)
loss_HB, gradp, sig = PINN_HB.loss_PDE_HB(X_train,u_train, v_train,w_HB)

rapport_loss=loss_HB/loss_car
print(rapport_loss)
for i in range(epoch):
    if i<20_000:
        optimizer_HB.zero_grad()
        loss_HB, gradp, sig = PINN_HB.loss_PDE_HB(X_train,u_train, v_train,w_HB)
        if i % 200 == 0:
            print('HB')
            print('#########',i,'/','loss:',loss_HB.item(),'/','a1',PINN_HB.alpha_1_HB.item(),'/','a2',PINN_HB.alpha_2_HB.item(),'#########')
        # zeroes the gradient buffers of all parameters
        loss_HB.backward()
        optimizer_HB.step()
        ##############################
        optimizer_carreau.zero_grad()
        loss_car, gradp, sig = PINN_Car.loss_PDE_car(X_train,u_train, v_train,w_car)
        if i % 200 == 0:
            print('Carreau')
            print('#########',i,'/','loss:',loss_car.item(),'/','a1',PINN_Car.alpha_1_car.item(),'/','a2',PINN_Car.alpha_2_car.item(),'/','a3',PINN_Car.alpha_3_car.item(),'#########')
        # zeroes the gradient buffers of all parameters
        loss_car.backward()
        optimizer_carreau.step()

        select=rapport_loss*loss_car/loss_HB
        if i % 200 ==0:
            print(select.item())

    else:
        if select<1:
            optimizer_carreau.zero_grad()
            loss_car, gradp, sig = PINN_Car.loss_PDE_car(X_train,u_train, v_train,w_car)
            if i % 200 == 0:
                print('Carreau')
                print('#########',i,'/','loss:',loss_car.item(),'/','a1',PINN_Car.alpha_1_car.item(),'/','a2',PINN_Car.alpha_2_car.item(),'/','a3',PINN_Car.alpha_3_car.item(),'#########')
            # zeroes the gradient buffers of all parameters
                loss_car.backward()
                optimizer_carreau.step()

        else:
            optimizer_HB.zero_grad()
            loss_HB, gradp, sig = PINN_HB.loss_PDE_HB(X_train,u_train, v_train,w_HB)
            if i % 200 == 0:
                print('HB')
                print('#########',i,'/','loss:',loss_HB.item(),'/','a1',PINN_HB.alpha_1_HB.item(),'/','a2',PINN_HB.alpha_2_HB.item(),'#########')
            # zeroes the gradient buffers of all parameters
            loss_HB.backward()
            optimizer_HB.step()
            


    

 