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

path_data='/home/mlardy2/Documents/work/simulation/snaps/'
#path_data='/home/mlardy2/Documents/work/PINN/Pinn'
namedata='lbd_0_75_n_1'
data=f'Macro_{namedata}.dat'

#Set default dtype to float32
torch.set_default_dtype(torch.float)

#PyTorch random number generator
#torch.manual_seed(1234)

# Random number generators in other libraries
#np.random.seed(1234)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

if device == 'cuda': 
    print(torch.cuda.get_device_name()) 

mainpath=f'/home/mlardy2/Documents/work/PINN/Pinn/output'   

nbr=np.random.randint(0,1000)
lossfile=f'loss_right{nbr}'
lbdfile=f'lbd_right{nbr}'
nfile=f'nval_right{nbr}'
outputfile=f'output_right{nbr}'

file_exists = exists(f'{mainpath}/{lossfile}.dat')
while file_exists==True:
    print(nbr)
    nbr+=1
    lossfile=f'loss_resample{nbr}'
    lbdfile=f'lbd_resample{nbr}'
    nfile=f'nval_resample{nbr}'
    outputfile=f'output_resample{nbr}'
    file_exists = exists(f'{mainpath}/{lossfile}.dat')
print(nbr)
loss_file = open(f"{mainpath}/{lossfile}.dat",'w')
loss_file.close()
loss_file = open(f"{mainpath}/{lbdfile}.dat","w")
loss_file.close()
loss_file = open(f"{mainpath}/{nfile}.dat","w")
loss_file.close()




N_train = 400
   
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

noise=np.random.normal(0,np.mean(UU)*0.1,len(UU))
x = XX# This forms a rank-2 array with a single vector component, shape=(N,1)
y = YY
u = UU
v = VV
p = PP
s = SS
eta = EE


u=u
v=v


######################################################################
######################## Noiseles Data ###############################
######################################################################
# Training Data
wavy=False
idx = np.random.choice(N, N_train, replace=False)
if wavy==True: #sample only inside walls
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
    leps=6
    for i in range(int(min(x)),int(max(x))):
        close=np.argmin(abs(i-wall_inf_x))
        temp=y[x==i]
        aa=np.logical_and(temp>wall_inf_y[close]+leps,temp<wall_sup_y[close]-leps)
        #c=np.logical_and(eta[x==i]<1e8,eta[x==i]>)
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
    N=len(ysorti)
    idx = np.random.choice(N, N_train, replace=False)

x_train = x[idx]
y_train = y[idx]
u_train = u[idx]
v_train = v[idx]


if wavy==True:
    x_train = xsorti[idx]
    y_train = ysorti[idx]
    u_train = usorti[idx]
    v_train = vsorti[idx]
    eta_train = etasorti[idx]
    s_train = sorti[idx]

# Normalization



ubar=np.mean(u_train)
vbar=np.mean(v_train)
u_train = u_train - ubar
v_train = v_train - vbar
max_u = max(np.max(abs(u_train)),np.max(abs(v_train)))
max_utot = max(np.max(abs(u-np.mean(u))),np.max(abs(v-np.mean(v))))
u_train = 0.01 * u_train / max_u
v_train = 0.01 * v_train / max_u

print("Velocity scaling factor C=", max_u*100.)

# Fixing D and U0 defining the Bingham number
# Arbitrary definition: with experimental data, we can use D and U0 from the data
D = 50.
U0 = 1.e-4

coeff_k = (D / (U0*0.01/max_u ))**-1
print("coef k",coeff_k)
X_train = np.zeros((N_train,2))
for l in range(0, N_train) :
    X_train[l,0] = x_train[l]
    X_train[l,1] = y_train[l]
lb=X_train.min()
ub=X_train.max()
u_train=torch.from_numpy(u_train).to(device)
v_train=torch.from_numpy(v_train).to(device)

class Sequentialmodel(nn.Module):
    
    def __init__(self,layers):
        super().__init__() #call __init__ from parent class 
              
        'activation function'
        self.activation = nn.Tanh()
    
        'Initialise neural network as a list using nn.Modulelist'  
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        
        self.iter = 0
        
        '''
        Alternatively:
        
        *all layers are callable 
    
        Simple linear Layers
        self.fc1 = nn.Linear(2,50)
        self.fc2 = nn.Linear(50,50)
        self.fc3 = nn.Linear(50,50)
        self.fc4 = nn.Linear(50,1)
        
        '''
    
        'Xavier Normal Initialization'
        # std = gain * sqrt(2/(input_dim+output_dim))
        for i in range(len(layers)-1):
            
            # weights from a normal distribution with 
            # Recommended gain value for tanh = 5/3?
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)
            
        self.lambda_1 = nn.Parameter(torch.ones([1], dtype=torch.float32)*0.99)
        self.nval = nn.Parameter(torch.ones([1], dtype=torch.float32)*0.75)

        
    'foward pass'
    def forward(self,x,y):
        
        
        x=torch.stack([x,y],axis=1)
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x).to(device)                
        
        # u_b = torch.from_numpy(ub).float().to(device)
        # l_b = torch.from_numpy(lb).float().to(device)
        u_b=ub
        l_b=lb
        #preprocessing input 
        x = (x - l_b)/(u_b - l_b)+1e-16 #feature scaling
        #convert to float
        a = x.float()
        '''     
        Alternatively:
        
        a = self.activation(self.fc1(a))
        a = self.activation(self.fc2(a))
        a = self.activation(self.fc3(a))
        a = self.fc4(a)
        
        '''
        
        for i in range(len(layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
            
        a = self.linears[-1](a)
        return a
    
    def loss_PDE(self, X):
        
        x_train=X[:,0]
        y_train=X[:,1]
        
        lambda_1 = self.lambda_1
        gama0 = coeff_k
        n=self.nval
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
        

        epsinf=torch.tensor(1.e-10)
        gammap=torch.maximum(gammap,epsinf)
        gammap_mean = torch.mean(gammap)
        
        eta = lambda_1 * (gama0**n)/(gama0*gammap) + (gama0*gammap)**(n-1)
        
        S11 = S11 / gammap_mean
        S22 = S22 / gammap_mean
        S12 = S12 / gammap_mean
            
        sig11 = 2. * eta * S11
        sig12 = 2. * eta * S12
        sig22 = 2. * eta * S22
        
        sig11_x = autograd.grad(sig11,x,torch.ones(x.shape).to(device),create_graph=True)[0]
        sig12_x = autograd.grad(sig12,x,torch.ones(x.shape).to(device),create_graph=True)[0]
        sig12_y = autograd.grad(sig12,y,torch.ones(x.shape).to(device),create_graph=True)[0]
        sig22_y = autograd.grad(sig22,y,torch.ones(x.shape).to(device),create_graph=True)[0] ##bug here for low nbr of points##
        
        eps = 1.e-6
        f_u = (- p_x + sig11_x + sig12_y) / (eta * gammap / gammap_mean )
        f_v = (- p_y  + sig12_x + sig22_y) / (eta * gammap / gammap_mean)
        temp=(eta * gammap / gammap_mean + eps)

        
        loss_phy =  0.001*(torch.sum(torch.square(f_u)) + torch.sum(torch.square(f_v))) #0.001
        loss_u=torch.sum(torch.square(u_train - u)) + torch.sum(torch.square(v_train - v))
        #loss_pos=torch.maximum(torch.as_tensor(0),-lambda_1)+torch.maximum(torch.as_tensor(0),-n)+torch.maximum(torch.as_tensor(0),-(2-lambda_1))+torch.maximum(torch.as_tensor(0),-(2-n))
        loss_file = open(f"{mainpath}/{lossfile}.dat","a")
        loss_file.write(f'{loss_u:.3e}'+" "+\
                            f'{loss_phy:.3e}'+" "+\
                            f'{loss_phy+loss_u:.3e}'+"\n")
        loss_file.close()
        
        lbd_file = open(f"{mainpath}/{lbdfile}.dat","a") 
        lbd_file.write(f'{self.lambda_1.item()}'+"\n") 
        lbd_file.close()
        lbd_file = open(f"{mainpath}/{nfile}.dat","a")
        lbd_file.write(f'{self.nval.item()}'+"\n")
        lbd_file.close()
        return loss_phy+loss_u, p/temp, (sig11_x + sig12_y)/temp

    'callable for optimizer'                                       
    def closure(self):
        
        optimizer.zero_grad()
        
        loss = self.loss_PDE(X_train)
        
        loss.backward()
                
        self.iter += 1

        return loss        

    def lasteval(self,x,y):

        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)
               
        x.requires_grad = True
        y.requires_grad = True
        
        psi_and_p = self.forward(x,y)
        
        psi = psi_and_p[:,0:1].T[0]
        p = psi_and_p[:,1:2].T[0]
        u = autograd.grad(psi,y,torch.ones(x.shape).to(device), retain_graph=True, create_graph=True)[0]
        v = -autograd.grad(psi,x,torch.ones(x.shape).to(device), retain_graph=True, create_graph=True)[0]
        
        return u.cpu().detach().numpy(),v.cpu().detach().numpy()


        
        
X_u_train = torch.from_numpy(X_train).float().to(device)

f_hat = torch.zeros(X_u_train.shape[0],1).to(device)

PINN = Sequentialmodel(layers)
       
PINN.to(device)

'Neural Network Summary'
print(PINN)

params = list(PINN.parameters())

'''Optimization'''

start_time = time.time()

param = list(PINN.parameters())
'Adam Optimizer'
optimizer = optim.Adam(param, lr=0.001, amsgrad=False)
U0 = data[:,[3,4]] # shape = (N,2)
P0 = data[:,2] / 3. # shape = (N)
X0 = data[:,[0,1]]
epoch =400000
eps=5e-6
epsvalue=1e-13
start_time = time.time()
lbdtemp=0
nvaltemp=0

wall = np.loadtxt("/home/mlardy2/Documents/work/simulation_wavy/simulation/snaps/Markers_on_live.dat")

for i in range(epoch):

    optimizer.zero_grad()

    loss,p,sig = PINN.loss_PDE(X_train)
    
    if i % 200 == 0:
        #print(torch.mean(p).item(),torch.mean(sig).item())
        #us,vs = PINN.lasteval(x_train,y_train)
        #utemp=(u_train.cpu() - us.cpu())**2
        #plt.scatter(xsorti,ysorti,c=sorti)
        #plt.scatter(wall[:,0],wall[:,1])
        #plt.scatter(x_train,y_train,c=(utemp.detach().numpy()))
        #plt.colorbar(cmap=utemp.detach().numpy())
        #plt.savefig(f"/home/mlardy2/Documents/work/PINN/Pinn/output/outputimage/image{i}.png")
        #plt.close()        
        print('#########',i,'/','loss:',loss.item(),'/','lbd',PINN.lambda_1.item(),'/','n',PINN.nval.item(),'#########')
        lbdtemp=abs(lbdtemp-PINN.lambda_1.item())
        nval=abs(nvaltemp-PINN.nval.item())
    # zeroes the gradient buffers of all parameters


    loss.backward()
    optimizer.step()


u,v=PINN.lasteval(x,y)

plt.tricontourf(x,y,u,200)
plt.colorbar(cmap=u)
plt.savefig("nonoise.png")
'''
print('#########',i,'/','loss:',loss.item(),'/','lbd',PINN.lambda_1.item(),'/','n','lbd',PINN.nval.item(),'#########')
elapsed = time.time() - start_time                
print('Training time: %.2f' % (elapsed))
print(PINN.lambda_1.item())
'L-BFGS Optimizer'

optimizer = torch.optim.LBFGS([PINN.lambda_1,PINN.nval],
                            max_iter = 250, 
                            max_eval = None, 
                            tolerance_grad = 1e-20, 
                            tolerance_change = 1e-22, 
                            history_size = 100, 
                            line_search_fn = 'strong_wolfe')

optimizer.step(PINN.closure)
print(PINN.lambda_1.item(),PINN.nval.item())
'''



