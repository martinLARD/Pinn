import torch
import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    
    

N_train = 400
   
layers = [2, 20, 20, 20, 20, 2]

# Load Data
data = np.loadtxt("Macro_select_0_75.dat") # i, j, rho, u, v

U = data[:,[3,4]] # shape = (N,2)
P = data[:,2] / 3. # shape = (N)
X = data[:,[0,1]] # shape = (N,2)

N = X.shape[0]
Nx = int(np.sqrt(N))
Ny = int(np.sqrt(N))

# Rearrange Data
XX = X[:,0]
YY = X[:,1]

UU = U[:,0]
VV = U[:,1]
PP = P[:]

x = XX# This forms a rank-2 array with a single vector component, shape=(N,1)
y = YY
u = UU
v = VV
p = PP

######################################################################
######################## Noiseles Data ###############################
######################################################################

u=torch.from_numpy(u)
v=torch.from_numpy(v)


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
            
        self.lambda_1 = nn.Parameter(torch.ones([1], dtype=torch.float32))

        
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
        x = (x - l_b)/(u_b - l_b) #feature scaling
        
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
    
    def loss_PDE(self, x_train,y_train,u_train,v_train):
        

        #u_train= torch.from_numpy(u_train).to(device)
            
        #v_train= torch.from_numpy(v_train).to(device)
        lambda_1 = self.lambda_1
        lambda_2 = coeff_k
    
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
        
        #gammap = torch.clamp(gammap, max=1.e-14)
        
        gammap_mean = torch.mean(gammap)
        
        eta = lambda_1 * gammap**(-1.) + lambda_2
        
        eta = eta / lambda_2
        S11 = S11 / gammap_mean
        S22 = S22 / gammap_mean
        S12 = S12 / gammap_mean
            
        sig11 = 2. * eta * S11
        sig12 = 2. * eta * S12
        sig22 = 2. * eta * S22
            
        sig11_x = autograd.grad(sig11,x,torch.ones(x.shape).to(device), create_graph=True)[0]
        sig12_x = autograd.grad(sig12,x,torch.ones(x.shape).to(device), create_graph=True)[0]
        sig12_y = autograd.grad(sig12,y,torch.ones(x.shape).to(device), create_graph=True)[0]
        sig22_y = autograd.grad(sig22,y,torch.ones(x.shape).to(device), create_graph=True)[0]
        
        
        
        eps = 1.e-6
        f_u = (- p_x + sig11_x + sig12_y) / (eta * gammap / gammap_mean + eps)
        f_v = (- p_y  + sig12_x + sig22_y) / (eta * gammap / gammap_mean + eps)
        
        loss_phy = 0.001 * (torch.sum(torch.square(f_u)) + torch.sum(torch.square(f_v)))
        loss_u=torch.sum(torch.square(u_train - u)) + torch.sum(torch.square(v_train - v))
        
        loss_file = open(f"output/{lossfile}.dat","a")
        loss_file.write(f'{loss_u:.3e}'+" "+\
                            f'{loss_phy:.3e}'+" "+\
                            f'{loss_phy+loss_u:.3e}'+"\n")
        loss_file.close()
        
        lbd_file = open(f"output/{lbdfile}.dat","a") 
        lbd_file.write(f'{self.lambda_1.item()}'+"\n") 
        lbd_file.close()
        
        return loss_phy+loss_u


    def loss_colloc(self, x_train,y_train,u_train,v_train):
            
            
            lambda_1 = self.lambda_1
            lambda_2 = coeff_k
            #u_train= torch.from_numpy(u_train).to(device)
            
            #v_train= torch.from_numpy(v_train).to(device)
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
            
            #gammap = torch.clamp(gammap, max=1.e-14)
            
            gammap_mean = torch.mean(gammap)
            
            eta = lambda_1 * gammap**(-1.) + lambda_2
            
            eta = eta / lambda_2
            S11 = S11 / gammap_mean
            S22 = S22 / gammap_mean
            S12 = S12 / gammap_mean
                
            sig11 = 2. * eta * S11
            sig12 = 2. * eta * S12
            sig22 = 2. * eta * S22
                
            sig11_x = autograd.grad(sig11,x,torch.ones(x.shape).to(device), create_graph=True)[0]
            sig12_x = autograd.grad(sig12,x,torch.ones(x.shape).to(device), create_graph=True)[0]
            sig12_y = autograd.grad(sig12,y,torch.ones(x.shape).to(device), create_graph=True)[0]
            sig22_y = autograd.grad(sig22,y,torch.ones(x.shape).to(device), create_graph=True)[0]
            
            
            
            eps = 1.e-6
            f_u = (- p_x + sig11_x + sig12_y) / (eta * gammap / gammap_mean + eps)
            f_v = (- p_y  + sig12_x + sig22_y) / (eta * gammap / gammap_mean + eps)

            loss_phy = 0.001 * (torch.square(f_u) + torch.square(f_v))

            
            
            return loss_phy
    'callable for optimizer'                                       
    def closure(self):
        
        optimizer.zero_grad()
        
        loss = self.loss_PDE(x_train,y_train,u_train,v_train)
        
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
        
        u=u*100*max_u+ubar
        v=v*100*max_u+vbar
        for i in range(len(x)):
            loss_file = open(f"output/{outputfile}.dat","a")
            loss_file.write(f'{x[i]:.3e}'+" "+\
                                f'{y[i]:.3e}'+" "+\
                                f'{p[i]:.3e}'+" "+\
                                f'{u[i]:.3e}'+" "+\
                                f'{v[i]:.3e}'+"\n")
            loss_file.close()
        
# Training Data    
idx = np.random.choice(N, N_train, replace=False)

x_train = x[idx]
y_train = y[idx]
u_train = u[idx]
v_train = v[idx]

# Normalization
u_train=u_train.numpy()
v_train=v_train.numpy()
u=u.numpy()
v=v.numpy()
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

coeff_k = D / (U0*0.01/max_utot )


X_train = np.zeros((N_train,2))
for l in range(0, N_train) :
    X_train[l,0] = x_train[l]
    X_train[l,1] = y_train[l]

lb=X_train.min()
ub=X_train.max()

u_train=torch.from_numpy(u_train).to(device)
v_train=torch.from_numpy(v_train).to(device)


u = torch.from_numpy(u).to(device)
v = torch.from_numpy(v).to(device)

PINN = Sequentialmodel(layers)
       
PINN.to(device)


lossfile=f'loss{N_train}pytorch2'
lbdfile=f'lbd{N_train}pytorch2'
outputfile=f'output{N_train}pytorch2'
loss_file = open(f"output/{lossfile}.dat","w")
loss_file.close()
loss_file = open(f"output/{lbdfile}.dat","w")
loss_file.close()
loss_file = open(f"output/{outputfile}.dat","w")
loss_file.close()
'Neural Network Summary'
print(PINN)

params = list(PINN.parameters())

'''Optimization'''

start_time = time.time()

param = list(PINN.parameters())
'Adam Optimizer'
optimizer = optim.Adam(param, lr=0.001, amsgrad=False)

epoch = 100000
eps=0.5e-5
start_time = time.time()


def lossS(x,y,u,v,S):
    '''
    RAR-D sampling

    Parameters
    ----------

    Returns
    -------
    sampling position

    '''
    k=1
    c=0
    q=np.zeros(len(x))
    q=PINN.loss_colloc(x,y,u,v)
    q=q.cpu()
    q=q.detach().numpy()
    prob=q**k/sum(q**k)+c
    return np.random.choice(len(q),S,p=prob, replace=False)

plt.scatter(x_train,y_train)
plt.show()
S=5
for i in range(epoch):

    optimizer.zero_grad()
    loss = PINN.loss_PDE(x_train,y_train,u_train,v_train)
    if i % 5000 ==0 and i>30000 and len(x_train)<1000:
        print(i)	
        idx2 = np.random.choice(N, 1000, replace=False)
        q=lossS(x[idx2],y[idx2],u[idx2],v[idx2],S)
        #idx2 = np.random.choice(N, N_train, replace=False)
        # x_train=x_train.numpy()
        # y_train=y_train.numpy()
        x_train = np.concatenate((x_train,x[idx2][q]))
        y_train = np.concatenate((y_train,y[idx2][q]))
        plt.scatter(x_train,y_train)
        plt.show()
        u_train = np.concatenate((u_train.cpu(),u.cpu()[idx2][q]))
        v_train = np.concatenate((v_train.cpu(),v.cpu()[idx2][q]))
        ubar=np.mean(u_train)
        vbar=np.mean(v_train)
        u_train = u_train - ubar
        v_train = v_train - vbar
        u=u.cpu().numpy()
        v=v.cpu().numpy()
        max_u = max(np.max(abs(u_train)),np.max(abs(v_train)))
        max_utot = max(np.max(abs(u-np.mean(u))),np.max(abs(v-np.mean(v))))
        u=torch.from_numpy(u).to(device)
        v=torch.from_numpy(v).to(device)

        u_train = 0.01 * u_train / max_u
        v_train = 0.01 * v_train / max_u
        u_train=torch.from_numpy(u_train).to(device)
        v_train=torch.from_numpy(v_train).to(device)
    if i % 1000 == 0:
                
        print('#########',i,'/','loss:',loss.item(),'/','lbd',PINN.lambda_1.item(),'#########')
                

    # zeroes the gradient buffers of all parameters


    loss.backward()
    optimizer.step()
    if loss.item()<eps:
             break
         
print('#########',i,'/','loss:',loss.item(),'/','lbd',PINN.lambda_1.item(),'#########')

elapsed = time.time() - start_time                
print('Training time: %.2f' % (elapsed))
print(PINN.lambda_1.item())
'L-BFGS Optimizer'
optimizer = torch.optim.LBFGS(PINN.parameters(),  
                               max_iter = 250, 
                               max_eval = None, 
                               tolerance_grad = 1e-20, 
                               tolerance_change = 1e-22, 
                               history_size = 100, 
                               line_search_fn = 'strong_wolfe')

optimizer.step(PINN.closure)
print(PINN.lambda_1.item())

PINN.lasteval(x,y)


''' Solution Plot '''
# solutionplot(u_pred,X_u_train.cpu().detach().numpy(),u_train.cpu().detach().numpy())

