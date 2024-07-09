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

path_data='/home/mlardy2/Documents/work/Carreau/snaps/'
#path_data='/home/mlardy2/Documents/work/PINN/Pinn'
namedata='select'
data=f'Macro_{namedata}.dat'

#Set default dtype to float32
torch.set_default_dtype(torch.float)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

if device == 'cuda': 
    print(torch.cuda.get_device_name()) 

mainpath=f'/home/mlardy2/Documents/work/PINN/Pinn/output/carreau'   

#save the output in files
nbr=np.random.randint(0,1000)
lossfile=f'loss_right{nbr}'
lbdfile=f'lbd_right{nbr}'
nfile=f'nval_right{nbr}'
outputfile=f'output_right{nbr}'

file_exists = exists(f'{mainpath}/{lossfile}.dat')
while file_exists==True:
    print(nbr)
    nbr+=1
    lossfile=f'loss_right{nbr}'
    lbdfile=f'lbd_right{nbr}'
    nfile=f'nval_right{nbr}'
    outputfile=f'output_right{nbr}'
    file_exists = exists(f'{mainpath}/{lossfile}.dat')
print(nbr)
loss_file = open(f"{mainpath}/{lossfile}.dat",'w')
loss_file.close()
loss_file = open(f"{mainpath}/{lbdfile}.dat","w")
loss_file.close()
loss_file = open(f"{mainpath}/{nfile}.dat","w")
loss_file.close()



# Size of the NN
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

x = XX # This forms a rank-2 array with a single vector component, shape=(N,1)
y = YY
u = UU
v = VV
p = PP
s = SS
eta = EE


######################################################################
######################## Data Preprocessing ##########################
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
tau=0.01
ubar=np.mean(u_train)
vbar=np.mean(v_train)
u_train = u_train - ubar
v_train = v_train - vbar
max_u = max(np.max(abs(u_train)),np.max(abs(v_train)))
max_utot = max(np.max(abs(u-np.mean(u))),np.max(abs(v-np.mean(v))))
u_train = tau*u_train / max_u
v_train = tau*v_train / max_u

print("Velocity scaling factor C=", max_u*100.)

# Fixing D and U0 defining the Bingham number
# Arbitrary definition: with experimental data, we can use D and U0 from the data
D = 50.
U0 = 1.e-4

X_train = np.zeros((N_train,2))
for l in range(0, N_train) :
    X_train[l,0] = x_train[l]
    X_train[l,1] = y_train[l]
Xmin=X_train.min()
Xmax=X_train.max()
u_train=torch.from_numpy(u_train).to(device)
v_train=torch.from_numpy(v_train).to(device)

Dnn = (D) #/(Xmax )
U0nn = (U0)/max_u

gama_c_nn = ( U0nn/Dnn )
print(r"$\gamma_c_nn",(U0*0.01-ubar),(U0*1e-1))

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
            # Recommended gain value for tanh = 5/3?
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)
            
        self.alpha_1 = nn.Parameter(torch.ones([1], dtype=torch.float32)*0.99)
        self.alpha_2 = nn.Parameter(torch.ones([1], dtype=torch.float32)*0.99)
        self.alpha_3 = nn.Parameter(torch.ones([1], dtype=torch.float32)*0.99)


        self.nval = nn.Parameter(torch.ones([1], dtype=torch.float32)*0.75)

        
    'foward pass'
    def forward(self,x,y):
        
        
        x=torch.stack([x,y],axis=1)
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x).to(device)                

        #preprocessing input 
        x = (x - Xmin)/(Xmax - Xmin)+1e-16 #feature scaling
        #convert to float
        a = x.float()
        
        for i in range(len(layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
            
        a = self.linears[-1](a)
        return a
    
    def loss_PDE(self, X):
        
        x_train=X[:,0]
        y_train=X[:,1]
        
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2

        gama_c = gama_c_nn
        n=self.nval

        x = torch.from_numpy(x_train).to(device)
        y = torch.from_numpy(y_train).to(device)
        x.requires_grad = True
        y.requires_grad = True

        psi_and_p = self.forward(x,y)
        psi = psi_and_p[:,0:1].T[0]
        pstar = psi_and_p[:,1:2].T[0]
        u = autograd.grad(psi,y,torch.ones(x.shape).to(device), retain_graph=True, create_graph=True)[0]
        v = -autograd.grad(psi,x,torch.ones(x.shape).to(device), retain_graph=True, create_graph=True)[0]
        
        u_x = autograd.grad(u,x,torch.ones(x.shape).to(device), create_graph=True)[0]
        u_y = autograd.grad(u,y,torch.ones(x.shape).to(device), create_graph=True)[0]
    
        v_x = autograd.grad(v,x,torch.ones(x.shape).to(device), create_graph=True)[0]
        v_y = autograd.grad(v,y,torch.ones(x.shape).to(device), create_graph=True)[0]
           
        pstar_x = autograd.grad(pstar,x,torch.ones(x.shape).to(device), create_graph=True)[0]
        pstar_y = autograd.grad(pstar,y,torch.ones(x.shape).to(device), create_graph=True)[0]
        S11 = u_x
        S22 = v_y
        S12 = 0.5 * (u_y + v_x)

        gammapstar = (2.*(S11**2. + 2.*S12**2. + S22**2.))**(0.5)
        
        epsinf=torch.tensor(1.e-10)
        #gammapstar=torch.maximum(gammapstar,epsinf)
        gammapstar_mean = torch.mean(gammapstar)
        
        eta_star=alpha_1+(1-alpha_1)*(1+(1e6*gammapstar*gama_c)**2)**((n-1)/2) #1e-6
        #eta_star = alpha_1 * gammapstar**(-1.) + gama_c**(-n)*gammapstar**(n-1)
        #eta_star = eta_star / (gama_c**(-n))
        S11 = S11 #/ gammapstar_mean
        S22 = S22 #/ gammapstar_mean
        S12 = S12 #/ gammapstar_mean

        sig11 = 2. * eta_star * S11
        sig12 = 2. * eta_star * S12
        sig22 = 2. * eta_star * S22
        
        sig11_x = autograd.grad(sig11,x,torch.ones(x.shape).to(device),create_graph=True)[0]
        sig12_x = autograd.grad(sig12,x,torch.ones(x.shape).to(device),create_graph=True)[0]
        sig12_y = autograd.grad(sig12,y,torch.ones(x.shape).to(device),create_graph=True)[0]
        sig22_y = autograd.grad(sig22,y,torch.ones(x.shape).to(device),create_graph=True)[0] ##bug here for low nbr of points##
        
        eps = 1.e-6
        f_u = (- pstar_x* 0.0001+ sig11_x + sig12_y) / (eta_star * gammapstar)
        f_v = (- pstar_y* 0.0001 + sig12_x + sig22_y) / (eta_star * gammapstar)
        temp=(eta_star * gammapstar)
        
        loss_phy =  0.0001*(torch.sum(torch.square(f_u)) + torch.sum(torch.square(f_v))) #0.001
        loss_u=torch.sum(torch.square(u_train - u)) + torch.sum(torch.square(v_train - v))
        loss_pos=torch.maximum(torch.as_tensor(0),-alpha_1)+torch.maximum(torch.as_tensor(0),-n)+torch.maximum(torch.as_tensor(0),-(2-alpha_1))+torch.maximum(torch.as_tensor(0),-(2-n))
        
        
        loss_file = open(f"{mainpath}/{lossfile}.dat","a")
        loss_file.write(f'{loss_u:.3e}'+" "+\
                            f'{loss_phy:.3e}'+" "+\
                            f'{loss_phy+loss_u:.3e}'+"\n")
        loss_file.close()
        
        lbd_file = open(f"{mainpath}/{lbdfile}.dat","a") 
        lbd_file.write(f'{self.alpha_1.item()}'+" "+\
                       f'{self.alpha_2.item()}'+"\n") 
        lbd_file.close()
        lbd_file = open(f"{mainpath}/{nfile}.dat","a")
        lbd_file.write(f'{self.nval.item()}'+"\n")
        lbd_file.close()

        return loss_phy+loss_u+loss_pos, pstar_x*gama_c, (sig11_x + sig12_y)

    'callable for optimizer'                                       
    def closure(self):
        
        optimizer.zero_grad()
        
        loss = self.loss_PDE(X_train)
        
        loss.backward()
                
        self.iter += 1

        return loss        

        
        
PINN = Sequentialmodel(layers)
       
PINN.to(device)

'Neural Network Summary'
print(PINN)

params = list(PINN.parameters())

######################################################################
######################## Optimization ################################
######################################################################

start_time = time.time()

param = list(PINN.parameters())
'Adam Optimizer'
optimizer = optim.Adam(param, lr=0.001, amsgrad=False)


epoch =400000
eps=5e-6
start_time = time.time()

for i in range(epoch):

    optimizer.zero_grad()

    loss,p,sig = PINN.loss_PDE(X_train)
    
    if i % 200 == 0:
        print(torch.mean(p).item(),torch.mean(sig).item())
        print('#########',i,'/','loss:',loss.item(),'/','a1',PINN.alpha_1.item(),'a2',PINN.alpha_2.item(),'/','n',PINN.nval.item(),'#########')

    # zeroes the gradient buffers of all parameters


    loss.backward()
    optimizer.step()

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



