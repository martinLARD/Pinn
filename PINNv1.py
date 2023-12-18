import tensorflow as tf
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.optimize

N_train = 400
   
layers = [2, 20, 20, 20, 20, 2]

# Load Data
data = np.loadtxt("/home/mlardy2/Documents/work/PINN/Pinn/Macro_select_0_75.dat") # i, j, rho, u, v
#data = np.loadtxt("Macro_select.dat") # i, j, rho, u, v
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
# Training Data    
idx = np.random.choice(N, N_train, replace=False)

x_train = x[idx]
y_train = y[idx]
u_train = u[idx]
v_train = v[idx]

# Normalization

u_train = u_train - np.mean(u_train)
v_train = v_train - np.mean(v_train)
max_u = max(np.max(abs(u_train)),np.max(abs(v_train)))
u_train = 0.01 * u_train / max_u
v_train = 0.01 * v_train / max_u

print("Velocity scaling factor C=", max_u*100.)

# Fixing D and U0 defining the Bingham number
# Arbitrary definition: with experimental data, we can use D and U0 from the data
D = 50.
U0 = 1.e-4

coeff_k = D / (U0 * 0.01 / max_u)
Re=1.e-4
beta=1/U0


X_train = np.zeros((N_train,2))
for l in range(0, N_train) :
    X_train[l,0] = x_train[l]
    X_train[l,1] = y_train[l]
np.savetxt("x_train.dat", X_train)

lb=X_train.min()
ub=X_train.max()

X_train=tf.cast(X_train, dtype=tf.float32)
class Sequentialmodel(tf.Module): 
    def __init__(self, layers, name=None):
        lbd = tf.random.normal([1], dtype = 'float32') * 1 # sig0
        w = tf.Variable(lbd, trainable=True, name = 'lbd')

        self.lambda_1 = tf.Variable(tf.cast(tf.ones([1]), dtype = 'float32'), trainable = True,constraint=tf.keras.constraints.NonNeg())       
        self.nval = tf.Variable(tf.cast(tf.ones([1])*0.5, dtype = 'float32'), trainable = True,constraint=tf.keras.constraints.NonNeg())   
       

        self.W = []  #Weights and biases
        self.parameters = 0 #total number of parameters
        
        for i in range(len(layers)-1):
            
            input_dim = layers[i]
            output_dim = layers[i+1]
            
            #Xavier standard deviation 
            std_dv = np.sqrt((2.0/(input_dim + output_dim)))

            #weights = normal distribution * Xavier standard deviation + 0
            w = tf.random.normal([input_dim, output_dim], dtype = 'float32') * std_dv
                       
            w = tf.Variable(w, trainable=True, name = 'w' + str(i+1))

            b = tf.Variable(tf.cast(tf.zeros([output_dim]), dtype = 'float32'), trainable = True, name = 'b' + str(i+1))
                    
            
            self.W.append(w)
            self.W.append(b)
            
            self.parameters +=  input_dim * output_dim + output_dim
    
        # Lagrange multipliers
        
        # Boundary terms      
        
        # Residual terms
    @tf.function
    def evaluate(self,x,y):

        x=tf.stack([x,y],axis=1)
        x = (x-lb)/(ub-lb)
        

        i=0
        a=tf.cast(x, dtype=tf.float32)
        for i in range(len(layers)-2):
            
            z = tf.add(tf.matmul(a, self.W[2*i]), self.W[2*i+1])
            a = tf.math.tanh(z)
            
        a = tf.add(tf.matmul(a, self.W[-2]), self.W[-1]) # For regression, no activation to last layer
        return a
    
    def loss(self, X,step):
        
        x=X[:,0]
        y=X[:,1]
        lambda_1 = self.lambda_1
        lambda_2 = coeff_k
        n=self.nval
    
        

        with tf.GradientTape(persistent=True) as tape:

            tape.watch(x)
            tape.watch(y)
            psi_and_p= self.evaluate(x,y)
            psi = psi_and_p[:,0:1]
            p =  psi_and_p[:,1:2]
            u = tape.gradient(psi, y)
            v = -tape.gradient(psi, x)  
            
            u_x = tape.gradient(u, x)
            u_y = tape.gradient(u, y)
       
            v_x = tape.gradient(v, x)
            v_y = tape.gradient(v, y)
       
            p_x = tape.gradient(p, x)
            p_y = tape.gradient(p, y)
        
            S11 = u_x
            S22 = v_y
            S12 = 0.5 * (u_y + v_x)

            gammap = (2.*(S11**2. + 2.*S12**2. + S22**2.))**(0.5)
        
            gammap = tf.math.maximum(gammap, 1.e-14)
        
            gammap_mean = tf.math.reduce_mean(gammap)
            lambda_2=tf.cast(lambda_2, dtype=tf.float32)
            
            eta = lambda_1 * gammap**(-1.) + tf.math.pow(lambda_2,n)*tf.math.pow(gammap,n-1)

            eta = eta / (tf.math.pow(lambda_2,n))
            S11 = S11 / gammap_mean
            S22 = S22 / gammap_mean
            S12 = S12 / gammap_mean
            
            sig11 = 2. * eta * S11
            sig12 = 2. * eta * S12
            sig22 = 2. * eta * S22


        sig11_x = tape.gradient(sig11, x)
        sig12_x = tape.gradient(sig12, x)
        sig12_y = tape.gradient(sig12, y)
        sig22_y = tape.gradient(sig22, y)

        del tape

        eps = 1.e-6
        f_u = (- p_x + sig11_x + sig12_y) / (eta * gammap / gammap_mean + eps)
        f_v = (- p_y  + sig12_x + sig22_y) / (eta *gammap/ gammap_mean + eps)
        
        loss_phy = step * (tf.reduce_sum(tf.square(f_u)) + tf.reduce_sum(tf.square(f_v)))
        loss_data = tf.reduce_sum(tf.square(u_train - u)) + tf.reduce_sum(tf.square(v_train - v))
        losstot=loss_phy+loss_data
        output=[p,u,v,tf.math.reduce_mean(eta)]
        
        return losstot,loss_phy,loss_data, output
    
    @tf.function   
    def adaptive_gradients(self,step):
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.W)
            tape.watch(self.lambda_1)
            loss_val, loss_ph, loss_data,output = self.loss(X_train,step)

        grads = tape.gradient(loss_val,self.W)

        gradslbd = tape.gradient(loss_ph,self.lambda_1)
        gradsn = tape.gradient(loss_ph,self.nval)

        del tape
        return loss_val, grads, gradslbd,loss_ph,loss_data, output,gradsn
    
    ##############BFGS####################
    
    def set_weights(self,parameters):
                
        for i in range (len(layers)-1):

            shape_w = tf.shape(self.W[2*i]).numpy() # shape of the weight tensor
            size_w = tf.size(self.W[2*i]).numpy() #size of the weight tensor 
            
            shape_b = tf.shape(self.W[2*i+1]).numpy() # shape of the bias tensor
            size_b = tf.size(self.W[2*i+1]).numpy() #size of the bias tensor 
                        
            pick_w = parameters[0:size_w] #pick the weights 
            pick_w=tf.cast(pick_w,"float32")
            self.W[2*i].assign(tf.reshape(pick_w,shape_w)) # assign  
            parameters = np.delete(parameters,np.arange(size_w),0) #delete 
            
            pick_b = parameters[0:size_b] #pick the biases 
            pick_b=tf.cast(pick_b,"float32")

            self.W[2*i+1].assign(tf.reshape(pick_b,shape_b)) # assign 
            parameters = np.delete(parameters,np.arange(size_b),0) #delete 
    
    def get_weights(self):

        parameters_1d = []  # [.... W_i,b_i.....  ] 1d array
        
        for i in range (len(layers)-1):
            
            w_1d = tf.reshape(self.W[2*i],[-1])   #flatten weights 
            b_1d = tf.reshape(self.W[2*i+1],[-1]) #flatten biases
            
            parameters_1d = tf.concat([parameters_1d, w_1d], 0) #concat weights 
            parameters_1d = tf.concat([parameters_1d, b_1d], 0) #concat biases
        parameters_1d = tf.concat([parameters_1d, self.lambda_1], 0)
        parameters_1d = tf.concat([parameters_1d, self.nval], 0)
        return parameters_1d

    
    
    def optimizerfunc(self,parameters):
        
        self.set_weights(parameters[:-2])
        self.lambda_1[0].assign(parameters[-2])
        self.nval[0].assign(parameters[-1])
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.trainable_variables)
            tape.watch(self.lambda_1)
            tape.watch(self.nval)
            loss_val, loss_u, loss_f,output = self.loss(X_train,0.001)
            
        grads = tape.gradient(loss_val,self.trainable_variables)
        gradslbd = tape.gradient(loss_val,self.lambda_1)
        gradsn = tape.gradient(loss_val,self.nval)

                
        del tape
        
        grads_1d = [ ] #flatten grads 
        
        for i in range (len(layers)-1):

            
            grads_w_1d = tf.reshape(grads[2*i],[-1]) #flatten weights 
            grads_b_1d = tf.reshape(grads[2*i+1],[-1]) #flatten biases
            
            grads_w_1d=tf.cast(grads_w_1d,dtype="float64")
            grads_b_1d=tf.cast(grads_b_1d,dtype="float64")
            grads_1d = tf.concat([grads_1d, grads_w_1d], 0) #concat grad_weights 
            grads_1d = tf.concat([grads_1d, grads_b_1d], 0) #concat grad_biases
            
        gradslbd=tf.cast(gradslbd,dtype="float64")
        grads_1d=tf.concat([grads_1d, gradslbd], 0)
        
        gradsn=tf.cast(gradsn,dtype="float64")
        grads_1d=tf.concat([grads_1d, gradsn], 0)
        
        loss_file = open(f"output/{lossfile}.dat","a")
        loss_file.write(f'{loss_u:.3e}'+" "+\
                            f'{loss_f:.3e}'+" "+\
                            f'{loss_val:.3e}'+"\n")
        loss_file.close()
        
        lbd_file = open(f"output/{lbdfile}.dat","a") 
        lbd_file.write(f'{PINN.lambda_1.numpy()[0]}'+"\n") 
        lbd_file.close()
        lbd_file = open(f"output/{lbdfile}.dat","a") 
        lbd_file.write(f'{PINN.nval.numpy()[0]}'+"\n") 
        lbd_file.close()
        return loss_val.numpy(), grads_1d.numpy()

    
PINN = Sequentialmodel(layers)

start_time = time.time() 


num_epochs = 1

save=np.zeros(num_epochs)
savelbd=np.zeros(num_epochs)



lossfile=f'loss{N_train}bfgs3'
lbdfile=f'lbd{N_train}bfgs3'
nfile=f'lbd{N_train}bfgs3'
outputfile='output'
loss_file = open(f"output/{lossfile}.dat","w")
loss_file.close()
loss_file = open(f"output/{lbdfile}.dat","w")
loss_file.close()
loss_file = open(f"output/{outputfile}.dat","w")
loss_file.close()
loss_file = open(f"output/{nfile}.dat","w")
loss_file.close()

learn1=0.001
learn2=0.001
learn3=0.0001

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learn1   , epsilon=1e-07)
optimizer_lbd1 = tf.keras.optimizers.Adam(learning_rate=learn2,  epsilon=1e-07)
optimizer_n = tf.keras.optimizers.Adam(learning_rate=learn3,  epsilon=1e-07)

step=0.001

num_epochs=100000
eps=1e-6

print('§§§§§§§§§',"Ntrain:",N_train,'§§§§§§§§§')
for epoch in range(num_epochs):
        

        loss_value, grads, gradslbd,loss_u,loss_ph, output,gradsn= PINN.adaptive_gradients(step)
        if epoch % 1000 == 0:
            print('#########',epoch,'/','loss:',tf.get_static_value(loss_value),'/','lbd',PINN.lambda_1.numpy()[0],'/','n',PINN.nval.numpy()[0],'#########')
            
        loss_file = open(f"output/{lossfile}.dat","a")
        loss_file.write(f'{loss_u:.3e}'+" "+\
                            f'{loss_ph:.3e}'+" "+\
                            f'{loss_value:.3e}'+"\n")
        loss_file.close()
        lbd_file = open(f"output/{lbdfile}.dat","a") 
        lbd_file.write(f'{PINN.lambda_1.numpy()[0]}'+"\n") 
        lbd_file.close()
        lbd_file = open(f"output/{nfile}.dat","a") 
        lbd_file.write(f'{PINN.nval.numpy()[0]}'+"\n") 
        lbd_file.close()
        
        
        optimizer_lbd1.apply_gradients(zip([gradslbd], [PINN.lambda_1]))
        optimizer_n.apply_gradients(zip([gradsn], [PINN.nval]))
        for i in range((len(layers)-1)*2-1):
            optimizer.apply_gradients(zip([grads[i]], [PINN.W[i]]))
     #gradient descent weights 
        if loss_value<eps:
             break
init_params = PINN.get_weights().numpy()
print('#########',epoch,'/','loss:',tf.get_static_value(loss_value),'/','lbd',PINN.lambda_1.numpy()[0],'/','n',PINN.nval.numpy()[0],'#########')


print("launch bfgs")
#results=scipy.optimize.minimize(PINN.optimizerfunc, x0=init_params, method='L-BFGS-B', jac= True, options={'ftol':10**-7})
results=scipy.optimize.fmin_l_bfgs_b(PINN.optimizerfunc, x0=init_params,  factr=10e0)

bfgs_file=open("output/bfgs_result.dat","w")
bfgs_file.write(f"{results}")
bfgs_file.close()

elapsed = time.time() - start_time                
print('Training time: %.2f' % (elapsed))
print(PINN.lambda_1.numpy()[0],PINN.nval.numpy()[0])
