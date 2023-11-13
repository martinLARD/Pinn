import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt


N_train = 10
   
layers = [2, 16, 16, 16, 16, 2]

# Load Data
data = np.loadtxt("Macro_select.dat") # i, j, rho, u, v

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

np.savetxt("test.dat", u_train)
np.savetxt("test2.dat", v_train)

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
    
    def evaluate(self,x,y):

        x=tf.stack([x,y],axis=1)
        x = (x-lb)/(ub-lb)
        
        a = x
        i=0
        a=tf.cast(a, dtype=tf.float32)
        for i in range(len(layers)-2):
            
            z = tf.add(tf.matmul(a, self.W[2*i]), self.W[2*i+1])
            a = tf.math.tanh(z)
            
        a = tf.add(tf.matmul(a, self.W[-2]), self.W[-1]) # For regression, no activation to last layer
        return a
    
    def get_weights(self):

        parameters_1d = []  # [.... W_i,b_i.....  ] 1d array
        
        for i in range (len(layers)-1):
            
            w_1d = tf.reshape(self.W[2*i],[-1])   #flatten weights 
            b_1d = tf.reshape(self.W[2*i+1],[-1]) #flatten biases
            
            parameters_1d = tf.concat([parameters_1d, w_1d], 0) #concat weights 
            parameters_1d = tf.concat([parameters_1d, b_1d], 0) #concat biases
        
        return parameters_1d
        
    def set_weights(self,parameters):
                
        for i in range (len(layers)-1):

            shape_w = tf.shape(self.W[2*i]).numpy() # shape of the weight tensor
            size_w = tf.size(self.W[2*i]).numpy() #size of the weight tensor 
            
            shape_b = tf.shape(self.W[2*i+1]).numpy() # shape of the bias tensor
            size_b = tf.size(self.W[2*i+1]).numpy() #size of the bias tensor 
                        
            pick_w = parameters[0:size_w] #pick the weights 
            self.W[2*i].assign(tf.reshape(pick_w,shape_w)) # assign  
            parameters = np.delete(parameters,np.arange(size_w),0) #delete 
            
            pick_b = parameters[0:size_b] #pick the biases 
            self.W[2*i+1].assign(tf.reshape(pick_b,shape_b)) # assign 
            parameters = np.delete(parameters,np.arange(size_b),0) #delete 
            
    def loss_data(self,u_pred,v_pred):
        
        loss_data = tf.reduce_sum(tf.square(u_train - u_pred)) + tf.reduce_sum(tf.square(v_train - v_pred))
        return loss_data
    
    def loss_PDE(self, x, y):
        lambda_1 = self.lambda_1
        lambda_2 = coeff_k
    
        psi_and_p = self.evaluate(x,y)
        psi = psi_and_p[:,0:1]
        p = psi_and_p[:,1:2]
        
        xtemp=tf.Variable(x, trainable=False)
        ytemp=tf.Variable(y, trainable=False)
        with tf.GradientTape(persistent=True) as tape:
    
            tape.watch(xtemp)
            tape.watch(ytemp)
            psi_and_p= self.evaluate(xtemp,ytemp)
            psi = psi_and_p[:,0:1]
            p =  psi_and_p[:,1:2]
            u = tape.gradient(psi, ytemp)
            v = -tape.gradient(psi, xtemp)  
            
            u_x = tape.gradient(u, xtemp)
            u_y = tape.gradient(u, ytemp)
       
            v_x = tape.gradient(v, xtemp)
            v_y = tape.gradient(v, ytemp)
       
            p_x = tape.gradient(p, xtemp)
            p_y = tape.gradient(p, ytemp)
        
            S11 = u_x
            S22 = v_y
            S12 = 0.5 * (u_y + v_x)
        
            gammap = (2.*(S11**2. + 2.*S12**2. + S22**2.))**(0.5)
        
            gammap = tf.math.maximum(gammap, 1.e-14)
        
            gammap_mean = tf.math.reduce_mean(gammap)
            
            eta = lambda_1 * gammap**(-1.) + lambda_2
        
            eta = eta / lambda_2
            S11 = S11 / gammap_mean
            S22 = S22 / gammap_mean
            S12 = S12 / gammap_mean
            
            sig11 = 2. * eta * S11
            sig12 = 2. * eta * S12
            sig22 = 2. * eta * S22
            
            sig11_x = tape.gradient(sig11, xtemp)
            sig12_x = tape.gradient(sig12, xtemp)
            sig12_y = tape.gradient(sig12, ytemp)
            sig22_y = tape.gradient(sig22, ytemp)
    
        del tape
        
        eps = 1.e-6
        f_u = (- p_x + sig11_x + sig12_y) / (eta * gammap / gammap_mean + eps)
        f_v = (- p_y  + sig12_x + sig22_y) / (eta * gammap / gammap_mean + eps)
        
        loss_phy = 0.001 * (tf.reduce_sum(tf.square(f_u)) + tf.reduce_sum(tf.square(f_v)))
    
    
        return loss_phy
        
    def loss(self,X,u,v):
        
        loss_u = self.loss_data(u,v)
        loss_ph = self.loss_PDE(X[:,0],X[:,1])

        loss = loss_u + loss_ph

        return loss, loss_u, loss_ph
    
    def optimizerfunc(self,parameters):
        
        self.set_weights(parameters)
       
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            loss_val, loss_u, loss_f = self.loss(X_train, u_train, v_train)
            
        grads = tape.gradient(loss_val,self.trainable_variables)
                
        del tape
        
        grads_1d = [ ] #flatten grads 
        
        for i in range (len(layers)-1):

            grads_w_1d = tf.reshape(grads[2*i],[-1]) #flatten weights 
            grads_b_1d = tf.reshape(grads[2*i+1],[-1]) #flatten biases

            grads_1d = tf.concat([grads_1d, grads_w_1d], 0) #concat grad_weights 
            grads_1d = tf.concat([grads_1d, grads_b_1d], 0) #concat grad_biases

        return loss_val.numpy(), grads_1d.numpy()
    
    def optimizer_callback(self,parameters):
        
        psi_and_p = self.evaluate(X_train)
        x=X_train[:,0]
        
        y=X_train[:,1]
        psi = psi_and_p[:,0:1]
        p = psi_and_p[:,1:2]
        
        xtemp=tf.Variable(x, trainable=False)
        ytemp=tf.Variable(y, trainable=False)
        with tf.GradientTape() as tape:
            tape.watch(xtemp)
            tape.watch(ytemp)
            u = tape.gradient(psi, y)[0]
            v = -tape.gradient(psi, x)[0] 
            
        del tape
        
        loss_value, loss_u, loss_f = self.loss(X_train, u, v)
        
        
        tf.print(loss_value, loss_u, loss_f)
        
    def adaptive_gradients(self):
        
        x=X_train[:,0]
        y=X_train[:,1]
        xtemp=tf.Variable(x, trainable=False)
        ytemp=tf.Variable(y, trainable=False)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(ytemp)
            tape.watch(xtemp)
            psi_and_p= self.evaluate(xtemp,ytemp)
            psi = psi_and_p[:,0:1]

        v = -tape.gradient(psi, xtemp)
        u = tape.gradient(psi, ytemp)
        del tape        
            
        

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.W)
            tape.watch(self.lambda_1)
            loss_val, loss_u, loss_f = self.loss(X_train, u, v)

        grads = tape.gradient(loss_val,self.W)
        gradslbd = tape.gradient(loss_val,self.lambda_1)

        del tape
        

        return loss_val, grads, gradslbd,loss_u,loss_f
    
    
PINN = Sequentialmodel(layers)

start_time = time.time() 

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
optimizer2 = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

optimizer_lbd1 = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

num_epochs = 2500

save=np.zeros(num_epochs)
savelbd=np.zeros(num_epochs)

for epoch in range(num_epochs):
        loss_value, grads, gradslbd,loss_u,loss_ph= PINN.adaptive_gradients()
        save[epoch]=tf.get_static_value(loss_value)
        

        if epoch % 50 == 0:
            print('#########',epoch,':',tf.get_static_value(loss_value),'#########')
            loss_file = open("loss.dat","a")
            loss_file.write(f'{loss_u:.3e}'+" "+\
                                f'{loss_ph:.3e}'+" "+\
                                f'{loss_value:.3e}'+"\n")
        optimizer_lbd1.apply_gradients(zip([gradslbd], [PINN.lambda_1]))
        savelbd[epoch]=tf.get_static_value(PINN.lambda_1)
        for i in range((len(layers)-1)*2-1):
            optimizer.apply_gradients(zip([grads[i]], [PINN.W[i]]))
        

     #gradient descent weights 
init_params = PINN.get_weights().numpy()




# train the model with Scipy L-BFGS optimizer

elapsed = time.time() - start_time                
print('Training time: %.2f' % (elapsed))

# print(results)

# PINN.set_weights(results.x)

# ''' Model Accuracy ''' 
# u_pred = PINN.evaluate(X_u_test)

# error_vec = np.linalg.norm((u-u_pred),2)/np.linalg.norm(u,2)        # Relative L2 Norm of the error (Vector)
# print('Test Error: %.5f'  % (error_vec))

# u_pred = np.reshape(u_pred,(256,100),order='F')                        # Fortran Style ,stacked column wise!

# ''' Solution Plot '''
# # solutionplot(u_pred,X_u_train,u_train)
