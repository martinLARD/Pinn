# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import time



class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x, y, u, v, layers, coeff_k):
        
        X = np.concatenate([x, y], 1) # shape = (Ntrain, 2)
        
        self.lb = X.min(0)
        self.ub = X.max(0)
                 
        self.X = X
        # 
        self.x = X[:,0:1]
        self.y = X[:,1:2]

        self.lb = x.min()
        self.ub = x.max()
        
        self.u = u
        self.v = v
        
        self.layers = layers
        
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)        
        
        # Initialize parameters
        self.lambda_1 = tf.Variable([0.0], dtype=tf.float32, constraint=tf.keras.constraints.NonNeg()) # sig0
        # self.lambda_1 = 0.5
        self.lambda_2 = coeff_k # k
        
        # tf placeholders and graph
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        # batch_size = 16
        self.x_tf = tf.keras.Input(dtype=tf.float32, shape=[self.x.shape[1]])
        self.y_tf = tf.keras.Input(dtype=tf.float32, shape=[self.y.shape[1]])
        self.u_tf = tf.keras.Input(dtype=tf.float32, shape=[self.u.shape[1]])
        self.v_tf = tf.keras.Input(dtype=tf.float32, shape=[self.v.shape[1]])
        
        self.u_pred, self.v_pred, self.p_pred, self.f_u_pred, self.f_v_pred, \
        self.gammap_pred, self.eta_pred, self.psi_pred = self.net_NS(self.x_tf, self.y_tf)
        
        self.loss_phy = 0.001 * (tf.reduce_sum(tf.square(self.f_u_pred)) + tf.reduce_sum(tf.square(self.f_v_pred)))
        self.loss_data = tf.reduce_sum(tf.square(self.u_tf - self.u_pred)) + tf.reduce_sum(tf.square(self.v_tf - self.v_pred))
        self.loss_reg = self.L2_reg(self.layers, self.weights, 1.e-9)
        self.loss = self.loss_phy + self.loss_data #+ self.loss_reg

        self.adam = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
        self.train_op_adam = self.adam.minimize(self.loss)
    
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

    def L2_reg(self, layers, weights, alpha):
        num_layers = len(layers)
        norm_weights = 0.
        for l in range(0, num_layers-1) :
            norm_weights += alpha * tf.reduce_sum(tf.square(weights[l]))
        return norm_weights

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
    
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    # def net_NS(self, x, y, t):
    def net_NS(self, x, y):
        lambda_1 = self.lambda_1 
        lambda_2 = self.lambda_2

        psi_and_p = self.neural_net(tf.concat([x,y], 1), self.weights, self.biases)
        psi = psi_and_p[:,0:1]
        p = psi_and_p[:,1:2]
        
        u = tf.gradients(psi, y)[0]
        v = -tf.gradients(psi, x)[0]  
        
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        
        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        
        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]

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

        sig11_x = tf.gradients(sig11, x)[0]
        sig12_x = tf.gradients(sig12, x)[0]
        sig12_y = tf.gradients(sig12, y)[0]
        sig22_y = tf.gradients(sig22, y)[0]

        eps = 1.e-6
        f_u = (- p_x + sig11_x + sig12_y) / (eta * gammap / gammap_mean + eps)
        f_v = (- p_y  + sig12_x + sig22_y) / (eta * gammap / gammap_mean + eps)

        return u, v, p, f_u, f_v, gammap, eta, psi
    

    def train(self, nIter): 

        tf_dict = {self.x_tf: self.x, self.y_tf: self.y,
                   self.u_tf: self.u, self.v_tf: self.v}
        
        loss_file = open("loss.dat","w")
        
        start_time = time.time()
        for it in range(nIter):

            self.sess.run(self.train_op_adam, tf_dict)
            
            # Print
            if it % 100 == 0:
                elapsed = time.time() - start_time
                loss_phy_value = self.sess.run(self.loss_phy, tf_dict)
                loss_data_value = self.sess.run(self.loss_data, tf_dict)
                #loss_reg_value = self.sess.run(self.loss_reg, tf_dict)
                lambda_1_value = self.sess.run(self.lambda_1)
                print('It: %d, Loss data: %.3e, Loss phy: %.3e, l1: %.3e, Time: %.2f' % 
                      (it, loss_data_value, loss_phy_value, lambda_1_value, elapsed))
                # print('It: %d, Loss data: %.3e, Loss phy: %.3e, Loss reg: %.3e, Time: %.2f' % 
                    #   (it, loss_data_value, loss_phy_value, loss_reg_value, elapsed))
                loss_file.write(f'{loss_data_value:.3e}'+" "+\
                                f'{loss_phy_value:.3e}'+" "+\
                                f'{lambda_1_value[0]:.3e}'+"\n")
                # loss_file.write(f'{loss_data_value:.3e}'+" "+\
                #                 f'{loss_phy_value:.3e}'+" "+\
                #                 f'{loss_reg_value:.3e}'+"\n")
                start_time = time.time()

            # save prediction
            if it % 1000 == 0:
                loss_file.flush()
                u_pred, v_pred, p_pred, gammap_pred, eta_pred , f_u_pred, f_v_pred, psi_pred = model.predict(x, y)
                data_predict = np.zeros((N,10)) # i, j, u, v, P, gammap, eta, f_u, f_v, psi
                for i in range(0, Nx) : 
                    for j in range(0, Ny) :
                        l = i * Ny + j
                        data_predict[l,0] = i
                        data_predict[l,1] = j
                        data_predict[l,2] = u_pred[l]
                        data_predict[l,3] = v_pred[l]
                        data_predict[l,4] = p_pred[l]
                        data_predict[l,5] = gammap_pred[l]
                        data_predict[l,6] = eta_pred[l]
                        data_predict[l,7] = f_u_pred[l]
                        data_predict[l,8] = f_v_pred[l]
                        data_predict[l,9] = psi_pred[l]
                np.savetxt("data_predict.dat", data_predict)

        loss_file.close()

                
    # def predict(self, x_star, y_star, t_star):
    def predict(self, x_star, y_star): #fais une run du NN pour avoir les valeurs finales
        
        tf_dict = {self.x_tf: x_star, self.y_tf: y_star}
        
        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        p_star = self.sess.run(self.p_pred, tf_dict)
        gammap_star = self.sess.run(self.gammap_pred, tf_dict)
        eta_star = self.sess.run(self.eta_pred, tf_dict)
        f_u = self.sess.run(self.f_u_pred, tf_dict)
        f_v = self.sess.run(self.f_v_pred, tf_dict)
        psi = self.sess.run(self.psi_pred, tf_dict)

        return u_star, v_star, p_star, gammap_star, eta_star, f_u, f_v, psi

N_train = 100
    
layers = [2, 16, 16, 16, 16, 2]

# Load Data
data = np.loadtxt("/home/mlardy/Documents/these/work/codes/Pinn/Macro_select.dat") # i, j, rho, u, v

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

x = XX[:,None] # This forms a rank-2 array with a single vector component, shape=(N,1)
y = YY[:,None]
u = UU[:,None]
v = VV[:,None]
p = PP[:,None]

######################################################################
######################## Noiseles Data ###############################
######################################################################
# Training Data    
idx = np.random.choice(N, N_train, replace=False)

x_train = x[idx,:]
y_train = y[idx,:]
u_train = u[idx,:]
v_train = v[idx,:]

# Normalization

u_train[:,0] = u_train[:,0] - np.mean(u_train[:,0])
v_train[:,0] = v_train[:,0] - np.mean(v_train[:,0])
max_u = max(np.max(abs(u_train[:,0])),np.max(abs(v_train[:,0])))
u_train[:,0] = 0.01 * u_train[:,0] / max_u
v_train[:,0] = 0.01 * v_train[:,0] / max_u

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
    X_train[l,0] = x_train[l][0]
    X_train[l,1] = y_train[l][0]
np.savetxt("x_train.dat", X_train)

# Training
model = PhysicsInformedNN(x_train, y_train, u_train, v_train, layers, coeff_k)
#model.train(10000)
model.train(250000)

u_pred, v_pred, p_pred, gammap_pred, eta_pred , f_u_pred, f_v_pred, psi_pred = model.predict(x, y)
data_predict = np.zeros((N,10)) # i, j, u, v, P, gammap, eta, f_u, f_v, psi
for i in range(0, Nx) : 
    for j in range(0, Ny) :
        l = i * Ny + j
        data_predict[l,0] = i
        data_predict[l,1] = j
        data_predict[l,2] = u_pred[l][0]
        data_predict[l,3] = v_pred[l][0]
        data_predict[l,4] = p_pred[l][0]
        data_predict[l,5] = gammap_pred[l][0]
        data_predict[l,6] = eta_pred[l][0]
        data_predict[l,7] = f_u_pred[l][0]
        data_predict[l,8] = f_v_pred[l][0]
        data_predict[l,9] = psi_pred[l][0]
np.savetxt("data_predict.dat", data_predict)

