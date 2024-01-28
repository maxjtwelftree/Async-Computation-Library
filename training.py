import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split

Prices_Train = pd.read_csv("csv/Prices_Train_real.csv", header=None)
Strategy_Train = pd.read_csv("csv/Strategy_Train_real.csv", header=None)
Strikes_Train = pd.read_csv("csv/Strikes_Train_real.csv", header=None)

X = pd.concat([Prices_Train,Strikes_Train],axis = 1)
Y = Strategy_Train
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1, random_state=42)
X_train.reset_index(inplace=True, drop=True)

def XY_to_price_preprocessing(X,Y):
    h_plus = np.maximum(Y,0)
    h_minus = np.maximum(-Y,0)
    price =np.sum(np.array(h_plus)*np.array(X.iloc[:,:56]),axis = 1)-np.sum(np.array(h_minus)*np.array(X.iloc[:,56:112]),axis = 1)
    return price   

# Compute the prices of the strategies
Prices_train = XY_to_price_preprocessing(X_train,Y_train)
Prices_train[Prices_train == 0] = 1 # TO avoid division by zero

# Scale the prices such that either 0 or -1
Y_train = Y_train.div(np.abs(Prices_train),axis = 0)

def XY_to_price(X,Y):
    h_plus = tf.nn.relu(Y)
    h_minus = tf.nn.relu(-Y)
    summand_1 = tf.reduce_sum(h_plus*X.iloc[:,:56],axis = 1)
    summand_2 =-tf.reduce_sum(h_minus*X.iloc[:,56:112],axis = 1)
    return summand_1+summand_2

Y_prices_test = XY_to_price(X_test,Y_test)
Y_prices_train = XY_to_price(X_train,Y_train)

def generate_prices(Batch_size):
    X_sample = X_train.sample(Batch_size,replace = False)
    Y_sample = tf.cast(tf.reshape(tf.gather(Y_prices_train,X_sample.index),(Batch_size,1)),tf.float32)
    pi_plus= X_sample.iloc[:,1:56]
    pi_minus  = X_sample.iloc[:,57:112]
    K = tf.concat([np.zeros((Batch_size,5)),np.array(X_sample.iloc[:,112:])],axis = 1)
    yield pi_minus, pi_plus,K, Y_sample

def generate_sample_S(Batch_size,Batch_S):
    S = tf.random.uniform([Batch_size,n_assets,Batch_S],0,2)
    yield  S
    
def payoffs(S,K,Batch_size,Batch_S):    
    assets = S
    Calls1 = tf.nn.relu(tf.repeat(S[:,0,tf.newaxis],10,axis=1)-tf.repeat(K[:,5:15,tf.newaxis] ,Batch_S,axis = 2) )    
    Calls2 = tf.nn.relu(tf.repeat(S[:,1,tf.newaxis],10,axis=1)-tf.repeat(K[:,15:25,tf.newaxis] ,Batch_S,axis = 2) )
    Calls3 = tf.nn.relu(tf.repeat(S[:,2,tf.newaxis],10,axis=1)-tf.repeat(K[:,25:35,tf.newaxis] ,Batch_S,axis = 2) )
    Calls4 = tf.nn.relu(tf.repeat(S[:,3,tf.newaxis],10,axis=1)-tf.repeat(K[:,35:45,tf.newaxis] ,Batch_S,axis = 2))
    Calls5 = tf.nn.relu(tf.repeat(S[:,4,tf.newaxis],10,axis=1)-tf.repeat(K[:,45:55,tf.newaxis] ,Batch_S,axis = 2))
    return tf.concat([assets,Calls1,Calls2,Calls3,Calls4,Calls5],1)

class train_strategies:
    def __init__(self,
                 payoffs,
                     nr_payoffs,
                     gamma_start = 10,
                     gamma_end = 100,
                     depth = 5,
                     H_max = 1,
                     a_max = 1,
                     nr_neurons = 128,
                     Batch_size = 1024,
                     Batch_S = 32,
                     l_r = 0.001,
                     max_iter = 1000):
        # Initiliaze
        self.nr_payoffs = nr_payoffs
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.depth = depth
        self.H_max = H_max
        self.a_max = a_max
        self.nr_neurons = nr_neurons
        self.Batch_size = Batch_size
        self.Batch_S = Batch_S
        self.Batch_size = Batch_size
        self.l_r = l_r
        self.max_iter = max_iter

        # Create Optimizer and Model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = l_r, beta_1=0.99, beta_2=0.995)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = l_r)
        self.model = self.build_model()
        self.losses = []

    # Create Tensors for the Input    
    def build_model(self):
        K = keras.Input(shape=(self.nr_payoffs-1,),name = "K")
        pi_minus = keras.Input(shape=(self.nr_payoffs-1,),name = "pi_minus")
        pi_plus = keras.Input(shape=(self.nr_payoffs-1,),name = "pi_plus")
        combined = layers.concatenate([K, pi_minus, pi_plus])
        lay = layers.Dense(self.nr_neurons,activation = "tanh", dtype='float32')(combined)
        # Create deep layers
        for i in range(self.depth):
            lay = layers.Dense(self.nr_neurons,activation = "tanh")(lay) 
        # Output Layers
        a_out = self.a_max*layers.Dense(1,name = "a_out",activation = "tanh")(lay)
        h_minus_out = self.H_max*layers.Dense(self.nr_payoffs-1,name = "h_minus_out",activation = "sigmoid")(lay)
        h_plus_out = self.H_max*layers.Dense(self.nr_payoffs-1,name = "h_plus_out",activation = "sigmoid")(lay)
        model = keras.Model(inputs=[K,pi_minus,pi_plus],
                             outputs = [a_out,h_minus_out,h_plus_out])
        return model

    # Loss function
    def loss(self,model,K,pi_minus,pi_plus,S,epoch,Y):
        a, h_minus, h_plus = model({"K":K,"pi_minus":pi_minus,"pi_plus":pi_plus})
        f = a +tf.reshape(tf.reduce_sum(h_plus*pi_plus-h_minus*pi_minus,axis = 1),(self.Batch_size,1))
        a_expanded =tf.repeat(a,Batch_S,axis = 1)
        strat_expanded = payoffs(S,K,self.Batch_size,self.Batch_S)*tf.repeat((h_plus-h_minus)[:,:,tf.newaxis],self.Batch_S,axis = 2)
        I = a_expanded + tf.reduce_sum(strat_expanded,axis = 1 )
        loss = f+self.gamma(epoch)*tf.reshape(tf.reduce_mean(tf.nn.relu(-I)**2,axis =1),(self.Batch_size,1))
        return tf.reduce_mean(loss) +(self.gamma(epoch))*tf.reduce_mean((tf.nn.relu(-(Y+0.5)*f)))

    # Define Gradient    
    def grad(self,model,K,pi_minus,pi_plus,S,epoch,Y):
        with tf.GradientTape() as tape:
            loss_value = self.loss(model,K,pi_minus,pi_plus,S,epoch,Y)
        return loss_value, tape.gradient(loss_value,model.trainable_variables)

    def gamma(self,t):
        current_gamma = self.gamma_start*(self.max_iter-t)/self.max_iter+self.gamma_end*t/self.max_iter
        return current_gamma

    # Training Loop
    def train(self):
        for epoch in range(int(self.max_iter)):
            pi_minus, pi_plus,K,Y  = next(generate_prices(self.Batch_size))
            pi_minus = tf.cast(pi_minus, tf.float32)
            pi_plus = tf.cast(pi_plus, tf.float32)
            K = tf.cast(K, tf.float32)
            S = next(generate_sample_S(self.Batch_size,self.Batch_S))
            loss_value, grads = self.grad(self.model, K,pi_minus,pi_plus,S,epoch,Y)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            self.losses.append(loss_value.numpy())
            if epoch % 100 == 0 and epoch > 0:
                print("Iteration:{}, Avg. Loss: {}".format((epoch),np.mean(self.losses[-(round(epoch*0.05))])))     

        print("Iteration result: {}".format(np.mean(self.losses[-(round(self.max_iter*0.05))])))

n_assets = 5
nr_payoffs = 56 # Cash + 5 Stocks + 5*10 Calls
Batch_size = 4096
Batch_S =512
nr_neurons = 1024
gamma_start = 1
gamma_end = 10000
depth = 5
H_max = 1
a_max = 1
l_r = 0.0001
max_iter = 20000

strat = train_strategies(payoffs = payoffs,
                     nr_payoffs = nr_payoffs,
                     gamma_start = gamma_start,
                     gamma_end = gamma_end,
                     depth = depth,
                     H_max = H_max,
                     a_max =a_max,
                     nr_neurons = nr_neurons,
                     Batch_size = Batch_size,
                    Batch_S = Batch_S,
                     l_r = l_r,
                     max_iter = max_iter)

strat.train()