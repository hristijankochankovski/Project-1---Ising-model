import numpy as np
import scipy.sparse as sp
np.random.seed(12)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

L=40

# We will create 1000 Ising states
states=np.random.choice([-1, 1], size=(1000,L))
# Next, we define the ising energy
def ising_energies(states):
    L = states.shape[1]
    J = np.zeros((L, L),)
    for i in range(L): 
        J[i,(i+1)%L]=-1.0 
        
        E = np.einsum('...i,ij,...j->...',states,J,states)
        return E
    

energies=ising_energies(states)
states=np.einsum('...i,...j->...ij', states, states)
shape=states.shape
states=states.reshape((shape[0],shape[1]*shape[2]))
# build final data set
Data=[states,energies]

# define number of samples
n_samples= 800    



X_train=Data[0][:n_samples]
Y_train=Data[1][:n_samples] #+ np.random.normal(0,4.0,size=X_train.shape[0])
X_test=Data[0][n_samples:3*n_samples//2]
Y_test=Data[1][n_samples:3*n_samples//2] #+ np.random.normal(0,4.0,size=X_test.shape[0])

from sklearn import linear_model
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn
#% matplotlib inline

# set up Lasso and Ridge Regression models
leastsq=linear_model.LinearRegression()
ridge=linear_model.Ridge()
lasso = linear_model.Lasso()

# define error lists
train_errors_leastsq = []
test_errors_leastsq = []

train_errors_ridge = []
test_errors_ridge = []

train_errors_lasso = []
test_errors_lasso = []


# set regularisation strength values
lmbdas = np.logspace(-4, 5, 10)

#Initialize coeffficients for ridge regression and Lasso
coefs_leastsq = []
coefs_ridge = []
coefs_lasso=[]

n_boostraps = 100
Y_pred_ord = np.empty((Y_test.shape[0], n_boostraps))
Y_pred_lasso = np.empty((Y_test.shape[0], n_boostraps))
Y_pred_ridge = np.empty((Y_test.shape[0], n_boostraps))

for lmbda in lmbdas:
    for i in range(n_boostraps):
        x_, y_ = resample(X_train, Y_train)
        Y_pred_ord[:, i] = leastsq.fit(x_, y_).predict(X_test).ravel() 
        lasso.set_params(alpha=lmbda)
        Y_pred_lasso[:, i] = lasso.fit(x_,y_).predict(X_test).ravel()
        ridge.set_params(alpha=lmbda)
        Y_pred_ridge[:, i] = ridge.fit(x_,y_).predict(X_test).ravel()
    error_ord = np.mean( np.mean((Y_test.reshape(-1, 1) - Y_pred_ord)**2, axis=1, keepdims=True) )   
    train_errors_leastsq.append(error_ord)
    error_ridge = np.mean( np.mean((Y_test.reshape(-1, 1) - Y_pred_ridge)**2, axis=1, keepdims=True) )   
    train_errors_ridge.append(error_ridge)
    error_lasso = np.mean( np.mean((Y_test.reshape(-1, 1) - Y_pred_lasso)**2, axis=1, keepdims=True) )   
    train_errors_lasso.append(error_lasso)

plt.plot(train_errors_leastsq)    
plt.plot(train_errors_ridge)
plt.plot(train_errors_lasso)




