import numpy as np
import scipy.sparse as sp
np.random.seed(10000)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import sklearn.linear_model as skl
from IPython.display import display
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model

L=40
#Definig the statex
states=np.random.choice([-1, 1], size=(10000,L))
# Defining the function for calculating the ising energy
def ising_energies(states):
    L = states.shape[1]
    J = np.zeros((L, L),)
    for i in range(L): 
        J[i,(i+1)%L]=-1.0 
# only the interactions between the nearest neighbours i and i+1 are
# taken into account        
        
    E = np.einsum('...i,ij,...j->...',states,J,states)

    return E

# calculating the Ising energies
energies=ising_energies(states)


# reshaping of the Ising states
states=np.einsum('...i,...j->...ij', states, states)
#einsum computes the einsten sum of the two matricies
shape=states.shape
states=states.reshape((shape[0],shape[1]*shape[2]))
# build final data set
Data=[states,energies]

# define number of samples
n_samples = 400
# define train and test data sets


X_train, X_test, Y_train, Y_test = train_test_split(Data[0][:n_samples],Data[1][:n_samples], test_size = 0.2)


#Defining the number of boostraps cicles
n_boostraps = 400

#Defining the three regressions we are going to use
leastsq=linear_model.LinearRegression()
ridge=linear_model.Ridge()
lasso = linear_model.Lasso()


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn

#Defining the arrays we will use for the test data
Y_pred_leastsq = np.empty((Y_test.shape[0], n_boostraps))
Y_pred_lasso = np.empty((Y_test.shape[0], n_boostraps))
Y_pred_ridge = np.empty((Y_test.shape[0], n_boostraps))

#Definf the arrays we will use for obtaining the statistical parameters
error_leastsq = []
bias_leastsq = [] 
variance_leastsq = []

error_ridge  = []
bias_ridge = [] 
variance_ridge = []

error_lasso = []
bias_lasso = [] 
variance_lasso = []

#Main part of the program. It consists of two loops - one over lambda - to test
#the parameters and over the boostraps number. In every iteration the data is resampled
#and then the prediction is tested at the same test data. After the whole boostraping is
#performed, the error, variance and bias are computed
for lmbda in (np.logspace(-4, 5, 10)):
    for i in range(n_boostraps):
        x_, y_ = resample(X_train, Y_train)
    
        # Evaluate the new model on the same test data each time.
        Y_pred_leastsq[:, i] = leastsq.fit(x_, y_).predict(X_test).ravel()
        
        ridge.set_params(alpha=lmbda)
        Y_pred_ridge[:, i] = ridge.fit(x_, y_).predict(X_test).ravel()
        
        lasso.set_params(alpha=lmbda)
        Y_pred_lasso[:, i] = lasso.fit(x_, y_).predict(X_test).ravel()
        
    error_leastsq.append(np.mean( np.mean((Y_test[:, np.newaxis] - Y_pred_leastsq)**2, axis = 1, keepdims=True) ))
    bias_leastsq.append(np.mean( (Y_test[:, np.newaxis] - np.mean(Y_pred_leastsq, axis=1, keepdims=True))**2 ))
    variance_leastsq.append(np.mean( np.var(Y_pred_leastsq, axis=1, keepdims=True) ))
    
    error_ridge.append(np.mean( np.mean((Y_test[:, np.newaxis] - Y_pred_ridge)**2, axis = 1, keepdims=True) ))
    bias_ridge.append(np.mean( (Y_test[:, np.newaxis] - np.mean(Y_pred_ridge, axis=1, keepdims=True))**2 ))
    variance_ridge.append(np.mean( np.var(Y_pred_ridge, axis=1, keepdims=True) ))
    
    error_lasso.append(np.mean( np.mean((Y_test[:, np.newaxis] - Y_pred_lasso)**2, axis = 1, keepdims=True) ))
    bias_lasso.append(np.mean( (Y_test[:, np.newaxis] - np.mean(Y_pred_lasso, axis=1, keepdims=True))**2 ))
    variance_lasso.append(np.mean( np.var(Y_pred_lasso, axis=1, keepdims=True) ))




#Plotting the obtained results
plt.semilogx(np.logspace(-4, 5, 10), error_leastsq,'r-', label = 'OLS error')
plt.semilogx(np.logspace(-4, 5, 10), bias_leastsq, 'r--', label = 'OLS bias')
plt.semilogx(np.logspace(-4, 5, 10), variance_leastsq, 'r-.', label = 'OLS variance')


plt.semilogx(np.logspace(-4, 5, 10), error_ridge,'b-s', label = 'Ridge error')
plt.semilogx(np.logspace(-4, 5, 10), bias_ridge,'b--s', label = 'Ridge bias')
plt.semilogx(np.logspace(-4, 5, 10), variance_ridge, 'b-.s', label = 'Ridge variance')


plt.semilogx(np.logspace(-4, 5, 10), error_lasso, 'g-^', label = 'Lasso error')
plt.semilogx(np.logspace(-4, 5, 10), bias_lasso, 'g--^', label = 'Lasso bias')
plt.semilogx(np.logspace(-4, 5, 10), variance_lasso, 'g-.^', label = 'Lasso variance')
plt.legend()

plt.xlabel('Hyperparameter $\lambda$')
plt.ylabel('MSE, bias and variance')
plt.title('Bias - variance tradeoff')