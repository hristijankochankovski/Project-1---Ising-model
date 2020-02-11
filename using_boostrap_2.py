import numpy as np
import scipy.sparse as sp
np.random.seed(10000)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#definition of the bias
def compute_bias(y,ymodel):
    n = np.size(y)
    return np.sum((y-np.mean(ymodel))**2)/n

#definition of funciton variance
def compute_variance(ymodel):
    n = np.size(ymodel)
    return np.sum((ymodel-np.mean(ymodel))**2)/n


#import warnings
# Comment this to turn on warnings
#warnings.filterwarnings('ignore')

#First, lets define the data of the system. The size of the system is determined
#by the size of the periodic chain L 
# system size
L=40

# We will create 1000 Ising states
states=np.random.choice([-1, 1], size=(1000,L))
# Next, we define the ising energy
def ising_energies(states):
    L = states.shape[1]
    J = np.zeros((L, L),)
    for i in range(L): 
        J[i,(i+1)%L]=-1.0 
# only the interactions between the nearest neighbours i and i+1 are
# taken into account        
        
    # compute energies
    E = np.einsum('...i,ij,...j->...',states,J,states)

    return E
# calculate Ising energies
energies=ising_energies(states)

# reshape Ising states into RL samples: S_i S_j --> X_p
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
#
#X_train=Data[0][:n_samples]
#Y_train=Data[1][:n_samples] #+ np.random.normal(0,4.0,size=X_train.shape[0])
#X_test=Data[0][n_samples:3*n_samples//2]
#Y_test=Data[1][n_samples:3*n_samples//2] #+ np.random.normal(0,4.0,size=X_test.shape[0])

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
train_errors_leastsq_temp = []
test_errors_leastsq_temp = []
train_errors_ridge = []
test_errors_ridge = []
train_errors_ridge_temp = []
test_errors_ridge_temp = []
train_errors_lasso_temp = []
test_errors_lasso_temp = []
train_errors_lasso= []
test_errors_lasso = []
# set regularisation strength values
lmbdas = np.logspace(-3, 3, 7)

#Initialize coeffficients for ridge regression and Lasso
coefs_leastsq = []
coefs_ridge = []
coefs_lasso=[]
coefs_leastsq_temp = []
coefs_ridge_temp = []
coefs_lasso_temp = []
n_boostraps = 10




MSEtest_temp_leastsq =  []
MSEtrain_temp_leastsq =  []
variancetest_temp_leastsq = []
variancetrain_temp_leastsq = []
biastest_temp_leastsq = []
biastrain_temp_leastsq = []

MSEtest_leastsq =  []
MSEtrain_leastsq =  []
variancetest_leastsq = []
variancetrain_leastsq = []
biastest_leastsq = []
biastrain_leastsq = []


MSEtest_temp_ridge =  []
MSEtrain_temp_ridge =  []
variancetest_temp_ridge = []
variancetrain_temp_ridge = []
biastest_temp_ridge = []
biastrain_temp_ridge = []

MSEtest_ridge =  []
MSEtrain_ridge =  []
variancetest_ridge = []
variancetrain_ridge = []
biastest_ridge = []
biastrain_ridge = []


MSEtest_temp_lasso =  []
MSEtrain_temp_lasso =  []
variancetest_temp_lasso = []
variancetrain_temp_lasso = []
biastest_temp_lasso = []
biastrain_temp_lasso = []

MSEtest_lasso =  []
MSEtrain_lasso =  []
variancetest_lasso = []
variancetrain_lasso = []
biastest_lasso = []
biastrain_lasso = []












for lmbda in lmbdas:
# This is the loop for boostraping. At every iteration, the data is resampled
# and all the variables of interest are    
    for i in range(n_boostraps):
        x_, y_ = resample(X_train, Y_train)
        
        
        leastsq.fit(x_, y_)
        coefs_leastsq_temp.append(leastsq.coef_)
        train_errors_leastsq_temp.append(leastsq.score(x_, y_))
        test_errors_leastsq_temp.append(leastsq.score(X_test,Y_test))
        
        ridge.set_params(alpha=lmbda) # set regularisation parameter
        ridge.fit(x_, y_) # fit model 
        coefs_ridge_temp.append(ridge.coef_) 
        train_errors_ridge_temp.append(ridge.score(x_, y_))
        test_errors_ridge_temp.append(ridge.score(X_test,Y_test))
        
        
        
        lasso.set_params(alpha=lmbda) # set regularisation parameter
        lasso.fit(x_, y_) # fit model 
        coefs_lasso_temp.append(lasso.coef_) 
        train_errors_lasso_temp.append(lasso.score(x_, y_))
        test_errors_lasso_temp.append(lasso.score(X_test,Y_test))
        
       
    
    coefs_leastsq.append(np.mean(coefs_leastsq_temp)) # store weights
    coefs_ridge.append(np.mean(coefs_ridge_temp)) # store weights
    coefs_ridge.append(np.mean(coefs_lasso_temp))
    variancetrain_leastsq = variancetrain_leastsq.append(np.mean(variancetrain_temp_leastsq))
#     use the coefficient of determination R^2 as the performance of prediction.
    train_errors_leastsq.append(np.mean(train_errors_leastsq_temp))
    test_errors_leastsq.append(np.mean(test_errors_leastsq_temp))
    train_errors_ridge.append(np.mean(train_errors_ridge_temp))
    test_errors_ridge.append(np.mean(test_errors_ridge_temp))
    train_errors_lasso.append(np.mean(train_errors_lasso_temp))
    test_errors_lasso.append(np.mean(test_errors_lasso_temp))   
    ### apply RIDGE regression
#    ridge.set_params(alpha=lmbda) # set regularisation parameter
#    ridge.fit(X_train, Y_train) # fit model 
#    coefs_ridge.append(ridge.coef_) # store weights
#    # use the coefficient of determination R^2 as the performance of prediction.
# 
#    ### apply LASSO regression
#    lasso.set_params(alpha=lmbda) # set regularisation parameter
#    lasso.fit(X_train, Y_train) # fit model
#    coefs_lasso.append(lasso.coef_) # store weights
    # use the coefficient of determination R^2 as the performance of prediction.
   

    # plot Ising interaction J. We reshape the matrix in the form LxL
    J_leastsq=np.array(leastsq.coef_).reshape((L,L))
    J_ridge=np.array(ridge.coef_).reshape((L,L))
    J_lasso=np.array(lasso.coef_).reshape((L,L))

    cmap_args=dict(vmin=-1., vmax=1., cmap='seismic')

#    fig, axarr = plt.subplots(nrows=1, ncols=3)
#    
#    axarr[0].imshow(J_leastsq,**cmap_args)
#    axarr[0].set_title('OLS \n Train$=%.3f$, Test$=%.3f$'%(train_errors_leastsq[-1], test_errors_leastsq[-1]),fontsize=16)
#    axarr[0].tick_params(labelsize=16)
#    
#    axarr[1].imshow(J_ridge,**cmap_args)
#    axarr[1].set_title('Ridge $\lambda=%.4f$\n Train$=%.3f$, Test$=%.3f$' %(lmbda,train_errors_ridge[-1],test_errors_ridge[-1]),fontsize=16)
#    axarr[1].tick_params(labelsize=16)
#    
#    im=axarr[2].imshow(J_lasso,**cmap_args)
#    axarr[2].set_title('LASSO $\lambda=%.4f$\n Train$=%.3f$, Test$=%.3f$' %(lmbda,train_errors_lasso[-1],test_errors_lasso[-1]),fontsize=16)
#    axarr[2].tick_params(labelsize=16)
#    
#    divider = make_axes_locatable(axarr[2])
#    cax = divider.append_axes("right", size="5%", pad=0.01, add_to_figure=True)
#    cbar=fig.colorbar(im, cax=cax)
#    
#    cbar.ax.set_yticklabels(np.arange(-1.0, 1.0+0.25, 0.25),fontsize=14)
#    cbar.set_label('$J_{i,j}$',labelpad=15, y=0.5,fontsize=20,rotation=0)
#    
#    fig.subplots_adjust(right=0.8)
#    
#    plt.show()
#    
    
    
    
    # Plot our performance on both the training and test data
plt.semilogx(lmbdas, train_errors_leastsq, 'b',label='Train data (OLS)')
plt.semilogx(lmbdas, test_errors_leastsq,'--b',label='Test data (OLS)')
plt.semilogx(lmbdas, train_errors_ridge,'r',label='Train data (Ridge)',linewidth=1)
plt.semilogx(lmbdas, test_errors_ridge,'--r',label='Test data (Ridge)',linewidth=1)
plt.semilogx(lmbdas, train_errors_lasso, 'g',label='Train data (LASSO)')
plt.semilogx(lmbdas, test_errors_lasso, '--g',label='Test data (LASSO)')

fig = plt.gcf()

#plt.vlines(alpha_optim, plt.ylim()[0], np.max(test_errors), color='k',
#           linewidth=3, label='Optimum on test')
plt.legend(loc='middle left',fontsize=16)
plt.ylim([0, 1.1])
plt.xlim([min(lmbdas), max(lmbdas)])
plt.xlabel(r'$\lambda$',fontsize=16)
plt.ylabel('Performance',fontsize=16)
plt.tick_params(labelsize=12)
plt.show()

print(variancetrain_leastsq)