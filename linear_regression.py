import numpy as np
import scipy.sparse as sp
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



np.random.seed(10000)

#Defining the dimension of the system
L=40

# create 10000 random Ising states
states=np.random.choice([-1, 1], size=(10000,L))


# Creating a function that computes the energy in the Ising model
def ising_energies(states):

    L = states.shape[1]
    J = np.zeros((L, L),)
    for i in range(L): 
        J[i,(i+1)%L]=-1.0 # interaction between nearest-neighbors
        
    # compute energies
    E = np.einsum('...i,ij,...j->...',states,J,states)

    return E
# calculate Ising energies
energies=ising_energies(states)

# reshape Ising states into RL samples: S_iS_j --> X_p
states=np.einsum('...i,...j->...ij', states, states)
shape=states.shape
states=states.reshape((shape[0],shape[1]*shape[2]))
# build final data set
Data=[states,energies]


# define number of samples
n_samples = 400

X_train, X_test, Y_train, Y_test = train_test_split(Data[0][:n_samples],Data[1][:n_samples], test_size = 0.2)

# set up the regression
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

# set an array of the regularization hyperparameter that we will use
lmbdas = np.logspace(-4, 5, 10)

#Initialize coeffficients for ridge regression and Lasso
coefs_leastsq = []
coefs_ridge = []
coefs_lasso=[]

for lmbda in lmbdas:

#OLS method
    leastsq.fit(X_train, Y_train) # fit model 
    coefs_leastsq.append(leastsq.coef_) # store weights
    # use the coefficient of determination R^2 as the performance of prediction.
    train_errors_leastsq.append(leastsq.score(X_train, Y_train))
    test_errors_leastsq.append(leastsq.score(X_test,Y_test))
    
#Ridge method
    ridge.set_params(alpha=lmbda) # set regularisation parameter
    ridge.fit(X_train, Y_train) # fit model 
    coefs_ridge.append(ridge.coef_) # store weights
    # use the coefficient of determination R^2 as the performance of prediction.
    train_errors_ridge.append(ridge.score(X_train, Y_train))
    test_errors_ridge.append(ridge.score(X_test,Y_test))
    
#Lasso method
    lasso.set_params(alpha=lmbda) # set regularisation parameter
    lasso.fit(X_train, Y_train) # fit model
    coefs_lasso.append(lasso.coef_) # store weights
    # use the coefficient of determination R^2 as the performance of prediction.
    train_errors_lasso.append(lasso.score(X_train, Y_train))
    test_errors_lasso.append(lasso.score(X_test,Y_test))
    
    J_leastsq=np.array(leastsq.coef_).reshape((L,L))
    J_ridge=np.array(ridge.coef_).reshape((L,L))
    J_lasso=np.array(lasso.coef_).reshape((L,L))

    cmap_args=dict(vmin=-1., vmax=1., cmap='seismic')

    fig, axarr = plt.subplots(nrows=1, ncols=3)
    
    axarr[0].imshow(J_leastsq,**cmap_args)
    axarr[0].set_title('OLS \n Train$=%.3f$, Test$=%.3f$'%(train_errors_leastsq[-1], test_errors_leastsq[-1]),fontsize=16)
    axarr[0].tick_params(labelsize=16)
    
    axarr[1].imshow(J_ridge,**cmap_args)
    axarr[1].set_title('Ridge $\lambda=%.3f$\n Train$=%.3f$, Test$=%.3f$' %(lmbda,train_errors_ridge[-1],test_errors_ridge[-1]),fontsize=16)
    axarr[1].tick_params(labelsize=16)
    
    im=axarr[2].imshow(J_lasso,**cmap_args)
    axarr[2].set_title('LASSO $\lambda=%.3f$\n Train$=%.3f$, Test$=%.3f$' %(lmbda,train_errors_lasso[-1],test_errors_lasso[-1]),fontsize=16)
    axarr[2].tick_params(labelsize=16)
    
    divider = make_axes_locatable(axarr[2])
    cax = divider.append_axes("right", size="5%", pad=0.05, add_to_figure=True)
    cbar=fig.colorbar(im, cax=cax)
    
    cbar.ax.set_yticklabels(np.arange(-1.0, 1.0+0.25, 0.25),fontsize=14)
    cbar.set_label('$J_{i,j}$',labelpad=15, y=0.5,fontsize=20,rotation=0)
    
    fig.subplots_adjust(right=0.8)
    
    plt.show()


#
#plt.semilogx(lmbdas, train_errors_leastsq, 'b',label='Train (OLS)')
#plt.semilogx(lmbdas, test_errors_leastsq,'--y',label='Test (OLS)', linewidth = 4)
#plt.semilogx(lmbdas, train_errors_ridge,'r',label='Train (Ridge)',linewidth=2)
#plt.semilogx(lmbdas, test_errors_ridge,'--r',label='Test (Ridge)',linewidth=2)
#plt.semilogx(lmbdas, train_errors_lasso, 'g',label='Train (LASSO)')
#plt.semilogx(lmbdas, test_errors_lasso, '--g',label='Test (LASSO)')
#
#fig = plt.gcf()
#fig.set_size_inches(12.0, 6.0)
##fig.patch.set_facecolor('white')
##fig.patch.set_alpha(0.7)
##plt.vlines(alpha_optim, plt.ylim()[0], np.max(test_errors), color='k',
##           linewidth=3, label='Optimum on test')
#plt.legend(loc='middle left',fontsize = 14)
#plt.ylim([-0.1, 1.1])
#plt.xlim([min(lmbdas), max(lmbdas)])
#plt.xlabel(r'Hyperparameter ($\lambda$)',fontsize=18)
#plt.ylabel('$R^2$',fontsize=18)
#plt.tick_params(labelsize=18)
#plt.title('Performance of the different regressions taking 1000 samples', fontsize = 20)
#plt.show()
