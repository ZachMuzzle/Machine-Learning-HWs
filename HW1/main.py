# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed to plot 3-D surfaces

# tells matplotlib to embed plots within the notebook
#%matplotlib inline

# You need to read "data/data_description.txt" to understand data

# Load data from data.txt
# Total 9 columns: column 0-7 are numbers, column 8 are string
data_path = os.path.join('data','data.txt')
numbers = np.loadtxt(data_path, delimiter='\t', usecols=[0,1,2,3,4,5,6,7])
#strings = np.loadtxt('./data/data.txt', delimiter='\t', usecols=[8])

# We want to predict 'mpg'(1th column) based on 'horsepower'(4rd column)
x = numbers[:,3]
y = numbers[:,0]
m = y.size # the number of data examples

# Here is an [CHECKPOINT 0] example
# Your code will be graded based on: 
# 1. The correctness of the code you implement in the function
# 2. The correctness of the result/output by running the your code 

def plot_data(x, y):
    """
    Plots the data points x and y into a new figure. Plots the data 
    points and gives the figure axes labels of population and profit.
    
    Parameters
    ----------
    x : array_like
        Data point values for x-axis.

    y : array_like
        Data point values for y-axis. Note x and y should have the same size.
    
    Instructions
    ------------
    Plot the training data into a figure using the "figure" and "plot"
    functions. Set the axes labels using the "xlabel" and "ylabel" functions.
    Assume the population and revenue data have been passed in as the x
    and y arguments of this function.    
    
    Hint
    ----
    You can use the 'ro' option with plot to have the markers
    appear as red circles. Furthermore, you can make the markers larger by
    using plot(..., 'ro', ms=10), where `ms` refers to marker size. You 
    can also set the marker edge color using the `mec` property.
    """
    fig = plt.figure()  # open a new figure
    
    # ====================== YOUR CODE HERE ======================= 
    
    plt.plot(x, y, 'ro', mec='k', ms=7)
    plt.ylabel('mpg')
    plt.xlabel('horsepower')
    plt.savefig('figure/alpha.png')

    # =============================================================

    # Plots the data
plot_data(x, y)


    # [CHECKPOINT 1][8 points]
def  feature_normalization(x):
    """
    Normalizes the features in x. returns a normalized version of x where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when working with
    learning algorithms.
    
    Parameters
    ----------
    x : array_like
        The dataset of size (m).
    
    Returns
    -------
    x_norm : array_like
        The normalized dataset of size (m).
    
    Instructions
    ------------
    For each feature, compute the mean of the feature
    and subtract it from the dataset, storing the mean value in mu. 
    Next, compute the  standard deviation of each feature and divide
    each feature by it's standard deviation, storing the standard deviation 
    in sigma. 
    
    Hint
    ----
    You might find the 'np.mean' and 'np.std' functions useful.
    """
    # ====================== YOUR CODE HERE =====================
    # You need to set these values correctly
    x_norm = x.copy()
    mu = np.mean(x[:,None]) #or np.mean(x,axis=0)
    sigma = np.std(x[:,None])
    x_norm = (x-mu)/sigma
    # ===========================================================

    return x_norm, mu, sigma

# call featureNormalize on the loaded data
x_norm, mu, sigma = feature_normalization(x)

print('Computed mean:', mu)
print('Computed standard deviation:', sigma)

# Add a column of ones to X. The numpy function stack joins arrays along a given axis. 
# The first axis (axis=0) refers to rows (training examples) 
# and second axis (axis=1) refers to columns (features).
X = np.stack([np.ones(m), x_norm], axis=1)

# [CHECKPOINT 2][8 points]
def compute_cost(X, y, theta):
    """
    Compute cost for linear regression. Computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y.
    
    Parameters
    ----------
    X : array_like
        The input dataset of shape (m x n+1), where m is the number of examples,
        and n is the number of features. We assume a vector of one's already 
        appended to the features so we have n+1 columns.
    
    y : array_like
        The values of the function at each data point. This is a vector of
        shape (m, ).
    
    theta : array_like
        The parameters for the regression function. This is a vector of 
        shape (n+1, ).
    
    Returns
    -------
    J : float
        The value of the regression cost function.
    
    Instructions
    ------------
    Compute the cost of a particular choice of theta. 
    You should set J to the cost.
    """
    
    # initialize some useful values
    m = y.size  # number of training examples
    
    # You need to return the following variables correctly
    J = 0
    
    # ====================== YOUR CODE HERE =====================
    
    #J = np.sum(np.square((X*theta)- y))/ (2 * m)
    #temp = np.dot(X,theta) - y
    
    #J = np.sum(np.power(temp,2)) / (2*m)
    temp = np.dot(X,theta)
    J = np.sum(np.power((temp-y),2))/ (2*m)
    # ===========================================================
    
    return J

J = compute_cost(X, y, theta=np.array([0.0, 0.0]))
print('With theta = [0, 0] \nCost computed = %.2f' % J)
print('Expected cost value (approximately) 305.24\n')

# further testing of the cost function
J = compute_cost(X, y, theta=np.array([30.0, -0.1]))
print('With theta = [30, -0.1]\nCost computed = %.2f' % J)
print('Expected cost value (approximately) 51.26')

# [CHECKPOINT 3][8 points]
def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn `theta`. Updates theta by taking `num_iters`
    gradient steps with learning rate `alpha`.
    
    Parameters
    ----------
    X : array_like
        The input dataset of shape (m x n+1).
    
    y : arra_like
        Value at given features. A vector of shape (m, ).
    
    theta : array_like
        Initial values for the linear regression parameters. 
        A vector of shape (n+1, ).
    
    alpha : float
        The learning rate.
    
    num_iters : int
        The number of iterations for gradient descent. 
    
    Returns
    -------
    theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, ).
    
    J_history : list
        A python list for the values of the cost function after each iteration.
    
    Instructions
    ------------
    Peform a single gradient step on the parameter vector theta.

    While debugging, it can be useful to print out the values of 
    the cost function (computeCost) and gradient here.
    """
    # Initialize some useful values
    m = y.shape[0]  # number of training examples
    
    # make a copy of theta, to avoid changing the original array, since numpy arrays
    # are passed by reference to functions
    theta = theta.copy()
    
    J_history = [] # Use a python list to save cost in every iteration
    
    for it in range(num_iters):
        # ==================== YOUR CODE HERE =================================
        temp = np.dot(X,theta) - y
        temp2 = np.dot(X.T, temp)
        theta = theta - (alpha/m) * temp2


        # =====================================================================
        
        # save the cost J in every iteration
        J_history.append(compute_cost(X, y, theta))
        
    return theta, J_history

# initialize fitting parameters
theta = np.array([0.0, 0.0])

# some gradient descent settings
iterations = 200
alpha = .1

theta, J_history = gradient_descent(X ,y, theta, alpha, iterations)

# plot the linear fit
plot_data(X[:, 1], y)
plt.plot(X[:, 1], np.dot(X, theta), '-')
plt.legend(['Training data', 'Linear regression']);
plt.savefig('figure/gradient_descent.png')

print('Theta found by gradient descent: {:.4f}, {:.4f}'.format(*theta))
print('Expected theta values (approximately): [23.4459, -6.0679]')

# [CHECKPOINT 4][2 points]
# Predict MPG for horsepower of 50 and 200
#print(theta.shape)
predict1 = np.dot([1,(50-mu)/sigma], theta) # MPG = 1 column
print('For horsepower = 50, we predict a MPG of {:.2f}\n'.format(predict1))

predict2 = np.dot([1, (200-mu)/sigma], theta)
print('For horsepower = 200, we predict a MPG of {:.2f}\n'.format(predict2))

# grid over which we will calculate J
theta0_vals = np.linspace(-20, 70, 100)
theta1_vals = np.linspace(-50, 30, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))

# Fill out J_vals
for i, theta0 in enumerate(theta0_vals):
    for j, theta1 in enumerate(theta1_vals):
        J_vals[i, j] = compute_cost(X, y, np.array([theta0, theta1]))
        
# Because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals.T

# surface plot
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap='viridis')
plt.xlabel('theta0')
plt.ylabel('theta1')
plt.title('Surface')
# contour plot
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
ax = plt.subplot(122)
plt.contour(theta0_vals, theta1_vals, J_vals, linewidths=2, cmap='viridis', levels=np.logspace(-2, 3, 20))
plt.xlabel('theta0')
plt.ylabel('theta1')
plt.plot(theta[0], theta[1], 'ro', ms=10, lw=2)
plt.title('Contour, showing minimum')
plt.savefig('figure/surface_contour.png')
pass

# Load data
# We want to predict 'mpg'(0th column) based on 'horsepower'(3rd column) and 'weight' (5th column)
# Note that we use X instead of x to specify it is a matrix instead of a vector
X = numbers[:,3:5]
y = numbers[:,0]
m = y.size # the number of data examples

# print out some data points
print('{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:, 1]', 'y'))
print('-'*26)
for i in range(10):
    print('{:8.2f}{:8.2f}{:10.2f}'.format(X[i, 0], X[i, 1], y[i]))

# [CHECKPOINT 5][5 points]
def  feature_normalization_multiple(X):
    """
    Normalizes the features in X. returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when working with
    learning algorithms.
    
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n).
    
    Returns
    -------
    X_norm : array_like
        The normalized dataset of shape (m x n).
    
    Instructions
    ------------
    First, for each feature dimension, compute the mean of the feature
    and subtract it from the dataset, storing the mean value in mu. 
    Next, compute the  standard deviation of each feature and divide
    each feature by it's standard deviation, storing the standard deviation 
    in sigma. 
    
    Note that X is a matrix where each column is a feature and each row is
    an example. You needto perform the normalization separately for each feature. 
    
    Hint
    ----
    You might find the 'np.mean' and 'np.std' functions useful.
    """
    # You need to set these values correctly
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])

    # =========================== YOUR CODE HERE =====================
    #mu = np.mean(X[:,1]) # just does one
    #sigma = np.std(X[:,1]) # just does one
   # X_norm = (X-mu)/sigma 
    
    #mu = np.mean(X,axis=0)
    #sigma = np.std(X,axis=0)
    #X_norm = (X-mu)/sigma
    
    s = X.shape[1]
    for i in range(s):
        mu[i] = np.mean(X[:,i])
    for i in range(s):
        sigma[i] = np.std(X[:,i])
    for i in range(s):
        X_norm[:,i] = (X[:,i]-mu[i])/sigma[i]   
    
    # ================================================================
    return X_norm, mu, sigma

    # call featureNormalize on the loaded data
X_norm, mu, sigma = feature_normalization_multiple(X)

print('Computed mean:', mu)
print('Computed standard deviation:', sigma)

# print out some data points
print('{:>8s}{:>8s}{:>10s}'.format('X_norm[:,0]', 'X_norm[:, 1]', 'y'))
print('-'*26)
for i in range(10):
    print('{:8.2f}{:8.2f}{:10.2f}'.format(X_norm[i, 0], X_norm[i, 1], y[i]))

X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)

# [CHECKPOINT 6][8 points]
def compute_cost_multiple(X, y, theta):
    """
    Compute cost for linear regression with multiple variables.
    Computes the cost of using theta as the parameter for linear regression to fit the data points in X and y.
    
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).
    
    y : array_like
        A vector of shape (m, ) for the values at a given data point.
    
    theta : array_like
        The linear regression parameters. A vector of shape (n+1, )
    
    Returns
    -------
    J : float
        The value of the cost function. 
    
    Instructions
    ------------
    Compute the cost of a particular choice of theta. You should set J to the cost.
    """
    # Initialize some useful values
    m = y.shape[0] # number of training examples
    
    # You need to return the following variable correctly
    J = 0
    
    # ======================= YOUR CODE HERE ===========================
    #temp = np.dot(X,theta)
    #error = np.sum(np.square(temp-y))
    #J = error / (2 * m)
    ## Trying to fix this
    temp = np.dot(X,theta) - y
    
    J = np.sum(np.power(temp,2)) / (2*m)
    
    # ==================================================================
    return J

# [CHECKPOINT 7][8 points]
def gradient_descent_multiple(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn theta.
    Updates theta by taking num_iters gradient steps with learning rate alpha.
        
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).
    
    y : array_like
        A vector of shape (m, ) for the values at a given data point.
    
    theta : array_like
        The linear regression parameters. A vector of shape (n+1, )
    
    alpha : float
        The learning rate for gradient descent. 
    
    num_iters : int
        The number of iterations to run gradient descent. 
    
    Returns
    -------
    theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, ).
    
    J_history : list
        A python list for the values of the cost function after each iteration.
    
    Instructions
    ------------
    Peform a single gradient step on the parameter vector theta.

    While debugging, it can be useful to print out the values of 
    the cost function (computeCost) and gradient here.
    """
    # Initialize some useful values
    m = y.shape[0] # number of training examples
    
    # make a copy of theta, which will be updated by gradient descent
    theta = theta.copy()
    
    J_history = []
    
    for it in range(num_iters):
        # ======================= YOUR CODE HERE ==========================
        temp = np.dot(X,theta) - y
        
        temp2 = np.dot(X.T,temp)
        
        theta = theta - (alpha/m) * temp2

        
        
        





        # =================================================================
        
        # save the cost J in every iteration
        J_history.append(compute_cost_multiple(X, y, theta))
    
    return theta, J_history

    """
Instructions
------------
We have provided you with the following starter code that runs
gradient descent with a particular learning rate (alpha). 

Your task is to first make sure that your functions - `compute_cost`
and `gradient_descent` already work with  this starter code and
support multiple variables.

After that, try running gradient descent with different values of
alpha and see which one gives you the best result.
"""
# Choose some alpha value - change this
alpha = .1
num_iters = 200

# init theta and run gradient descent
theta = np.zeros(3)
theta, J_history = gradient_descent_multiple(X, y, theta, alpha, num_iters)

# Plot the convergence graph
plt.plot(np.arange(len(J_history)), J_history, lw=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')

# Display the gradient descent's result
print('theta computed from gradient descent: {:s}'.format(str(theta)))

# [CHECKPOINT 8][5 points]
"""
Finally, you should complete the code at the end to predict the MPG based on HORSEPOWER and WEIGHT.

Hint
----
At prediction, make sure you do the same feature normalization.
"""

# Horsepower = 100, Weight = 2500
X_test1 = np.array([100, 2500])

# Horsepower = 200, Weight = 4500
X_test2 = np.array([200, 4500])
#print(theta.shape)
# ======================= YOUR CODE HERE ==========================
array = [1,X_test1[0],X_test1[1]]
array[1:3] = (array[1:3] - mu) / sigma
y_test1 = np.dot(array,theta)

# Next predict
array2 = [1,X_test2[0],X_test2[1]]
array2[1:3] = (array2[1:3] - mu) / sigma
y_test2 = np.dot(array2,theta)




# =================================================================

print('Predicted MPG of Horsepower = 100 and Weight = 2500 is {:.0f}'.format(y_test1))
print('Predicted MPG of Horsepower = 200 and Weight = 4500 is {:.0f}'.format(y_test2))

# [CHECKPOINT 9][8 points]
# All the codes you need to repeat the gradient descent without feature normalization
# ======================= YOUR CODE HERE ==========================

x_norm = x.copy()
#----------------------------------
X = np.stack([np.ones(m), x_norm], axis=1)
#-----------------------------------------------
def compute_cost(X, y, theta):
    """
    Compute cost for linear regression. Computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y.
    
    Parameters
    ----------
    X : array_like
        The input dataset of shape (m x n+1), where m is the number of examples,
        and n is the number of features. We assume a vector of one's already 
        appended to the features so we have n+1 columns.
    
    y : array_like
        The values of the function at each data point. This is a vector of
        shape (m, ).
    
    theta : array_like
        The parameters for the regression function. This is a vector of 
        shape (n+1, ).
    
    Returns
    -------
    J : float
        The value of the regression cost function.
    
    Instructions
    ------------
    Compute the cost of a particular choice of theta. 
    You should set J to the cost.
    """
    
    # initialize some useful values
    m = y.size  # number of training examples
    
    # You need to return the following variables correctly
    J = 0
    
    # ====================== YOUR CODE HERE =====================
    
    #J = np.sum(np.square((X*theta)- y))/ (2 * m)
    temp = np.dot(X,theta) - y
    
    J = np.sum(np.power(temp,2)) / (2*m)
    # ===========================================================
    
    return J


#--------------------------------------------------------------------------
J = compute_cost(X, y, theta=np.array([0.0, 0.0]))
print('With theta = [0, 0] \nCost computed = %.2f' % J)
print('Expected cost value (approximately) 305.24\n')

# further testing of the cost function
J = compute_cost(X, y, theta=np.array([30.0, -0.1]))
print('With theta = [30, -0.1]\nCost computed = %.2f' % J)
print('Expected cost value (approximately) 51.26')
#---------------------------------------------------------------------------
def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn `theta`. Updates theta by taking `num_iters`
    gradient steps with learning rate `alpha`.
    
    Parameters
    ----------
    X : array_like
        The input dataset of shape (m x n+1).
    
    y : arra_like
        Value at given features. A vector of shape (m, ).
    
    theta : array_like
        Initial values for the linear regression parameters. 
        A vector of shape (n+1, ).
    
    alpha : float
        The learning rate.
    
    num_iters : int
        The number of iterations for gradient descent. 
    
    Returns
    -------
    theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, ).
    
    J_history : list
        A python list for the values of the cost function after each iteration.
    
    Instructions
    ------------
    Peform a single gradient step on the parameter vector theta.

    While debugging, it can be useful to print out the values of 
    the cost function (computeCost) and gradient here.
    """
    # Initialize some useful values
    m = y.shape[0]  # number of training examples
    
    # make a copy of theta, to avoid changing the original array, since numpy arrays
    # are passed by reference to functions
    theta = theta.copy()
    
    J_history = [] # Use a python list to save cost in every iteration
    
    for it in range(num_iters):
        # ==================== YOUR CODE HERE =================================
        temp = np.dot(X,theta) - y
        temp = np.dot(X.T, temp)
        theta = theta - (alpha/m) * temp


        # =====================================================================
        
        # save the cost J in every iteration
        J_history.append(compute_cost(X, y, theta))
        
    return theta, J_history
#-----------------------------------------------------------------------------
# initialize fitting parameters
theta = np.array([0.0, 0.0])

# some gradient descent settings
iterations = 200
alpha = 0.00001
theta, J_history = gradient_descent(X ,y, theta, alpha, iterations)

# plot the linear fit
plot_data(X[:, 1], y)
plt.plot(X[:, 1], np.dot(X, theta), '-')
plt.legend(['Training data', 'Linear regression']);
plt.savefig('figure/gradient_descent_no_normalization.png')

print('Theta found by gradient descent: {:.4f}, {:.4f}'.format(*theta))
print('Expected theta values (approximately): [23.4459, -6.0679]')




# =================================================================