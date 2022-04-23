# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
import matplotlib.pyplot as plt

# tells matplotlib to embed plots within the notebook
#%matplotlib inline

# Create training dataset
m = 200
mu = 0
sigma = 0.1
X_data = np.array([(i)/(m) for i in range(m)])

# [CHECKPOINT 1][5 points]
# Calculate Y
# Plot the dataset
# ======================= YOUR CODE HERE ===========================
#x = np.arange(0,1)# from 0 to 1.
#x = np.array([0,1])
#print(x)
f_sin = np.sin(2*np.pi*X_data) 
noise = np.random.normal(mu,(sigma),X_data.shape)
Y_data = f_sin + noise


#print(X_data.shape)
#print(noise)
#print(x)
#print(X_data)
#print(Y_data)
plt.plot(X_data,Y_data, "b+")
plt.savefig('figure/dataset.png')


# ==================================================================

# Here is an example 
# convert X to a column matrix of 5 x 1
# generate polynomial [1 X X^2 X^3]
# Use this hint to implement normal_equation()
X_col = X_data[:5].reshape(5,1)
X_poly = np.power(X_col, np.arange(4))
print(X_col)
print(X_poly)

# [CHECKPOINT 2][10 points]
def normal_equation(X, Y, n):
    """
    Computes the closed-form solution to linear regression using the normal equations.
    
    Parameters
    ----------
    X : array_like
        The dataset of shape (m, ).
    
    Y : array_like
        The value at each data point. A vector of shape (m, ).
        
    n : the order of polynomial regression model
        Remember the number of features will be n+1.
    
    Returns
    -------
    theta : array_like
        Estimated polynomial regression parameters. A vector of shape (n+1, ).
    
    Instructions
    ------------
    Complete the code to compute the closed form solution to linear
    regression and put the result in theta.
    
    Hint
    ----
    Look up the function `np.linalg.pinv` for computing matrix inverse.
    """
    m = X.size
    theta = np.zeros(n+1) 
    
    # ===================== YOUR CODE HERE ============================
    X_col = X[:m].reshape(m,1)
    X_poly = np.power(X_col,np.arange(n))
    #Y_col = (Y[:, np.newaxis])
    Y_col = Y[:m].reshape(m,1)
    #print("Theta size:")
   # print(theta.shape)
    theta =  np.linalg.pinv(X_poly.T.dot(X_poly)).dot(X_poly.T).dot(Y_col)
    #print("New theta size: ")
    #print(theta.shape)
    #print(theta)
    # =================================================================
    return theta.flatten()

    # [CHECKPOINT 3][5 points]
def polynomial_deploy(X, theta):
    """
    Computes the polynomial regression prediction for data X.
    
    Parameters
    ----------
    X : array_like
        The input data. A vector of shape (m, ).
    
    theta : array_like
        Polynomial regression parameters. A vector of shape (n+1, ).
    
    Returns
    -------
    Y : array_like
        Polynomial prediction. A vector of shape (m, ).
    
    """
    m = X.size
    n = theta.size # removed minus 1
    #print("m size")
    #print(m)
    #print("theta size:")
    #print(n)
    # ===================== YOUR CODE HERE ============================
    X_col = X[:m].reshape(m,1)
    #print(X_col.shape)
    X_poly = np.power(X_col,np.arange(n))
    
    theta_col = (theta[:n].reshape(n,1))
    
    #print("X_poly shape from deploy fun")
    #print(X_poly.shape)
    #print("theta shape from deploy fun")
    #print(theta_col.shape)
    Y = np.dot(X_poly,theta_col)
    # ===================== YOUR CODE HERE ============================
    
    return Y.flatten()
    
# [CHECKPOINT 4][5 points]
# You mush use loop to traverse the combinations of m and n
#
# Hint for debug:
# A quick way to check if your implementation is correct is to use 'np.polyfit'
# e.g. theta2 = np.polyfit(X_data, Y_data, n)
# Check if 'theta' estimated by normal_equation() is the same as 'theta2' 
#
for m in [10, 50, 100, 200]:
    for n in [0, 1, 2,3, 9]:
        subset_index = np.random.randint(0, X_data.size, m)
        
        X_subset = X_data[subset_index]
        Y_subset = Y_data[subset_index]
        
        # ===================== YOUR CODE HERE ============================
        theta = normal_equation(X_subset,Y_subset,n+1) #n+1 number of features
        #print("theta in main: ")
        #print(theta.shape)
        Y_predict = polynomial_deploy(X_subset, theta)
        MSE = np.square(np.subtract(Y_subset,Y_predict)).mean()
        theta2= np.polyfit(X_subset,Y_subset,n) #for testing
                        
        print('The order of polynomial: %d' % n)
        print('Theta:')
        print(theta)
        print('MSE: %.8f' % MSE)
        print('Theata2: ')
        print(theta2)
        
        # plot the polynomial curve
        plt.figure()
        plt.title('m=%d, n=%d, MSE=%.8f' % (m, n, MSE))
        plt.plot(X_subset, Y_subset, 'b+')
        X_plot = np.arange(0,1,0.01)
        Y_plot = polynomial_deploy(X_plot, theta)
        plt.plot(X_plot, Y_plot, 'r-')
        plt.savefig('figure/polynomial_regression/polynomial_curve_' + str(m) + '_' + str(n) + '.png')
        plt.close()          
        # =================================================================

# [CHECKPOINT 6][10 points]
def normal_equation_reg(X, Y, n, lambd):
    """
    Computes the closed-form solution to linear regression using the normal equations.
    
    Parameters
    ----------
    X : array_like
        The dataset of shape (m, ).
    
    Y : array_like
        The value at each data point. A vector of shape (m, ).
        
    n : the order of polynomial regression model
        Remember the number of features will be n+1.
        
    lambd: the weight to ballance fitting error and regularization loss
    
    Returns
    -------
    theta : array_like
        Estimated polynomial regression parameters. A vector of shape (n+1, ).
    
    Instructions
    ------------
    Complete the code to compute the closed form solution to linear
    regression and put the result in theta.
    
    Hint
    ----
    Look up the function `np.linalg.pinv` for computing matrix inverse.
    """
    theta = np.zeros(n+1)
    
    # ===================== YOUR CODE HERE ============================
    m = X.size
    X_col = X[:m].reshape(m,1)
    X_poly = np.power(X_col,np.arange(n))
    Y_col = Y[:m].reshape(m,1)
    theta = np.linalg.pinv(X_poly.T.dot(X_poly) + lambd *np.identity(n)).dot(X_poly.T).dot(Y_col)
    
    
    # =================================================================
    return theta

# [CHECKPOINT 7][5 points]
# Create the testing dataset
# Plot the dataset
# ======================= YOUR CODE HERE ===========================
m = 199
#mu = 0
#sigma = 0.1
#X_data2 = np.array([((i)/(m)) + (1/(2*(m-1))) for i in range(m)]) 
X_testing = X_data + (1/(2*(m-1)))
X_testing = np.delete(X_testing,0)
#f_sin = np.sin(2*np.pi*X_data2)
#noise = np.random.normal(mu,(sigma), X_data2.shape)
Y_testing = np.sin(2*np.pi*X_testing) + np.random.normal(mu,sigma,X_testing.shape)

plt.plot(X_testing,Y_testing, "b+")


# ==================================================================

# [CHECKPOINT 8][10 points]
#
# ======================= YOUR CODE HERE ===========================
for m in [10,50]:
    for n in [3,9]:
        subset_index = np.random.randint(0, X_testing.size, m)
        subset_index2 = np.random.randint(0,X_data.size,m)
        X_subset2 = X_data[subset_index2] #training
        Y_subset2 = Y_data[subset_index2]
        X_subset = X_testing[subset_index] #testing
        Y_subset = Y_testing[subset_index]
        
        theta2 = normal_equation(X_subset2, Y_subset2,n+1) # equation for training set
        theta = normal_equation_reg(X_subset, Y_subset,n+1,0)
        
        Y_predict2 = polynomial_deploy(X_subset2,theta2) # predict for training
        Y_predict = polynomial_deploy(X_subset,theta)
        
        MSE2 = np.square(np.subtract(Y_subset2,Y_predict2)).mean() #MSE for training
        MSE = np.square(np.subtract(Y_subset,Y_predict)).mean()


#print statements
        print('The order of polynomial: %d' % n)
        print('Theta for training:')
        print(theta2)
        print('Theta for testing:')
        print(theta)
        print('MSE for training: %.8f' % MSE2)
        print('MSE for testing: %.8f' % MSE)

#plot the curve
        plt.figure()
        plt.title('m=%d, n=%d, MSE=%.8f' % (m, n, MSE))
        plt.plot(X_subset, Y_subset, 'b+', label='testing')
        X_plot = np.arange(0,1,0.1)
        Y_plot = polynomial_deploy(X_plot,theta)
        plt.plot(X_plot, Y_plot, 'r-', label='testing regression')
        plt.plot(X_subset2,Y_subset2, 'y+',label='training')
        X_plot2 = np.arange(0,1,0.1)
        Y_plot2 = polynomial_deploy(X_plot2,theta2)
        plt.plot(X_plot2, Y_plot2, 'b-', label='training regression')
        plt.legend()
        plt.savefig('figure/199_curve/curve_' + str(m) + '_' + str(n) + '.png')
        plt.close()

# ==================================================================