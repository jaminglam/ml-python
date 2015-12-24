import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def cost_func(X, y, theta, reg_param):
    """ 
        @param X: X Matrix
        @param y: actual result y
        @param theta: feature params, in this case is a [[theta1], [theta2], ... , [thetaj]]  matrix, eg. [[theta1], [theta2]]
        @param reg_param: lambda to control the contribution of the current theta, if the current theta is large, it would affect the cost function j severely,
            if the current theta is small, it would not affect the cost function j result apparently
        @return cost function result matrix
    """
    m = X.shape[0]
    h = np.dot(X, theta)
    mse = (1./(2*m)) * h.T * h # mean square error part
    reg_param_control = (float(reg_param)/(2*m))*np.dot(theta.T, theta)
    j = mse + reg_param_control
    return j

def cost_func_derivative(X, y, theta, reg_param, iter):
    """ 
        @param X: X Matrix
        @param y: actual result y
        @param theta: feature params, in this case is a [[theta1], [theta2], ... , [thetaj]]  matrix, eg. [[theta1], [theta2]]
        @param reg_param: lambda to control the contribution of the current theta, if the current theta is large, it would affect the cost function j severely,
            if the current theta is small, it would not affect the cost function j result apparently
        @param iter: times for iteration
        @return cost function derivative result matrix
    """
    m = X.shape[0]
    h = np.dot(X, theta)
    se = np.dot(X.T, h-y) # square error
    derivative = (1./m)*(se+reg_param*theta)
    return derivative

def update_theta(X, y, theta, alpha, reg_param, iter_times):
    """ 
        @param X: X Matrix
        @param y: actual result y
        @param theta: feature params, in this case is a [[theta1], [theta2], ... , [thetaj]]  matrix, eg. [[theta1], [theta2]]
        @param alpha: learning rate, NOTICE, do NOT set learning rate with a too large number for avoiding missing the local peak
        @param reg_param: lambda to control the contribution of the current theta, if the current theta is large, it would affect the cost function j severely,
            if the current theta is small, it would not affect the cost function j result apparently
        @param iter_times: times for iteration
        @return updated theta matrix
    """
    #print theta.shape    
    derivative = cost_func_derivative(X, y, theta, reg_param, iter_times)
    new_theta = theta - alpha*derivative
    return new_theta

def map_feature(x, polynomial_degree):
    """
       @param X: X Matrix
       @param polynomial_degree: eg. If you set polynomial degree as 5, its highest power 5, x^0, x^1, x^2, x^3, x^4, x^5
    """
    X = np.ones((x.shape[0], polynomial_degree+1))

    for i in range(1, polynomial_degree):
        x_pow = np.power(x, i)
        X[:,i] = x_pow
    return X

def gradient_desc(X, y, theta, alpha, reg_param, iter_times):
    """
       @param X: X Matrix
       @param y: actual result y
       @param tetha: feature params, in this case is a [[theta1], [theta2], ... , [thetaj]]  matrix, eg. [[theta1], [theta2]]
       @param alpha: learning rate, do NOT set learning rate with a too large number for avoiding missing the local peak
       @param iter_times: times for iteration
    """
    theta_vec = []
    cost_func_j_vec = []
    m = X.shape[0]
    for i in range(iter_times):
        # update theta matrix
        theta = update_theta(X, y, theta, alpha, reg_param, i)
        #theta = theta - (alpha/m)*X.T*(X*theta-y)
        #print theta
        theta_vec.append(theta)
        cost_func_j = cost_func(X, y, theta, reg_param)
        cost_func_j_vec.append(cost_func_j)
        #print cost_func_j
    return theta, cost_func_j, theta_vec, cost_func_j_vec

x = np.loadtxt('ex5Linx.dat', delimiter=',')
X = map_feature(x, 5)
Y = np.loadtxt('ex5Liny.dat', delimiter=',')
y = np.zeros((X.shape[0], 1))
y[:,0] = Y
# init theta matrix
coefficient_size = X.shape[1]
theta = np.zeros((coefficient_size, 1))
iter_times = 100000
alpha = 0.003 # NOTICE, DO NOT SET learning rate with a too large number for avoiding missing the local peak
reg_param = 1 # lambda
final_theta, final_cost, theta_vec, cost_j_vec = gradient_desc(X, y, theta, alpha, reg_param, iter_times)
print final_theta.shape
#print final_theta
x_2 = np.linspace(-1.0,1.0,50)
print x_2
mapped_x_2 = map_feature(x_2, 5)
print mapped_x_2.shape
#h = np.dot(mapped_x_2, final_theta)
h = np.dot(mapped_x_2, final_theta)
print h
plt.plot(x_2, h)
plt.plot(x, y, 'ro')
plt.savefig('test_fig4')