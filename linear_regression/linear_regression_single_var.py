import numpy as np

def cost_func(X, y, theta):
	m = X.shape[0]
	prediction = X*theta # X is a N*2 Matrix, theta is a 2*1 Matrix, prediction is a N*1 Matrix
	cost_func_J = (1./(2*m)) * (prediction - y).T * (prediction-y) # y is N * 1 Matrix
	return cost_func_J

def update_theta(X, y, theta, alpha):
	""" 
       @param X: X Matrix
       @param y: actual result y
       @param tetha: feature params, in this case is a [[theta1], [theta2], ... , [thetaj]]  matrix, eg. [[theta1], [theta2]]
       @param alpha: learning rate, NOTICE, 
    """
	m = X.shape[0]
	prediction = X*theta
	new_theta = theta - (alpha/m)*X.T*(prediction - y)
	return new_theta

def gradient_desc(X, y, theta, alpha, iter_times):
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
        theta = update_theta(X, y, theta, alpha)
        #theta = theta - (alpha/m)*X.T*(X*theta-y)
        #print theta
        theta_vec.append(theta)
        cost_func_j = cost_func(X, y, theta)
        cost_func_j_vec.append(cost_func_j)
        #print cost_func_j
    return theta, cost_func_j, theta_vec, cost_func_j_vec

#Load the dataset, data is a N*2 Matrix
data = np.loadtxt('testdata2.csv', delimiter=',')
# init X matrix, in this case, it should be a N*2 Matrix
X = np.ones(data.shape)
print X.shape
X[:,1] = data[:,0]
# init destination result y matrix, it should be a N*1 Matrix, NOT an array
y = np.zeros((X.shape[0], 1))
y[:,0] = data[:,1]
# init theta matrix, NOTICE, it is a MATRIX, NOT an array, shape is (2,1)
theta = np.matrix([[0.], [0.]])
iter_times = 100000
alpha = 0.003 # NOTICE, DO NOT SET learning rate with a too large number for avoiding missing the local peak
final_theta, final_cost, theta_vec, cost_j_vec = gradient_desc(X, y, theta, alpha, iter_times)
print final_theta
