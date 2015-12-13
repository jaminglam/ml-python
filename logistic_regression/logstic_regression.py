import numpy as np
import math
import scipy.optimize as opt
def sigmoid_func(z):
    base = 1.+np.exp(-z)
    return 1./base

def cost_func(theta, X, y):
    """ 
    This is the cost function
    @param X: X Matrix
    @param y: actual result y
    @param theta: feature params or called coefficient, in this case is a [[theta1], [theta2], ... , [thetaj]]  matrix, eg. [[theta1], [theta2]]
    """
    m = X.shape[0]
    z = np.dot(X, theta)
    h = sigmoid_func(z)
    J = (float(-1)/m)*(y.T.dot(np.log(h))+(1.-y.T).dot(np.log(1.-h)))
    return J

def grad(theta, X, y):
    """ 
    This is the function to compute the gradient, if X has n columns or coefficient has n features, the gradient should be a (n,1) matrix/n dimension array
    @param X: X Matrix
    @param y: actual result y
    @param theta: feature params or called coefficient, in this case is a [[theta1], [theta2], ... , [thetaj]]  matrix, eg. [[theta1], [theta2]]
    """
    m = X.shape[0]
    z = np.dot(X, theta)
    h = sigmoid_func(z)
    #print X
    gradient = (float(1)/m)*((h-y).T.dot(X))
    #print gradient
    return gradient

def feature_scalling(X):
    """
    seems not used in this example
    """
    var_num = X.shape[1]
    for i in range(var_num):
        if i > 0:
            mean = X[:,i].mean()
            x_max = X[:,i].max()
            x_min = X[:,i].min()
            X[:,i] = (X[:,i] - mean) / (x_max - x_min)
    return X

def score(X, theta):
    """ 
    This is the score function which use the trained formula to determine the result
    @param X: X Matrix
    @param theta: feature params or called coefficient, in this case is a [[theta1], [theta2], ... , [thetaj]]  matrix, eg. [[theta1], [theta2]]
    """
    m = X.shape[0]
    z = np.dot(X, theta)
    h = sigmoid_func(z)
    prediction = []
    print h
    for i in np.nditer(h.T):
        if i >= 0.5:
            prediction.append(1)
        else:
            prediction.append(0)
    print prediction
    return prediction

def train(X, theta, y):
    """ 
    This is the training function, use fmin_bfgs algorithm in scipy.optimize to get the minimize result
    @param X: X Matrix
    @param y: actual result y
    @param theta: feature params or called coefficient, in this case is a [[theta1], [theta2], ... , [thetaj]]  matrix, eg. [[theta1], [theta2]]
    """
    final_theta = opt.fmin_bfgs(cost_func, theta, fprime=grad, args=(X, y))
    return final_theta

def validate(score, y):
    cnt = -1
    true_case = 0
    false_case = 0
    for i in np.nditer(y.T):
        cnt = cnt + 1
        if score[cnt] == i:
            true_case = true_case + 1
        else :
            false_case = false_case + 1
    print true_case
    print false_case
    accuracy = float(true_case) / (false_case+ true_case)
    return accuracy

data = np.loadtxt('ex2testdata1.csv', delimiter=',')
test_data = np.loadtxt('ex2testdata2.csv', delimiter=',')
X = np.ones(data.shape)
X[:,1] = data[:,0]
X[:,2] = data[:,1]
print X
#X = feature_scalling(X)
print X.shape
y = np.zeros((X.shape[0], 1))
y = data[:,2]
print y.shape
theta = np.matrix([[0.], [0.], [0.]])
#theta = 0.1* np.random.randn(3)
print theta.shape
final_theta = train(X, theta, y)
print final_theta
test_X = np.ones(test_data.shape)
test_X[:,1] = test_data[:,0]
test_X[:,2] = test_data[:,1]
test_y = np.zeros((test_X.shape[0], 1))
test_y = test_data[:,2]
prediction = score(test_X, final_theta)
accuracy = validate(prediction, test_y)
print accuracy