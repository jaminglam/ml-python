import numpy as np

class Perceptron:

    def __init__(self, alpha):
        self.alpha = alpha # learning rate
    def classify(self, x):
        return np.dot(self.w, x)+self.b


    def is_error_point(self, x, y):
        return self.classify(x)*y<=0

    def update_param(self, x, y):
        self.w = self.w + self.alpha*y*x
        self.b = self.b + self.alpha*y

    def train(self, x_, y_):
        train_size = x_.shape[0]
        feature_size = x_.shape[1]
        is_all_sep = False
        iterations = 0
        self.w = np.zeros(feature_size)
        self.b = 0
        while is_all_sep == False:
            is_all_sep = True
            iterations = iterations + 1
            for i in range(0, train_size):
                x_2d = x_[i,:]
                print x_2d
                print x_2d.shape
                x = x_2d.flatten()
                print "x: "
                print x
                print "x shape: "
                print x.shape
                y = y_[i]
                if self.is_error_point(x, y):
                    is_all_sep = False
                    self.update_param(x, y)
                    break
            if iterations > 100000:
                break

    def print_weight(self):
        print "weight: "
        print self.w
    def print_bias(self):
        print "bias: "
        print self.b

if __name__ == "__main__":
    #init x_ y_
    x_ = np.matrix([[1.,1.,1.],[-4.,0.,0.5],[-1.,-0.5,-1.]])
    y_ = np.array([1.,-1.,-1.])
    test = np.matrix([[1,2,3]])
    print test
    print test.shape
    print test.flatten().shape
    perceptron = Perceptron(0.3)
    perceptron.train(x_, y_)
    perceptron.print_weight()
    perceptron.print_bias()
    print perceptron.classify(x_[0,:])
    print perceptron.classify(x_[1,:])
    print perceptron.classify(x_[2,:])

