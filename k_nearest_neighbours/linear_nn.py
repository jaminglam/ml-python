import numpy as np
import time
def linear_serach_nn(dataset, x):
    m = dataset.shape[0]
    min_dist = float("inf")
    for i in range(0, m):
        row = dataset[i, :].A1
        dist = np.sqrt(np.sum((x-row)**2))
        if (dist <= min_dist):
            min_dist = dist
            nearest = row
    return nearest, min_dist

dataset = np.asmatrix(np.genfromtxt('rand_benchmark1.csv', delimiter=','))
start_time = time.clock()
print "start time: %f" %start_time
for i in range(0,10):
    x = np.random.rand(1,50).flatten()
    nearest, min_dist = linear_serach_nn(dataset, x)
    print ("nearest %s min_dist %f" %(nearest, min_dist))
end_time = time.clock()
print "end_time: %f" %end_time
print "process time: %f" %((end_time - start_time)/10)