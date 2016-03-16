import numpy as np

dataset = np.asmatrix(np.random.rand(150000,50))

np.savetxt("rand_benchmark1.csv", dataset, delimiter=",")