import numpy as np

d1 = np.loadtxt("data1.csv", delimiter=",", dtype=np.double)
d2 = np.loadtxt("data1.csv", delimiter=",", dtype=np.double)
d3 = np.loadtxt("data1.csv", delimiter=",", dtype=np.double)
d4 = np.loadtxt("data1.csv", delimiter=",", dtype=np.double)


dataset = np.vstack((d1, d2, d3, d4))

np.savetxt("joined.csv", dataset, delimiter=",")

