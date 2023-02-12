import numpy as np

phy = np.loadtxt("phy1.csv", delimiter=",", dtype=np.double)
ppg = np.loadtxt("ppg1.csv", delimiter=",", dtype=np.double)
ptt = np.loadtxt("ptt_newpart1.csv", delimiter=",", dtype=np.double).reshape((-1,1))
seven = np.loadtxt("seven1.csv", delimiter=",", dtype=np.double)


dataset = np.zeros((phy.shape[0], phy.shape[1]+ppg.shape[1]+ptt.shape[1]+seven.shape[1]-1))

for i in range(phy.shape[0]):
    ind = int(phy[i][0] - 9001)
    d1 = seven[ind]
    d2 = phy[i][1:]
    d3 = ptt[ind]
    d4 = ppg[ind]

    d = np.concatenate((d1, d2, d3, d4), axis=None)
    dataset[i] = d

np.savetxt("data1.csv", dataset, delimiter=",")


