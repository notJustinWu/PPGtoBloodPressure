import numpy as np

data = np.loadtxt("joined.csv", delimiter=",", dtype=np.double)

sbp = np.array(data[:,3], dtype=np.float64).reshape((-1,1))
dbp = np.array(data[:,4], dtype=np.float64).reshape((-1,1))
ppg = np.array(data[:,19:39], dtype=np.float64)
phy = np.array(data[:,9:18], dtype=np.float64)

features_1 = data[:,0:3]
features_2 = data[:,5:]
features = np.concatenate((features_1, features_2), axis=1, dtype=np.float64)

features_2 = data[:,6:8]
features_3 = data[:,9:15]
all_ppg = np.concatenate((features_2, features_3), axis=1, dtype=np.float64)

all_features = np.hstack((features, sbp, dbp))
ppg_only = np.hstack((ppg, sbp, dbp))
phy_only = np.hstack((phy, sbp, dbp))
all_ppg = np.hstack((all_ppg, ppg, sbp, dbp))

print(all_features.shape, ppg_only.shape, phy_only.shape)

np.savetxt("all_features.csv", all_features, delimiter=",")
np.savetxt("ppg_only_features.csv", ppg_only, delimiter=",")
np.savetxt("phy_only_features.csv", phy_only, delimiter=",")
np.savetxt("all_ppg_features.csv", all_ppg, delimiter=",")
# print(dbp)