import numpy as np
from matplotlib import pyplot as plt
import matplotlib

def getFeatures_SBP_DBP(ppg_only = False, phy_only = False, all_ppg = False):
    data = np.loadtxt("all_features.csv", delimiter=",", dtype=np.double)
    if(ppg_only):
        data = np.loadtxt("ppg_only_features.csv", delimiter=",", dtype=np.double)
    elif(phy_only):
        data = np.loadtxt("phy_only_features.csv", delimiter=",", dtype=np.double)
    elif(all_ppg):
        data = np.loadtxt("all_ppg_features.csv", delimiter=",", dtype=np.double)
    
    sbp_column_index = -2
    dbp_column_index = -1

    count = 0
    valid_rows =[]
    for i in range(data.shape[0]):
        row = data[i]
        is_valid = True
        if row[sbp_column_index]<60 or row[dbp_column_index]<40 or row[sbp_column_index]<row[dbp_column_index] or row[sbp_column_index]>200 or row[dbp_column_index]>160:
            is_valid = False
        for col in row:
            if np.isnan(col) or np.isinf(col):
                is_valid = False
                break
        if is_valid:
            valid_rows.append(count)
        count += 1

    cleaned_data = np.zeros((len(valid_rows), data.shape[1]))

    count = 0
    for row in valid_rows:
        cleaned_data[count] = data[row]
        count += 1

    data = np.array(cleaned_data)
    
    sbp = np.array(data[:,sbp_column_index], dtype=np.float64)
    dbp = np.array(data[:,dbp_column_index], dtype=np.float64)

    
    features = data[:,0:data.shape[1]-2]

    hypertensive = np.zeros(sbp.shape[0], dtype=np.int64)
    targets = np.zeros(sbp.shape[0], dtype=np.int64)

    c1 = 0
    c2 = 0
    c3 = 0
    c4 = 0
    c5 = 0

    for i in range(sbp.shape[0]):
        sbp_val = sbp[i]
        dbp_val = dbp[i]
        
        if(sbp_val<90 or dbp_val<60):
            targets[i] = 0
            c1+=1
        elif(sbp_val>=140 or dbp_val>=90):
            targets[i] = 4
            c5+=1
        elif(sbp_val>=130 or dbp_val>=80):
            targets[i] = 3
            c4+=1
        elif(sbp_val>120):
            targets[i] = 2
            c3 += 1
        else:
            targets[i] = 1
            c2 +=1

        if targets[i]>=4:
            hypertensive[i] = 1

    print(c1, c2, c3, c4, c5)

    
    return {
        "features": features,
        "sbp": sbp,
        "dbp": dbp,
        "targets": targets,
        "hypertensive": hypertensive
    }


def normalize(X):
    mu = np.mean(X, 0)
    sigma = np.std(X, 0)
    X = (X-mu) / sigma

    return X

def add_intercept(X):
    X = np.hstack((np.ones((X.shape[0],1)),X))
    return X

def get_labels(sbp, dbp):
    hypertensive = np.zeros(sbp.shape[0], dtype=np.int64)
    targets = np.zeros(sbp.shape[0], dtype=np.int64)

    for i in range(sbp.shape[0]):
        sbp_val = sbp[i]
        dbp_val = dbp[i]
        
        if(sbp_val<90 or dbp_val<60):
            targets[i] = 0
        elif(sbp_val>=140 or dbp_val>=90):
            targets[i] = 4
        elif(sbp_val>=130 or dbp_val>=80):
            targets[i] = 3
        elif(sbp_val>120):
            targets[i] = 2
        else:
            targets[i] = 1

        if targets[i]>=4:
            hypertensive[i] = 1
    
    return (targets, hypertensive)


dt = getFeatures_SBP_DBP(all_ppg=True)
print(dt["features"].shape)