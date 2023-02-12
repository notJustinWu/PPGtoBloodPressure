import pandas as pd 
import numpy as np  
import torch
import torch.nn as nn
import pandas as pd                    
import matplotlib.pyplot as plt        
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from sklearn import metrics  
from sklearn import datasets
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression  
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import csv

dataset = pd.read_csv('all_cleaned.csv', names = ['a', 'b','c','d','e','f','g','h','i','j', 'k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','aa','ab','sbp','dbp'])
params = dataset[['a', 'b','c','d','e','f','g','h','i','j', 'k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','aa','ab']]
map = list()
dbp = list()
sbp = list()
ss=StandardScaler()

with open('all_cleaned.csv', 'r') as csvfile:
	parse = csv.reader(csvfile, delimiter = ',')
	for row in parse:
		map.append((2*float(row[-1])+float(row[-2]))/3)
		dbp.append(float(row[-1]))
		sbp.append(float(row[-2]))

#x_train, x_test, y_train, y_test = train_test_split(params, map, test_size=0.2, shuffle = False, stratify = None) 
#x_train, x_test, y_train, y_test = train_test_split(params, dbp, test_size=0.2, shuffle = False, stratify = None) 
x_train, x_test, y_train, y_test = train_test_split(params, dbp, test_size=0.2, stratify = None) 

reg = LinearRegression()
reg.fit(x_train, y_train)
y_pred1 = reg.predict(x_train)
y_pred2 = reg.predict(x_test)

MAE1 = metrics.mean_absolute_error(y_train, y_pred1)
MAE2 = metrics.mean_absolute_error(y_test, y_pred2)


print('MAE train', MAE1)  
print('MAE test', MAE2)  