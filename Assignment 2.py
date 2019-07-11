#Case Study 2: Concrete dataset 

'''
Concrete dataset is related to the details of added ingredients that is used for making building, houses and other household 
structures. One has to detect the amount of composition of various ingredients to give good concrete strength. 
Predict the CMS (Concrete compressive strength) 
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression,Lasso,Ridge

data = pd.read_csv('Concrete Dataset.csv')
print(data.isnull().sum())

print(type(data)) # <class 'pandas.core.frame.DataFrame'>

x = data.iloc[:,0:-1]
y= data.iloc[:,-1]
x.head()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.22,random_state=39)

print(x_train.shape) # (803, 8)
print(x_test.shape) # (227, 8)

lr = LinearRegression()
lr.fit(x_train,y_train)

print(lr.score(x_train,y_train)) # 0.6314486028772601

ls = Lasso()
ls.fit(x_train,y_train)
print(ls.score(x_train,y_train)) # 0.6312422638786822

rd = Ridge()
rd.fit(x_train,y_train)
print(rd.score(x_train,y_train)) # 0.6314486028431515

from sklearn.svm import SVR
re = SVR(kernel='linear')
re.fit(x_train,y_train)
print(re.score(x_train,y_train)) # 0.5800858827699276

re2 = SVR(kernel='poly')
re2.fit(x_train,y_train)
print(re2.score(x_train,y_train))

re3 = SVR(kernel='rbf')
re3.fit(x_train,y_train)
print(re3.score(x_train,y_train)) # 0.114935717860293

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()

rfr.fit(x_train,y_train)
print(rfr.score(x_train,y_train)) # 0.9772550347123613


import pickle
fd = open('conc.obj','wb')
pickle.dump(rfr,fd)
fd.close()

fl = open('conc.obj','rb')
model = pickle.load(fl)
fl.close()

result = model.score(x_train,y_train)
print(result) # 0.9772550347123613
