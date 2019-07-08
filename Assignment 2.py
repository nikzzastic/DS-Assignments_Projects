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
from sklearn.linear_model import LinearRegression

data = pd.read_csv('Concrete Dataset.csv')
print(data.isnull().sum())

print(type(data))

x = data.iloc[:,0:-1]
y= data.iloc[:,-1]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.22,random_state=39)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)

lr = LinearRegression()

lr.fit(x_train,y_train)
print(lr.score(x_train,y_train)) # 0.6314486028772601

predict=lr.predict(x_test)
print('Predicted -> ',predict)
print('\nActual -> ',y_test)

error = mean_squared_error(predict,y_test)
print(error) # 117.56963288401944
