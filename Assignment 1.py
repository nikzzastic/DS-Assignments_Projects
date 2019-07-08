
#1. Case study 1: Abalone age prediction 

'''
Abalone is a species (of tortoise like animal) which is find nearby waterbodies such as lakes, river or ocean. 
In this dataset abalone physical body measurement are mentioned. In the dataset one of the feature ring is given mean age can
be predicted by counting total number of rings on the body. One has to predict the age of the abalone animal. Here number of 
rings attribute (column) can be considered as age,as the number of rings is the age.Suppose number of rings on the body of 
species is 4 hence age is 4; if rings are 2 the age is 2. In this dataset you have to predict the age. 
Study the details of the case study on UCI repository https://archive.ics.uci.edu/ml/datasets/abalone
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

file = pd.read_csv('abalone.csv',header=None,names=['Gender','Length','Diameter','Height','Whole Weight','Shucked Weight (Meat Weight)','Viscera Weight (After Bleeding)','Shell weight','Rings'])

print(file.head())

#To find if any null value exists
print(file.isnull().sum())

file['Gender'].unique()
#Sex is either Male ("M"), Female ("F") or Infant ("I"),not suitable for regression algorithms, so we create a binary/boolean feature for each of the 3 options:

for label in "MFI":
    file[label] = file["Gender"] == label

x = file.iloc[:,[1,2,3,4,5,6,7,9,10,11]]
print(x.head())

y = file.loc[:,'Rings']
y = y.astype('float')
print(y.head())

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.20,random_state=43)

print(x_train.shape) # (3341, 10)
print(x_test.shape) # (836, 10)


lr = LinearRegression()
lr.coef_

lr.fit(x_train,y_train)
print(lr.score(x_train,y_train))  # 0.5289026155660249

pred = lr.predict(x_test)
print('Predicted -> ',pred)
print('\nActual -> ',y_test)

error = mean_squared_error(pred,y_test)
print(error)  # 4.084093586700131

plt.scatter(pred,y_test)
