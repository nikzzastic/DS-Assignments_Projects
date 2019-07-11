
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
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data',names=['Gender','Length','Diameter','Height','Whole Weight','Shucked Weight (Meat Weight)','Viscera Weight (After Bleeding)','Shell weight','Rings'])
data.head()

print(data.isnull().sum()) # o find if there's any null value

# Making all data in same format
from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
data['Gender']=lbl.fit_transform(data['Gender'])

x = data.iloc[:,0:-1]
x.head()

y = data.iloc[:,-1]
y.head()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.23,random_state=54)

print(x_train.shape) #(3216, 7)
print(y_train.shape) #(3216,)
print(x_test.shape) #(961, 7)

lr = LinearRegression()
lr.fit(x_train,y_train)
print(lr.score(x_train,y_train)) # 0.5394496252265788

from sklearn.linear_model import Lasso,Ridge
ls = Lasso()
rd = Ridge()

ls.fit(x_train,y_train)
print(ls.score(x_train,y_train)) 

rd.fit(x_train,y_train)
print(rd.score(x_train,y_train)) # 0.5357349855817581

from sklearn.svm import SVR

list=['linear','poly','rbf']
for i in list:
    sv=SVR(kernel=i)
    sv.fit(x_train,y_train)
    print(i +"->",sv.score(x_train,y_train))  

#linear-> 0.4854197431325158
#poly-> 0.2563368744394211
#rbf-> 0.43880924241218155

from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
rfr = RandomForestRegressor()

rfr.fit(x_train,y_train)
print(rfr.score(x_train,y_train)) # 0.9138163598357825

ada=AdaBoostRegressor()
ada.fit(x_train,y_train)
print(ada.score(x_train,y_train)) # 0.22679435762829303

z = RandomForestRegressor(n_estimators=500,random_state=52)
z.fit(x_train,y_train)
z.score(x_train,y_train) # 0.9412253384632121 (One with best accuracy till now)

pred = z.predict(x_test)
print('Predicted ',pred)
print('Actual ',y_test)

error = mean_squared_error(pred,y_test)
print(error) # 4.959333789802289

plt.scatter(pred,y_test) # Visualizing the data got.

# Now saving the model
import pickle
fh = open('ab_train.obj','wb')
pickle.dump(z,fh)
fh.close()

#Opening the saved model

file_op = open('ab_train.obj','rb')
ob_file = pickle.load(file_op)
file_op.close()

result = ob_file.score(x_train,y_train)
print(result) # 0.9412253384632121
