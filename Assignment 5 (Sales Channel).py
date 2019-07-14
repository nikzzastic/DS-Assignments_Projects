'''
Sales Channel Prediction Case Study 
When a company enters a market, the distribution strategy and channel it uses are keys to its success in the market, as well 
as market know-how and customer knowledge and understanding. Because an effective distribution strategy under efficient supply-chain 
management opens doors for attaining competitive advantage and strong brand equity in the market, it is a component of the marketing mix 
that cannot be ignored . The distribution strategy and the channel design have to be right the first time. The case study of Sales channel
includes the detailed study of TV, radio and newspaper channel. The company has to select proper sales channel to predict the right sales
 channel to generate good revenue. 
 '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Advertise.csv',index_col=0)
print(df.head())

print(df.isnull().sum())

sns.pairplot(df, x_vars=['TV', 'radio', 'newspaper'], y_vars='sales', kind='reg',size = 8)

sns.pairplot(df)

sns.stripplot(x='TV',y='sales',data=df)

df['sales'].hist(bins=30)

df['TV'].hist(bins=30)

df.plot.hist(subplots=True, layout=(2,2), figsize=(10, 10), bins=20)

x = df.iloc[:,:3]
y = df['sales']

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=.25,random_state=47)

lr = LinearRegression()
lr.fit(train_x,train_y)
print(lr.score(train_x,train_y)) # 0.9115017572364001

pred = lr.predict(test_x)
print(r2_score(test_y,pred)) # 0.839164865048828

#Mean of Residuals
#Residuals as we know are the differences between the true value and the predicted value. One of the assumptions of linear 
#regression is that the mean of the residuals should be zero.

resi = test_y.values-pred
mean_res = np.mean(resi)
print(mean_res)
# -0.05688530949133067   ...Very close to zero so all good here.

# Trying other algorithm

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf.fit(train_x,train_y)
pred = rf.predict(test_x)
print(rf.score(train_x,train_y)) # 0.9950606191640787
print(r2_score(test_y,pred)) # 0.9516927233659064
