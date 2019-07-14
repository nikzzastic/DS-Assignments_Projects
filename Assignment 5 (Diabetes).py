import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../../../../Compressed/pima-indians-diabetes-database.zip')
print(df.info())

print(df.head()) # In data there's error coz Insulin, SkinThickness,BMI etc are having 0 which is not possible

# Replacing 0 with NaN first to find out what suitable value needs to be filled in
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

print(df.head())

print(df.isnull().sum())

df.hist(figsize=(15,15))

# After seeing the graph, mean value will be perfect for Glucose & BloodPressure and median for SkinThickness, BMI and Insulin
df['Glucose'].fillna(df['Glucose'].mean(), inplace = True)
df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace = True)

df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace = True)
df['Insulin'].fillna(df['Insulin'].median(), inplace = True)
df['BMI'].fillna(df['BMI'].median(), inplace = True)

print(df.head())
print(df.info())

df.hist(figsize=(20,20))

sns.countplot(x='Outcome',data=df)

sns.pairplot(df, hue = 'Outcome')

x = df.iloc[:,:-1]
print(x.head())

y = df['Outcome']

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

trainx,testx,trainy,testy = train_test_split(x,y,test_size=.20,random_state=56)

lr = LogisticRegression()
lr.fit(trainx,trainy)
pred = lr.predict(testx)
print(accuracy_score(testy,pred)) # 0.8311688311688312

knn = KNeighborsClassifier()
knn.fit(trainx,trainy)
pred = knn.predict(testx)
print(accuracy_score(testy,pred)) # 0.7337662337662337

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(trainx,trainy)
pred = gnb.predict(testx)
print(accuracy_score(testy,pred)) # 0.7987012987012987

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
rf = RandomForestClassifier()
rf.fit(trainx,trainy)
pred = rf.predict(testx)
print(accuracy_score(testy,pred)) # 0.7987012987012987

gd = GradientBoostingClassifier()
gd.fit(trainx,trainy)
pred = gd.predict(testx)
print(accuracy_score(testy,pred)) # 0.8051948051948052

from sklearn.linear_model import SGDClassifier
sgd =SGDClassifier()
sgd.fit(trainx,trainy)
pred = sgd.predict(testx)
print(accuracy_score(testy,pred)) # 0.7402597402597403

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(trainx,trainy)
pred = dt.predict(testx)
print(accuracy_score(testy,pred)) # 0.7142857142857143

from sklearn.svm import SVC
svc = SVC()
svc.fit(trainx,trainy)
pred = svc.predict(testx)
print(accuracy_score(testy,pred)) # 0.7987012987012987

rf2 = RandomForestClassifier(n_estimators=600,random_state=54)
rf2.fit(trainx,trainy)
pred = rf2.predict(testx)
print(accuracy_score(testy,pred)) # 0.8051948051948052

# Logistic Regression came to give best accuracy.

print(confusion_matrix(testy,pred))

# Saving the model

import pickle
f_save = open('diab.obj','wb')
pickle.dump(lr,f_save)
f_save.close()

f_open = open('diab.obj','rb')
res = pickle.load(f_open)
f_open.close()
