#In this project explore wine dataset to assess red wine quality. The objective of this data science project is to explore which 
#chemical properties will influence the quality of red wines.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Wine Quality.csv')

print(df.isnull().sum())
print(df.describe())

sns.boxplot(x='quality',y='volatile acidity',data=df) # So, when quality is increasing, volatile acidity is decreasing

sns.barplot(x='quality',y='fixed acidity',data=df) # Relation between quality and fixed acidity isn't giving much clarification

sns.barplot(x='quality',y='citric acid',data=df) #When quality is increased, quantity of ciric acid also increases

sns.barplot(x='quality',y='residual sugar',data=df) #residual sugar vs quantity isnt giving much understanding

sns.barplot(x='quality',y='chlorides',data=df) # as quality increases, chlorine amount in wine gets decreasd

sns.barplot(x='quality',y='free sulfur dioxide',data=df) # Not any clearification

sns.barplot(x='quality',y='total sulfur dioxide',data=df) # Not any clearification

sns.barplot(x='quality',y='sulphates',data=df) # Sulphate amount increases as quality is increased

sns.barplot(x='quality',y='alcohol',data=df) # Alcohol amount also increases as going higher in quality

sns.countplot('quality',data=df) # So 5 is the highest quality most of combinations got followed by 6

print(df['quality'].value_counts()) # Count of each quality column

# Training model
x = df.iloc[:,0:11]
y = df.iloc[:,-1]

from sklearn.preprocessing import StandardScaler
st = StandardScaler()
x = st.fit_transform(x)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=.2,random_state=50)
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

lr = LogisticRegression()
lr.fit(train_x,train_y)
pred = lr.predict(test_x)
print(accuracy_score(test_y,pred)) # 0.596875

dt = DecisionTreeClassifier()
dt.fit(train_x,train_y)
pred = dt.predict(test_x)
print(accuracy_score(test_y,pred)) # 0.671875

knn = KNeighborsClassifier()
knn.fit(train_x,train_y)
pred = knn.predict(test_x)
print(accuracy_score(test_y,pred)) # 0.590625

svc = SVC()
svc.fit(train_x,train_y)
pred = svc.predict(test_x)
print(accuracy_score(test_y,pred)) # 0.634375

rf = RandomForestClassifier()
rf.fit(train_x,train_y)
pred = rf.predict(test_x)
print(accuracy_score(test_y,pred)) # 0.715625

gd = GradientBoostingClassifier()
gd.fit(train_x,train_y)
pred = gd.predict(test_x)
print(accuracy_score(test_y,pred)) # 0.671875

gnb = GaussianNB()
gnb.fit(train_x,train_y)
pred = gnb.predict(test_x)
print(accuracy_score(test_y,pred)) # 0.565625

sgdc = SGDClassifier()
sgdc.fit(train_x,train_y)
pred = sgdc.predict(test_x)
print(accuracy_score(test_y,pred)) # 0.496875

from sklearn.model_selection import GridSearchCV
param = {
    'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
    'kernel':['linear', 'rbf'],
    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
}
grid_svc = GridSearchCV(svc, param_grid=param, scoring='accuracy', cv=10)
grid_svc.fit(train_x,train_y)

print(grid_svc.best_params_) # {'C': 1.4, 'gamma': 0.8, 'kernel': 'rbf'}

svc2 = SVC(C = 1.4, gamma =  0.8, kernel= 'rbf')
svc2.fit(train_x,train_y)
pred_svc2 = svc2.predict(test_x)
print(accuracy_score(test_y,pred_svc2)) # 0.715625

# So far RandomForest and Grid Search gave best accuracy, ie 71.5% . Lets convert quality column as binary classification: good or bad
new_df = df.copy()

print(new_df.isnull().sum())

#Dividing wine as good and bad by giving the limit for the quality
bins = (2, 6.5, 8)
group = ['bad', 'good']
new_df['quality'] = pd.cut(new_df['quality'], bins = bins, labels = group)

#Now lets assign a labels to our quality variable
from sklearn.preprocessing import LabelEncoder
label_quality = LabelEncoder()

#Bad becomes 0 and good becomes 1 
new_df['quality'] = label_quality.fit_transform(new_df['quality'])
print(new_df.head())

print(new_df['quality'].value_counts())

sns.countplot(new_df['quality'])

a = new_df.iloc[:,:11]
b = new_df.iloc[:,-1]

#Applying Standard scaling to get optimized result
a = st.fit_transform(a)

X_train, X_test, Y_train, Y_test = train_test_split(a,b, test_size = 0.2, random_state = 43)

#Logistic Regression
lr.fit(X_train,Y_train)
pre = lr.predict(X_test)
print(accuracy_score(Y_test,pre)) # 0.890625

#DecisionTreeClassifier
dt.fit(X_train,Y_train)
pre = dt.predict(X_test)
print(accuracy_score(Y_test,pre)) # 0.875

#KNeighborsClassifier
knn.fit(X_train,Y_train)
pre = knn.predict(X_test)
print(accuracy_score(Y_test,pre)) # 0.8875

#GaussianNB
gnb.fit(X_train,Y_train)
pre = gnb.predict(X_test)
print(accuracy_score(Y_test,pre)) # 0.8375

#AdaBoostClassifier
ad.fit(X_train,Y_train)
pre = ad.predict(X_test)
print(accuracy_score(Y_test,pre)) # 0.890625

#SVC
svc.fit(X_train,Y_train)
pre = svc.predict(X_test)
print(accuracy_score(Y_test,pre)) # 0.9

#RandomForestClassifier
rf.fit(X_train,Y_train)
pre = rf.predict(X_test)
print(accuracy_score(Y_test,pre)) # 0.9125

#Finding best parameters for our SVC model
param = {
    'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
    'kernel':['linear', 'rbf'],
    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
}
grid_svc = GridSearchCV(svc, param_grid=param, scoring='accuracy', cv=10)

grid_svc.fit(X_train, Y_train)

svc2 = SVC(C = 1.2, gamma =  0.9, kernel= 'rbf')
svc2.fit(X_train, Y_train)
pred_svc2 = svc2.predict(X_test)
print(accuracy_score(Y_test, pred_svc2)) # 0.909375

# So, RandomForest is giving highest accuracy till now

# Saving the model
import pickle
f_load = open('wine.obj','wb')
pickle.dump(rf,f_load)
f_load.close()

f_open = open('wine.obj','rb')
res = pickle.load(f_open)
f_open.close()

print(res.score(X_train,Y_train)) # 0.9913995308835027


