"""
The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, 
the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the 
international community and led to better safety regulations for ships. One of the reasons that the shipwreck led to such loss of life 
was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the
sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class. In this challenge, 
we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of 
machine learning to predict which passengers survived the tragedy. 
"""

#Since code is written in Jupyter Notebook, print statements will be missing at some places

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split

train = pd.read_csv('../../../../train.csv')
test = pd.read_csv('../../../../test.csv')
train.head()

#Checking if any Null Values exist
print(train.isnull().sum())

#Visualizing survived vs not-survived
sns.countplot(x='Survived',data=train)

#Which Gender was more in ship
sns.countplot(x='Sex',data=train)

#Finding count of female and male persons
train[train['Sex']=='female'].count()
train[train['Sex']=='male'].count()

#Checking which class had max population
sns.countplot(x='Pclass',data=train)

#People of which class survived more
sns.countplot(data=train,hue='Pclass',x='Survived')

#Which Gender survived more
sns.countplot(x='Survived',hue='Sex',data=train)

#Getting statistical details about data
train.describe()

train2 = train.copy()
#Preparing data now

#First filling mean age to null values of Age column
train2['Age']=train2['Age'].fillna((int(train2['Age'].mean())))

#Since Cabin column has very high quantity of null values and also it doesn't seems that important, so dropping that column
train2.drop('Cabin',axis=1,inplace=True)

#Embarked column has only 2 null values , so going for mode we got "S" is most occured
train2['Embarked'].mode()
train2['Embarked'].fillna("s", inplace = True) 

print(train2.isnull().sum()) # Got that no null value exists now

#Changing data to numeric one; Changing column-> Sex,PClass and Embarked
sex = pd.get_dummies(train2["Sex"])
embarked = pd.get_dummies(train2["Embarked"])
pclass = pd.get_dummies(train2["Pclass"])

#Concating new data
train2=pd.concat([train2,pclass,sex,embarked],axis=1)

#Dropping not much-useful columns 
train2.drop(["PassengerId","Pclass","Name","Sex","Ticket","Embarked"],axis=1,inplace=True)

#Our data is ready now. Starting model training

x = train2.drop("Survived",axis=1)
y = train2["Survived"]

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=.2,random_state=52)
lg=LogisticRegression()
lg.fit(train_x,train_y)
print(lg.score(train_x,train_y)) # 0.8061797752808989

# Trying with other models
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

gnb = GaussianNB()
gnb.fit(train_x,train_y)
print(gnb.score(train_x,train_y)) # 0.7865168539325843

knn = KNeighborsClassifier()
knn.fit(train_x,train_y)
print(knn.score(train_x,train_y)) # 0.800561797752809

svc = SVC()
svc.fit(train_x,train_y)
print(svc.score(train_x,train_y)) # 0.9002808988764045

rf = RandomForestClassifier()
rf.fit(train_x,train_y)
print(rf.score(train_x,train_y)) # 0.9691011235955056

dt = DecisionTreeClassifier()
dt.fit(train_x,train_y)
print(dt.score(train_x,train_y)) # 0.9845505617977528 The best we got still

pred = dt.predict(test_x)
print(pred)

print(accuracy_score(test_y,pred)) # 0.7541899441340782
print(confusion_matrix(test_y,pred))

# Saving this model
import pickle

fo = open('ti.obj','wb')
pickle.dump(dt,fo)
fo.close()

fl = open('ti.obj','rb')
res = pickle.load(fl)
fl.close()

print(res.score(train_x,train_y)) # 0.9845505617977528





