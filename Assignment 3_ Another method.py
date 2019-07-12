# From visualisation of previous way to perform this assignment, i am trying here to make train and test data in approx same way

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split

train = pd.read_csv('C:/../../Compressed/train.csv')
test = pd.read_csv('C:/../../Compressed/test.csv')

# Finding null values in train data
print(train.isnull().sum())

# Finding mean age of train and test data
print(train['Age'].mean())
print(test['Age'].mean())

# Filling Embarked column
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace = True)

# In test dataset there are some empty values in Fare, so applying median there too
test['Fare'].fillna(test['Fare'].median(), inplace = True)

#Cabin column has more than 75% of missing data in both Test & train data so dropping that column
train.drop('Cabin', axis=1, inplace = True)
test.drop('Cabin',axis=1,inplace=True)

#Both the test and train Age column contains more the 15% of missing Data so we are filling it with the median
train['Age'].fillna(train['Age'].median(), inplace = True)
test['Age'].fillna(test['Age'].median(), inplace = True)

#Re-checking if any null value still existed
print(train.isnull().sum())
print(test.isnull().sum())

# Making a copy for both dataset so that we can rollbac if something went wrong
tr = train.copy()
te = test.copy()

# Converting categorical data to numerical
tr = pd.get_dummies(tr,columns=['Sex','Embarked'])
te = pd.get_dummies(te,columns=['Sex','Embarked'])

#Finding the percentage of survived and not-survived
print(tr.Survived.value_counts()/len(train)*100)

# Dropping columns which are not useful
tr.drop(["PassengerId","Name","Ticket"],axis=1,inplace=True)
te.drop(["PassengerId","Name","Ticket"],axis=1,inplace=True)

# Now data training

x = tr.drop("Survived",axis=1)
y = tr["Survived"]

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=.12,random_state=42)
lr = LogisticRegression()
lr.fit(train_x,train_y)
print(lr.score(train_x,train_y)) # 0.7959183673469388

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

list = [GaussianNB(),KNeighborsClassifier(),SVC(),DecisionTreeClassifier(),RandomForestClassifier()]
for i in list:
    z = i
    z.fit(train_x,train_y)
    print(z.score(train_x,train_y))

#0.7908163265306123
#0.8022959183673469
#0.8966836734693877
#0.9783163265306123
#0.9630102040816326

#Came to know that Decision tree gave best score.

dt = DecisionTreeClassifier()
dt.fit(train_x,train_y)
print(dt.score(train_x,train_y)) # 0.9783163265306123

pr = dt.predict(test_x)

fin = dt.predict(te)
print(accuracy_score(test_y,pr)) # 0.7850467289719626
