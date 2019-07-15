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

#Now we create a new column called Review which will contain the values of 1,2, and 3. 
#1 - Bad
#2 - Average
#3 - Excellent
#This will be split in the following way. 
#1,2,3 --> Bad
#4,5,6,7 --> Average
#8,9,10 --> Excellent
reviews = []
for i in df['quality']:
    if i >= 1 and i <= 3:
        reviews.append('1')
    elif i >= 4 and i <= 7:
        reviews.append('2')
    elif i >= 8 and i <= 10:
        reviews.append('3')
df['Reviews'] = reviews

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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=.2,random_state=43)
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

lr = LogisticRegression()
lr.fit(train_x,train_y)
pred = lr.predict(test_x)
print(accuracy_score(test_y,pred)) # 0.9845

dt = DecisionTreeClassifier()
dt.fit(train_x,train_y)
pred = dt.predict(test_x)
print(accuracy_score(test_y,pred)) # 0.953125

svc = SVC()
svc.fit(train_x,train_y)
pred = svc.predict(test_x)
print(accuracy_score(test_y,pred)) # 0.9875

rf = RandomForestClassifier()
rf.fit(train_x,train_y)
pred = rf.predict(test_x)
print(accuracy_score(test_y,pred)) # 0.9875

#  RandomForest and SVC both giving same accuracy.

#saving the model
import pickle
f_load = open('wine.obj','wb')
pickle.dump(rf,f_load)
f_load.close()

f_open = open('wine.obj','rb')
res = pickle.load(f_open)
f_open.close()

print(res.score(train_x,train_y)) # 0.9945269741985927

