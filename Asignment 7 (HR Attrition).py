'''
Every year a lot of companies hire a number of employees. The companies invest time and money in training those employees, 
not just this but there are training programs within the companies for their existing employees as well. The aim of these programs 
is to increase the effectiveness of their employees. But where HR Analytics fit in this? and i s it just about improving the performance 
of employees? 

HR Analytics Human resource analytics (HR analytics) is an area in the field of analytics that refers to applying analytic processes to the
human resource department of an organization in the hope of improving employee performance and therefore getting a better return on 
investment. HR analytics does not just deal with gathering data on employee efficiency. Instead, it aims to provide insight into each
process by gathering data and then using it to make relevant decisions about how to improve these processes. 
Attrition in HR Attrition in human resources refers to the gradual loss of employees over time. In general, relatively high attrition is 
problematic for companies. HR professionals often assume a leadership role in designing company compensation programs, work culture and
motivation systems that help the organization retain top employees. 
How does Attrition affect companies? and how does HR Analytics help in analyzing attrition? We will discuss the first question here and
for the second question we will write the code and try to understand the process step by step. 
Attrition affecting Companies A major problem in high employee attrition is its cost to an organization. Job postings, hiring processes,
paperwork and new hire training are some of the common expenses of losing employees and replacing them. Additionally, regular employee 
turnover prohibits your organization from increasing its collective knowledge base and experience over time. This is especially
concerningif your business is customer facing, as customers often prefer to interact with familiar people. Errors and issues are more 
likely ifyou constantly have new workers. 
  '''
  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/../../../ibm-hr-analytics-employee-attrition-performance.zip')

print(df.head())
print(df.info())
print(df.isnull().sum())

plt.bar(df['Age'].values,df['TotalWorkingYears'].values)
plt.xlabel('Age')
plt.ylabel('Total Working Years')
plt.show()

plt.figure(figsize=(16,4))
sns.boxplot('Age','DailyRate',data=df)
plt.show()

sns.swarmplot('YearsInCurrentRole','Age',data=df)

plt.figure(figsize=(15,15))
sns.factorplot('Age','YearsAtCompany','JobSatisfaction',data=df,kind='bar',aspect=4)

sns.factorplot('WorkLifeBalance','JobSatisfaction',data=df,kind='bar')

plt.figure(figsize=(15,15))
sns.heatmap(df.corr())
plt.show()

#Removing strongly correlated variables
data = df[['Age','DailyRate','DistanceFromHome', 
                       'EnvironmentSatisfaction', 'HourlyRate',                     
                       'JobInvolvement', 'JobLevel',
                       'JobSatisfaction', 
                       'RelationshipSatisfaction', 
                       'StockOptionLevel',
                        'TrainingTimesLastYear']].copy()
print(data.head())

#Categoriacal data in seperate dataframe for changing it to numerical data
categ_data = df[['Attrition', 'BusinessTravel','Department',
                       'EducationField','Gender','JobRole',
                       'MaritalStatus',
                       'Over18', 'OverTime']].copy()
print(categ_data.head())

from sklearn.preprocessing import LabelEncoder
oh = LabelEncoder()
categ_data['Attrition'] = oh.fit_transform(categ_data['Attrition'])
print(categ_data['Attrition'].value_counts())
'''
Output:
0    1233
1     237
Name: Attrition, dtype: int64
'''

print(df['Attrition'].value_counts())
'''
Output
No     1233
Yes     237
Name: Attrition, dtype: int64
'''

categ_data = pd.get_dummies(categ_data)
print(categ_data.head())

# Merging final numerical data
final = pd.concat([data,categ_data],axis=1)
print(final.head())

x=final.drop('Attrition', axis = 1)
y=final['Attrition']

print(x.head())

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=.15,random_state=26)

svc = SVC()
svc.fit(train_x,train_y)
pred1 = svc.predict(test_x)
print(accuracy_score(test_y,pred1)) # 0.8868778280542986

dt = DecisionTreeClassifier()
dt.fit(train_x,train_y)
pred2 = dt.predict(test_x)
print(accuracy_score(test_y,pred2)) # 0.7918552036199095

gd = GradientBoostingClassifier()
gd.fit(train_x,train_y)
pred3 = gd.predict(test_x)
print(accuracy_score(test_y,pred3)) # 0.9230769230769231

rf = RandomForestClassifier()
rf.fit(train_x,train_y)
pred4 = rf.predict(test_x)
print(accuracy_score(test_y,pred4)) # 0.8914027149321267

# Gradient Boosting is giving good accuracy . Saving the model now

import pickle

dump = open('hr.obj','wb')
pickle.dump(gd,dump)
dump.close()

load = open('hr.obj','rb')
res = pickle.load(load)
load.close()

print(res.score(train_x,train_y)) # 0.9351481184947958

print(accuracy_score(test_y,pred3)) # 0.9230769230769231

print(confusion_matrix(test_y,pred3)) 
'''
Output:
[[193   3]
 [ 14  11]]
 
 '''
print(classification_report(test_y,pred3))
'''
Output:

             precision    recall  f1-score   support

          0       0.93      0.98      0.96       196
          1       0.79      0.44      0.56        25

avg / total       0.92      0.92      0.91       221

'''

