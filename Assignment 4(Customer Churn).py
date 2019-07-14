'''
Customer churn is when a company's customers stop doing business with that company. Businesses are very keen on measuring churn because 
keeping an existing customer is far less expensive than acquiring a new customer. New business involves working leads through a sales 
funnel, using marketing and sales budgets to gain additional customers. Existing customers will often have a higher volume of service 
consumption and can generate additional customer referrals. 
Customer retention can be achieved with good customer service and products. But the most effective way for a company to prevent attrition 
of customers is to truly know them. The vast volumes of data collected about customers can be used to build churn prediction models. 
Knowing who is most likely to defect means that a company can prioritise focused marketing efforts on that subset of their customer base.
Preventing customer churn is critically important to the telecommunications sector, as the barriers to entry for switching services are so
low. You will examine customer data from IBM Sample Data Sets with the aim of building and comparing several customer churn prediction models.
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Telecom_customer_churn.csv')

print(df.isnull().sum()) #Nothing found

print(df.head())

df['Churn'].value_counts().plot('bar').set_title('Churned')

df['tenure'].hist(bins=30)

df['MonthlyCharges'].hist(bins=30)

fig, ax = plt.subplots(1, 2, figsize=(14, 4))
df[df.Churn == 'No']['Contract'].value_counts().plot('bar', ax=ax[0]).set_title('Not Churned')
df[df.Churn == 'Yes']['Contract'].value_counts().plot('bar', ax=ax[1]).set_title('Churned')
plt.show()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

col = ['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','PaymentMethod','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Churn','PaperlessBilling','Contract']
for i in col:
    df[i] = le.fit_transform(df[i])
    
    
print(df.info()) # There's acolumn TotalPrice having obect value. Later found on it was actually blank values which idicates no money was there, so filling 0 there.

df[df['TotalCharges'].isnull()].index.tolist() # To find all index having no value (blank value) in Total Price

#df['TotalCharges'] = pd.to_numeric(df.TotalCharges.str.replace('[^\d.]', ''), errors='coerce')

df['TotalCharges'] = df['TotalCharges'].fillna(0)

churn_rate = df['Churn'].value_counts()
print(churn_rate)
df['Churn'].value_counts().plot('bar').set_title('Churned')

'''
0    5174
1    1869
Name: Churn, dtype: int64
'''

x = df.iloc[:,1:20]
print(x.head())

y = df.iloc[:,-1]

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=.25,random_state=45)

lr = LogisticRegression()
lr.fit(train_x,train_y)
pred = lr.predict(test_x)
print(accuracy_score(test_y,pred)) # 0.8131743327654741
print(confusion_matrix(test_y,pred))

from sklearn.metrics import roc_curve

y_pred_prob = lr.predict_proba(test_x)[:,1]
fpr,tpr,threshold = roc_curve(test_y,y_pred_prob)

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression')
plt.show()

from sklearn.metrics import roc_auc_score

score = roc_auc_score(test_y,lr.predict(test_x))
print(score) # 0.7375736882729053

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(train_x,train_y)
pred = gnb.predict(test_x)
print(accuracy_score(test_y,pred)) # 0.7541169789892107
print(confusion_matrix(test_y,pred))

y_pred_prob = gnb.predict_proba(test_x)[:,1]
fpr,tpr,threshold = roc_curve(test_y,y_pred_prob)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='Gaussian NB')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Gaussian NB')
plt.show()

from sklearn.svm import SVC
svc = SVC()
svc.fit(train_x,train_y)
pred = svc.predict(test_x)
print(accuracy_score(test_y,pred)) # 0.7830777967064169
print(confusion_matrix(test_y,pred))

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(train_x,train_y)
pred = knn.predict(test_x)
print(accuracy_score(test_y,pred)) # 0.7677455990914254
print(confusion_matrix(test_y,pred))

y_pred_prob = knn.predict_proba(test_x)[:,1]
fpr,tpr,threshold = roc_curve(test_y,y_pred_prob)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNeighborsClassifier')
plt.show()

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(train_x,train_y)
pred = dt.predict(test_x)
print(accuracy_score(test_y,pred)) 0.7382169222032936
print(confusion_matrix(test_y,pred))

y_pred_prob = dt.predict_proba(test_x)[:,1]
fpr,tpr,threshold = roc_curve(test_y,y_pred_prob)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='DecisionTreeClassifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('DecisionTreeClassifier')
plt.show()

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(train_x,train_y)
pred = rf.predict(test_x)
print(accuracy_score(test_y,pred)) # 0.7819420783645656
print(confusion_matrix(test_y,pred))

y_pred_prob = rf.predict_proba(test_x)[:,1]
fpr,tpr,threshold = roc_curve(test_y,y_pred_prob)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='RandomForestClassifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RandomForestClassifier')
plt.show()

from sklearn.ensemble import GradientBoostingClassifier
gd = GradientBoostingClassifier()
gd.fit(train_x,train_y)
pred = gd.predict(test_x)
print(accuracy_score(test_y,pred)) # 0.8097671777399205

y_pred_prob = gd.predict_proba(test_x)[:,1]
fpr,tpr,threshold = roc_curve(test_y,y_pred_prob)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='GradientBoostingClassifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('GradientBoostingClassifier')
plt.show()

from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier()
sgd.fit(train_x,train_y)

pred = gd.predict(test_x)
print(accuracy_score(test_y,pred)) # 0.8097671777399205
print(confusion_matrix(test_y,pred))

y_pred_prob = gd.predict_proba(test_x)[:,1]
fpr,tpr,threshold = roc_curve(test_y,y_pred_prob)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='SGDClassifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SGDClassifier')
plt.show()

# Till now Logistic Regression gave best accuracy. Saving the model now
import pickle
fl = open('ab.obj','wb')
pickle.dump(lr,fl)
fl.close()

fo = open('ab.obj','rb')
res = pickle.load(fo)
fo.close()

print(res.score(train_x,train_y))
# 0.8023475956077244
