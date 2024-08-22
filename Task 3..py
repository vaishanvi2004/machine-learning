#!/usr/bin/env python
# coding: utf-8

# Task 3 : Customer Churn Prediction

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import LabelEncoder


# In[4]:


ccp=pd.read_csv(r'C:\Users\HP\Downloads\Churn_Modelling.csv')


# In[5]:


ccp.head()


# In[6]:


ccp.info()


# In[7]:


ccp[ccp.duplicated()].shape[0]


# In[8]:


ccp.dtypes


# In[9]:


ccp.tail()


# In[10]:


ccp.columns


# In[11]:


ccp=ccp.drop(['RowNumber', 'CustomerId', 'Surname'],axis=1)


# In[12]:


ccp['Geography'].unique()


# In[13]:


#there are two methods two convert string to numeric
#either use replace() or labelencoder()
ccp['Geography']=ccp['Geography'].replace(['France'],'0')
ccp['Geography']=ccp['Geography'].replace(['Spain'],'1')
ccp['Geography']=ccp['Geography'].replace(['Germany'],'2')
ccp['Gender']=ccp['Gender'].replace(['Female'],'0')
ccp['Gender']=ccp['Gender'].replace(['Male'],'1')


# In[14]:


ccp.head()


# In[15]:


ccp['Geography']=pd.to_numeric(ccp['Geography'])
ccp['Gender']=pd.to_numeric(ccp['Gender'])


# In[16]:


encoder=LabelEncoder()
ccp['Geography']=encoder.fit_transform(ccp['Geography'])
ccp['Gender']=encoder.fit_transform(ccp['Gender'])


# In[17]:


ccp.head()


# In[18]:


ccp.dtypes


# In[19]:


ccp['Exited'].value_counts()


# In[20]:


plt.subplot(121)
sns.countplot(data=ccp,x="Exited")
plt.title("Distribution of  Exited")
plt.show()


# In[21]:


plt.figure(figsize=(10,5))
plt.pie(ccp['Exited'].value_counts(),autopct="%.1f%%",labels=['yes','no'])
plt.show()


# In[22]:


X=ccp.drop('Geography',axis=1)
y=ccp['Geography']


# In[23]:


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[24]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[25]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print("confusion_matrix:\n",confusion_matrix(y_test,y_pred))
print("classification report:\n",classification_report(y_test,y_pred))
print("Accuracy:",accuracy_score(y_test,y_pred))


# In[26]:


from sklearn.ensemble import RandomForestClassifier
rfc_model=RandomForestClassifier()
rfc_model.fit(X_train,y_train)
y_pred=rfc_model.predict(X_test)
print("confusion_matrix:\n",confusion_matrix(y_test,y_pred))
print("classification report:\n",classification_report(y_test,y_pred))
print("Accuracy Score:",rfc_model.score(X_test, y_test))


# In[ ]:

#please check if xgboost is install or not otherwise it will not run
pip install xgboost


# In[27]:


import xgboost as xgb


# In[28]:


from xgboost import XGBClassifier
xgb_model= XGBClassifier()
xgb_model.fit(X_train,y_train)
y_pred=xgb_model.predict(X_test)
print("confusion_matrix:\n",confusion_matrix(y_test,y_pred))
print("classification report:\n",classification_report(y_test,y_pred))
print("Accuracy Score:",xgb_model.score(X_test, y_test))


# In[30]:


results=pd.DataFrame({
    'model':['Random Forest','Logistic Regression','XGBoost'],
    'score':[0.547,0.4985,0.5305]})
result_spam=results.sort_values(by='score',ascending=False)
result_spam=result_spam.set_index('score')
result_spam.head()


# In[ ]:




