#!/usr/bin/env python
# coding: utf-8

# Task 2: Credit Card Fraud Detection

# In[29]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import LabelEncoder


# In[30]:


CCF=pd.read_csv(r'C:\Users\HP\Documents\ml\fraudTest.csv\fraudTest.csv')


# In[31]:


CCF.head()


# In[32]:


CCF.columns


# In[33]:


CCF.describe()


# In[34]:


CCF.info()


# In[35]:


CCF.shape


# In[38]:


CCF.tail()


# In[39]:


CCF['category'].unique()


# In[40]:


encoder=LabelEncoder()
CCF['merchant']=encoder.fit_transform(CCF['merchant'])
CCF['category']=encoder.fit_transform(CCF['category'])
CCF['street']=encoder.fit_transform(CCF['street'])
CCF['job']=encoder.fit_transform(CCF['job'])
CCF['trans_num']=encoder.fit_transform(CCF['trans_num'])
CCF['first']=encoder.fit_transform(CCF['first'])
CCF['city']=encoder.fit_transform(CCF['city'])
CCF['last']=encoder.fit_transform(CCF['last'])
CCF['gender']=encoder.fit_transform(CCF['gender'])
CCF['state']=encoder.fit_transform(CCF['state'])
CCF['dob']=encoder.fit_transform(CCF['dob'])
CCF['trans_date_trans_time']=encoder.fit_transform(CCF['trans_date_trans_time'])


# In[41]:


CCF.head()


# In[42]:


CCF.duplicated().any()


# In[43]:


data=CCF.head(n=15000)
CCF['is_fraud'].value_counts()


# In[44]:


CCF.head()


# In[45]:


sns.countplot(x='is_fraud',data=CCF)
plt.title('fraud and non-fraud')
plt.xlabel('0:non fraud ,1:fraud')
plt.ylabel('count')
plt.show()


# In[46]:


X=CCF.drop('is_fraud',axis=1)
y=CCF['is_fraud']


# In[47]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[48]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[49]:


from sklearn.linear_model import LogisticRegression 
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print("confusion_matrix:\n",confusion_matrix(y_test,y_pred))
print("classification report:",classification_report(y_test,y_pred))
print("Accuracy:",accuracy_score(y_test,y_pred))


# In[24]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_test)
print("confusion_matrix:\n",confusion_matrix(y_test,y_pred))
print("classification report: \n",classification_report(y_test,y_pred))
print("Accuracy Score:",dtree.score(X_test, y_test))


# In[28]:


from sklearn.ensemble import RandomForestClassifier
rfc_model=RandomForestClassifier()
rfc_model.fit(X_train,y_train)
y_pred=rfc_model.predict(X_test)
print("confusion_matrix:\n",confusion_matrix(y_test,y_pred))
print("classification report:\n",classification_report(y_test,y_pred))
print("Accuracy Score:",rfc_model.score(X_test, y_test))


# In[ ]:




