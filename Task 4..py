#!/usr/bin/env python
# coding: utf-8

# Task 4: Spam SMS Detection 

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


# In[2]:


spam=pd.read_csv(r"C:\Users\HP\Downloads\spam.csv",encoding="latin-1",usecols=['v1','v2'])


# In[3]:


spam.head()


# In[4]:


spam.describe()


# In[5]:


spam.columns=['label','message']


# In[6]:


spam.head()


# In[7]:


sns.displot(spam.label,color='yellow')


# In[8]:


encode=LabelEncoder()
spam['label']=encode.fit_transform(spam["label"].values)


# In[9]:


spam.head()


# In[10]:


vectorizer=TfidfVectorizer()


# In[11]:


X=vectorizer.fit_transform(spam["message"])
X.toarray()


# In[12]:


y=spam["label"]


# In[13]:


X.shape


# In[14]:


y.shape


# In[15]:


spam.count()


# In[17]:


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[18]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[19]:


from sklearn.naive_bayes import MultinomialNB
nb_model=MultinomialNB()
nb_model.fit(X_train,y_train)
y_pred_nb=nb_model.predict(X_test)
print("confusion_matrix:\n",confusion_matrix(y_test,y_pred_nb))
print("classification report:\n",classification_report(y_test,y_pred_nb))
print("Accuracy:",accuracy_score(y_test,y_pred_nb))


# In[20]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print("confusion_matrix:\n",confusion_matrix(y_test,y_pred))
print("classification report:\n",classification_report(y_test,y_pred))
print("Accuracy:",accuracy_score(y_test,y_pred))


# In[21]:


from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print("confusion_matrix:\n",confusion_matrix(y_test,y_pred))
print("classification report:\n",classification_report(y_test,y_pred))
print("Accuracy:",accuracy_score(y_test,y_pred))


# In[22]:


results=pd.DataFrame({
    'model':['Navie Bayes','Logistic Regression','Support Vector Machine'],
    'score':[0.96,0.96,0.97]})
result_spam=results.sort_values(by='score',ascending=False)
result_spam=result_spam.set_index('score')
result_spam.head()


# In[ ]:




