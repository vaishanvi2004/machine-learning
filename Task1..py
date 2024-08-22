#!/usr/bin/env python
# coding: utf-8

# Task 1: Movie Gerne Classification

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer


# In[3]:


movie_test=pd.read_csv(r'C:\Users\HP\Documents\ml\test_data.txt\test_data.txt',sep=":::",engine='python')
movie_train=pd.read_csv(r'C:\Users\HP\Documents\ml\train_data.txt\train_data.txt',sep=":::",engine='python')


# In[4]:


movie_test.head()


# In[5]:


movie_test.describe()


# In[6]:


movie_test.columns=['SN','movie_name','description']


# In[7]:


movie_test.head()


# In[8]:


movie_train.head()


# In[9]:


movie_train.describe()


# In[10]:


movie_train.columns=['SN','movie_name','category','description']


# In[11]:


movie_train.head()


# In[12]:


plt.figure(figsize=(15,20))
sns.countplot(x='category',data=movie_train)
plt.xlabel('Movie Gernes')
plt.ylabel('Count')
plt.title('Plots')
plt.xticks(rotation=90);
plt.show()


# In[13]:


movie_combined=pd.concat([movie_test,movie_train],axis=1)


# In[14]:


movie_combined.head()


# In[15]:


movie_combined.shape


# In[16]:


vectorizer=TfidfVectorizer()


# In[17]:


X=vectorizer.fit_transform(movie_combined["category"])
X.toarray()


# In[18]:


y=movie_combined['category']


# In[19]:


X.shape


# In[20]:


y.shape


# In[21]:


movie_combined.count()


# In[22]:


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[23]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[24]:


from sklearn.naive_bayes import MultinomialNB
nb_model=MultinomialNB()
nb_model.fit(X_train,y_train)
y_pred_nb=nb_model.predict(X_test)
print("confusion_matrix:\n",confusion_matrix(y_test,y_pred_nb))
print("classification report:\n",classification_report(y_test,y_pred_nb))
print("Accuracy:",accuracy_score(y_test,y_pred_nb))


# In[25]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print("confusion_matrix:\n",confusion_matrix(y_test,y_pred))
print("classification report:\n",classification_report(y_test,y_pred))
print("Accuracy:",accuracy_score(y_test,y_pred))


# In[26]:


from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print("confusion_matrix:\n",confusion_matrix(y_test,y_pred))
print("classification report:\n",classification_report(y_test,y_pred))
print("Accuracy:",accuracy_score(y_test,y_pred))


# In[ ]:




