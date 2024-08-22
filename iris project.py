#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import seaborn as sns
import plotly.express as px
import sklearn


# In[2]:


iris=pd.read_csv(r"C:\Users\HP\Downloads\iris.csv")


# In[3]:


iris.head()


# In[4]:


iris.shape


# In[5]:


Species = iris['species'].value_counts().reset_index()
Species


# In[6]:


iris.describe()


# In[7]:


iris.groupby('species').mean()


# In[8]:


X = iris.drop('species', axis =1)
y = iris['species']


# In[9]:


X


# In[10]:


sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=iris)
plt.show()


# In[11]:


sns.lineplot(data=iris.drop(['species'], axis=1))
plt.show()


# In[12]:


fig = px.scatter_3d(iris,x='sepal_length', y='petal_width', z='petal_length', color='species')
fig.show()


# In[13]:


iris.plot.hist(subplots=True, layout=(2,2), figsize=(10, 10), bins=20)
plt.show()


# In[14]:


sns.heatmap(iris.corr(), annot=True)
plt.show()


# In[15]:


g = sns.FacetGrid(iris, col='species')
g = g.map(sns.kdeplot, 'sepal_length')


# In[16]:


continuous_cols= ['sepal_length','sepal_width'    ,'petal_length',    'petal_width' ]
# Plotting the boxplots for continuous variables
fig, axes = plt.subplots(len(continuous_cols), 1, figsize=(10, 20))
fig.tight_layout(pad=5.0)

for i, col in enumerate(continuous_cols):
    sns.boxplot(x=iris[col], ax=axes[i])
    axes[i].set_title(f'Boxplot of {col}')
    axes[i].set_xlabel(col)

plt.show()


# In[17]:


sns.pairplot(iris)


# In[18]:


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[19]:


x = iris.drop('species', axis=1)
y= iris.species

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=5)


# In[20]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
knn.score(x_test, y_test)


# In[21]:


cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


# In[22]:


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[23]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x, y)
y_pred = logreg.predict(x)
print(metrics.accuracy_score(y, y_pred))


# In[24]:


print('confusion metrics: \n',confusion_matrix(y,y_pred))


# In[25]:


print("\nClassification Report:")
print(classification_report(y, y_pred))


# In[26]:


from sklearn.svm import SVC
svm = SVC()
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)

svm.score(x_test, y_test)


# In[27]:


cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


# In[28]:


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[29]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(x_train, y_train)
y_pred = dtree.predict(x_test)

dtree.score(x_test, y_test)


# In[30]:


conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# In[31]:


class_report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(class_report)


# In[32]:


results=pd.DataFrame({
    'model':['KNN','Logistic Regression','Support Vector Machine','Desicion Tree'],
    'score':[0.967,0.974,0.983,0.967]})
result_iris=results.sort_values(by='score',ascending=False)
result_iris=result_iris.set_index('score')
result_iris.head()

