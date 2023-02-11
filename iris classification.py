#!/usr/bin/env python
# coding: utf-8

# <img src="oasis.jpg"/>

# ## AUTHOR-SHAIKH SADAF
# ### OASIS INFOBYTE INTERNSHIP IN DATA SCIENCE
# ###                         TASK1- IRIS FLOWER CLASSIFICATION WITH MACHINE LEARNING

# In[30]:


import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.offline as pyo

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv('Iris.csv')
df


# In[3]:


df.shape


# In[4]:


df.drop('Id',axis=1,inplace=True)


# In[5]:


df['Species'].value_counts()


# In[6]:


sns.countplot(df['Species']);


# In[7]:


plt.bar(df['Species'],df['PetalWidthCm']) 


# In[8]:


sns.pairplot(df,hue='Species')


# In[9]:


df.rename(columns={'SepalLengthCm':'SepalLength','SepalWidthCm':'SepalWidth','PetalWidthCm':'PetalWidth','PetalLengthCm':'PetalLength'},inplace=True)


# In[10]:


x=df.drop(['Species'],axis=1)


# In[12]:


x


# In[13]:


y=df['Species']


# In[14]:


y


# In[15]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[16]:


x_test


# In[17]:


x_test.size


# In[18]:


x_train.size


# In[19]:


y_test.size


# In[20]:


dt_model = DecisionTreeClassifier()
dt_model.fit(x_train,y_train) 


# In[21]:


ypredtrain = dt_model.predict(x_train)

Accuracy = accuracy_score(y_train,ypredtrain)
print('Accuracy:',Accuracy)

Confusion_matrix = confusion_matrix(y_train,ypredtrain)
print('Confusion_matrix: \n',Confusion_matrix)

Classification_report = classification_report(y_train,ypredtrain)
print('Classification_report: \n',Classification_report)


# In[22]:


ypredtest = dt_model.predict(x_test)

Accuracy = accuracy_score(y_test,ypredtest)
print('Accuracy:',Accuracy)

Confusion_matrix = confusion_matrix(y_test,ypredtest)
print('Confusion_matrix: \n',Confusion_matrix)

Classification_report = classification_report(y_test,ypredtest)
print('Classification_report: \n',Classification_report)


# In[23]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# In[24]:


model.fit(x_train,y_train)


# In[25]:


ypredtest = dt_model.predict(x_test)


# In[26]:


ypredtest


# In[27]:


from sklearn.metrics import accuracy_score,confusion_matrix


# In[28]:


confusion_matrix(y_test,ypredtest)


# In[29]:


accuracy=accuracy_score(y_test,ypredtest)*100
print("Accuracy of the model is {:.2f}".format(accuracy))

