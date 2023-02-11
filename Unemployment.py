#!/usr/bin/env python
# coding: utf-8

# # AUTHOR: SHAIKH SADAF
# ## DATA SCIENCE INTERN AT OASIS INFOBYTE
# ### Task 2- UNEMPLOYMENT ANALYSIS USING PYTHON
# ### Details- Unemployment is measured by the unemployment rate which is the number of people who are unemployed as a percentage of the total labour force. We have seen a sharp increase in the unemployment rate during Covid-19, so analyzing the unemployment rate can be a good data science project.
# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("Unemployment in India.csv")
data1 = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")


# In[3]:


data.head()


# In[4]:


data1.head()


# In[5]:


data.tail()


# In[6]:


data1.tail()


# In[7]:


data.columns


# In[8]:


data1.columns


# In[9]:


data.shape


# In[10]:


data1.shape


# In[11]:


data.size


# In[12]:


data1.size


# In[13]:


data.info()


# In[14]:


data1.info()


# In[15]:


data.describe()


# In[16]:


data1.describe()


# In[17]:


data.isnull()


# In[18]:


data1.isnull()


# In[19]:


data.isnull().value_counts()


# In[20]:


data1.isnull().value_counts()


# In[21]:


data.dropna(inplace=True) 


# In[22]:


data.tail()


# In[23]:


data.shape


# In[24]:


data1.dropna(inplace=True)


# In[25]:



data1.shape


# In[26]:


data.isnull().sum()


# In[27]:


data1.isnull().sum()


# In[28]:


data.duplicated()


# In[29]:


data1.duplicated()


# In[30]:


data.duplicated().sum()


# In[31]:


data1.duplicated().sum()


# In[32]:


data.duplicated().value_counts()


# In[33]:


sns.pairplot(data)


# In[34]:


sns.pairplot(data1)


# In[35]:


sns.heatmap(data.corr(),annot=True,cmap="seismic")


# In[36]:


sns.heatmap(data1.corr(),annot=True,cmap="seismic")


# In[37]:


plt.bar(data['Region'], data[' Estimated Unemployment Rate (%)'])
plt.xticks(rotation=90)
plt.xlabel('States')
plt.ylabel('Unemployment Rate')
plt.title('Unemployment according to States')
plt.show()


# In[38]:


sns.histplot(x=' Estimated Unemployment Rate (%)', data =data, kde = True,hue ='Area')
plt.title('Unemployment according to Area')
plt.xlabel('Unemployment Rate')
plt.show()


# In[39]:


plt.scatter(data1['Region.1'], data1[' Estimated Unemployment Rate (%)'])
plt.xlabel('States')
plt.ylabel('Unemployment Rate')
plt.title('Unemployment according to Region')
plt.show()


# In[40]:


plt.bar(data['Region'], data[' Estimated Labour Participation Rate (%)'])
plt.xticks(rotation=90)
plt.xlabel('States')
plt.ylabel('Labour Participation Rate')
plt.title('Labour Participation according to States')
plt.show()


# In[41]:


sns.histplot(x=' Estimated Labour Participation Rate (%)', data =data, kde = True,hue ='Area')
plt.title('Labour Participation according to Area')
plt.xlabel('Labour Participation Rate')
plt.show()


# In[42]:


sns.lineplot(y= ' Estimated Labour Participation Rate (%)', x= ' Date', data=data)
plt.title('Labour Participation according to Date')
plt.xlabel('Date')
plt.xticks(rotation=90)
plt.ylabel('Labour Participation Rate')
plt.show()


# In[43]:


plt.scatter(data1['Region.1'], data1[' Estimated Labour Participation Rate (%)'])
plt.xlabel('Region')
plt.ylabel('Labour Participation Rate')
plt.title('Labour Participation according to Region')
plt.show()


# In[44]:


plt.bar(data['Region'], data[' Estimated Employed'])
plt.xticks(rotation=90)
plt.xlabel('States')
plt.ylabel('Employement Rate')
plt.title('Employement according to States')
plt.show()


# In[45]:


sns.histplot(x=' Estimated Employed', data =data, kde = True,hue ='Area')
plt.title('Employement according to Area')
plt.xlabel('Employement Rate')
plt.show()


# In[46]:


sns.lineplot(y= ' Estimated Employed', x= ' Date', data=data)
plt.title(' Employement according to Date')
plt.xlabel('Date')
plt.xticks(rotation=90)
plt.ylabel(' Employement Rate')
plt.show()


# In[47]:


plt.scatter(data1['Region.1'], data1[' Estimated Employed'])
plt.xlabel('Region')
plt.ylabel('Employement Rate')
plt.title('Employement according to Region')
plt.show()

