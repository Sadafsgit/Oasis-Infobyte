#!/usr/bin/env python
# coding: utf-8

# # AUTHOR: PAGARE MAITHILI
# ## DATA SCIENCE INTERN AT OASIS INFOBYTE
# ### TASK 5 - SALES PREDICTION USING PYTHON
# ### DETAILS - Sales prediction means predicting how much of a product people will buy based on factors such as the amount you spend to advertise your product, the segment of people you advertise for, or the platform you are advertising on about your product.
# 
# 

# In[1]:


#importing required libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# In[2]:


dt=pd.read_csv('Advertising.csv')
dt


# In[3]:


#Delete column 1
data=dt.drop(columns=['Unnamed: 0'],axis=1)
data


# In[4]:


#to check null values
data.isna().sum()


# In[5]:


data.shape


# In[6]:


data.dtypes


# In[7]:


data.info()


# In[8]:


data.describe()


# In[9]:


#Correlation of each column
data.corr()


# In[10]:


sns.pairplot(data)


# In[11]:


sns.heatmap(data.corr(),annot=True)


# In[12]:


#Splitting data into training and testing dataset
from sklearn.model_selection import train_test_split
x = data.drop(columns="Sales",axis=1)
y = data["Sales"]
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.30, random_state=0)


# In[13]:


x_train


# In[14]:


y_train


# In[15]:


x_test


# In[16]:


y_test


# ## Linear Regression
# 

# In[17]:


#Multiple Linear Regression
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)


# In[18]:


y_pred=reg.predict(x_test)
y_pred


# In[19]:


print("Beta1=",reg.coef_)


# In[20]:


print("Beta0=",reg.intercept_)


# In[21]:


from sklearn.metrics import mean_squared_error,r2_score
r2_score(y_test,y_pred)


# In[22]:


#MSE
mean_squared_error(y_test,y_pred,squared=True)


# In[23]:


#RMSE
mean_squared_error(y_test,y_pred,squared=False)


# In[24]:


reg.predict([['151.5','9.3','8.1']])


# In[25]:


#To transform features by scaling each feature to a given range.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
names=data.columns
model1=scaler.fit_transform(data)
scaled_df=pd.DataFrame(model1,columns=names)
scaled_df


# In[26]:


#splitting scaled dataset in training and testing set
x1=scaled_df.drop(columns="Sales",axis=1)
y1=scaled_df["Sales"]
x1_train,x1_test,y1_train,y1_test = train_test_split(x1, y1, test_size=0.30, random_state=0)


# In[27]:


model=LinearRegression().fit(x1_train,y1_train)
y1_pred=model.predict(x1_test)


# In[28]:


y1_pred


# In[29]:


print("Beta1=",model.coef_)


# In[30]:


print("Beta0=",model.intercept_)


# In[31]:


r2_score(y1_test,y1_pred)


# In[32]:


mean_squared_error(y1_test,y1_pred,squared=True)


# In[33]:


mean_squared_error(y1_test,y1_pred,squared=False)


# ## XGBoost

# In[34]:


from xgboost import XGBRegressor
model1=XGBRegressor()
model1.fit(x_train,y_train)


# In[35]:


y_pred2=model1.predict(x_test)
y_pred2


# In[36]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,recall_score,precision_score,accuracy_score
mean_absolute_error(y_test,y_pred2)


# In[37]:


#MSE value
mean_squared_error(y_test,y_pred2)


# In[38]:


#RMSE value
mean_squared_error(y_test,y_pred2,squared=False)


# In[39]:


r2_score(y_test,y_pred2)

