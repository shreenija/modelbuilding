#!/usr/bin/env python
# coding: utf-8

# # Linear Regression - Data1

# In[12]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[5]:


df=pd.read_csv("C:/Users/babya/Downloads/1_2015.csv")
df


# In[6]:


df.describe()


# In[7]:


df.info


# In[8]:


df.head()


# In[9]:


df.tail()


# In[10]:


df.columns


# In[13]:


sns.pairplot(df)


# In[14]:


sns.displot(df['Freedom'])


# In[15]:


newdata=df[['Happiness Rank', 'Happiness Score',
       'Standard Error', 'Economy (GDP per Capita)', 'Family',
       'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)',
       'Generosity', 'Dystopia Residual']]


# In[16]:


sns.heatmap(newdata.corr())


# In[17]:


x=newdata[['Happiness Rank', 'Happiness Score',
       'Standard Error', 'Economy (GDP per Capita)', 'Family',
       'Health (Life Expectancy)', 'Trust (Government Corruption)',
       'Generosity', 'Dystopia Residual']]
y=df['Freedom']


# In[18]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30)


# In[19]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)


# In[20]:


predx=lr.predict(x_test)
print(predx)


# In[21]:


print(lr.score(x_test,y_test))


# In[22]:


plt.scatter(y_test,predx)


# # Linear Regression - Data2

# In[23]:


df1=pd.read_csv("C:/Users/babya/Downloads/4_Drug200.csv")
df1


# In[24]:


df1.describe()


# In[25]:


df1.info


# In[26]:


df1.head()


# In[27]:


df1.tail()


# In[28]:


df.columns


# In[29]:


sns.pairplot(df1)


# In[30]:


BP={'BP':{'LOW':0,'NORMAL':1,'HIGH':2}}
df1=df1.replace(BP)
df1


# In[31]:


Cholesterol={'Cholesterol':{'LOW':0,'NORMAL':1,'HIGH':2}}
df1=df1.replace(Cholesterol)
df1


# In[32]:


sns.displot(df1['Cholesterol'])


# In[33]:


new_df1=df1[['Age', 'BP', 'Cholesterol', 'Na_to_K']]
sns.heatmap(new_df1.corr())


# In[34]:


x=new_df1[['Age', 'Cholesterol', 'Na_to_K']]
y=df1['BP']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)


# In[35]:


predx=lr.predict(x_test)
print(predx)


# In[36]:


print(lr.score(x_test,y_test))


# In[37]:


plt.scatter(y_test,predx)


# # Linear Regression - Data3

# In[ ]:


df2=pd.read_csv("C:/Users/babya/Downloads/7_Uber.csv")
df2


# In[39]:


df2.head()


# In[40]:


df2.tail()


# In[41]:


df2.describe()


# In[42]:


df2.info


# In[43]:


df2.columns


# In[ ]:




