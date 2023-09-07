#!/usr/bin/env python
# coding: utf-8

# # Random Forest- Data1

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[3]:


df=pd.read_csv("C:/Users/babya/Downloads/C1_Ionosphere.csv")
df


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.info


# In[7]:


df.describe()


# In[8]:


df["g"].value_counts()


# In[9]:


x = df.drop("g",axis=1)
y = df["g"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.40)
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)


# In[10]:


rf=RandomForestClassifier()
params={'max_depth':[1,2,3,4,5],
        'min_samples_leaf':[2,4,6,8,10],
        'n_estimators':[1,3,5,7]
       }


# In[11]:


from sklearn.model_selection import GridSearchCV
gs=GridSearchCV(estimator=rf,param_grid=params,cv=2,scoring='accuracy')
gs.fit(x_train,y_train)


# In[12]:


rf_best=gs.best_estimator_
rf_best


# In[13]:


from sklearn.tree import plot_tree
plt.figure(figsize=(40,40))
plot_tree(rf_best.estimators_[4],feature_names=None,class_names=['Yes','No'])


# # Random forest-Data2

# In[14]:


df1=pd.read_csv("C:/Users/babya/Downloads/C10_Loan1.csv")
df1


# In[15]:


df1.describe()


# In[16]:


df1.info


# In[17]:


Home_Owner = {"Home Owner":{"Yes":1,"No":2}}
df1 = df1.replace(Home_Owner)
Defaulted_Borrower = {"Defaulted Borrower":{"Yes":1,"No":2}}
df1 = df1.replace(Defaulted_Borrower)
Marital_Status = {"Marital Status":{"Divorced":0,"Single":1,"Married":2}}
df1 = df1.replace(Marital_Status)
df1


# In[18]:


df1["Home Owner"].value_counts()


# In[19]:


df1["Home Owner"].value_counts()


# In[20]:


x = df1.drop("Marital Status",axis=1)
y = df1["Marital Status"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.40)
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)


# In[21]:


rf=RandomForestClassifier()
params={'max_depth':[1,2,3,4,5],
        'min_samples_leaf':[2,4,6,8,10],
        'n_estimators':[1,3,5,7]
       }
from sklearn.model_selection import GridSearchCV
gs=GridSearchCV(estimator=rf,param_grid=params,cv=2,scoring='accuracy')
gs.fit(x_train,y_train)


# In[22]:


rf_best=gs.best_estimator_
rf_best


# In[25]:


from sklearn.tree import plot_tree
plt.figure(figsize=(4,4))
plot_tree(rf_best.estimators_[2],feature_names=None,class_names=['Yes','No'])


# In[ ]:




