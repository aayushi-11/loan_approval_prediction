#!/usr/bin/env python
# coding: utf-8

# # Loan Approval Prediction

# In[1]:


import pandas as pd
import numpy as np
import os


# In[2]:


## Pandas
## Numpy
## SKlearn
## Matplotlib


# In[3]:


train=pd.read_csv('train.csv')
train.Loan_Status=train.Loan_Status.map({'Y':1,'N':0})


# ### finding number of missing value

# In[4]:


train.isnull().sum()


# ### Preprocessing

# In[5]:


Loan_status=train.Loan_Status
train.drop('Loan_Status',axis=1,inplace=True)
test=pd.read_csv('test.csv')
Loan_ID=test.Loan_ID
data=train.append(test)
data.head()


# In[6]:


data.shape


# In[7]:


data=data.drop('Loan_ID',axis=1)


# In[8]:


data.describe()


# In[9]:


data.isnull().sum()


# In[10]:


data.Dependents.dtypes


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)


# ### encoding categorical data

# In[12]:


## Label encoding for gender
data.Gender=data.Gender.map({'Male':1,'Female':0})
data.Gender.value_counts()


# In[13]:


## Let's see correlations
corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)


# In[14]:


## Labelling 0 & 1 for Marrital status
data.Married=data.Married.map({'Yes':1,'No':0})


# In[15]:


data.Married.value_counts()


# In[16]:


## Labelling 0 & 1 for Dependents
data.Dependents=data.Dependents.map({'0':0,'1':1,'2':2,'3+':3})


# In[17]:


data.Dependents.value_counts()


# In[18]:


## Let's see correlations for it
corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)


# In[19]:


## Labelling 0 & 1 for Education Status
data.Education=data.Education.map({'Graduate':1,'Not Graduate':0})


# In[20]:


data.Education.value_counts()


# In[21]:


## Labelling 0 & 1 for Employment status
data.Self_Employed=data.Self_Employed.map({'Yes':1,'No':0})


# In[22]:


data.Self_Employed.value_counts()


# In[23]:


data.Property_Area.value_counts()


# In[24]:


## Labelling 0 & 1 for Property area
data.Property_Area=data.Property_Area.map({'Urban':2,'Rural':0,'Semiurban':1})


# In[25]:


data.Property_Area.value_counts()


# In[26]:


corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)


# In[27]:


data.head()


# In[28]:


data.Credit_History.size


# ### Handling missing values

# In[29]:


data.Credit_History.fillna(np.random.randint(0,2),inplace=True)


# In[30]:


data.isnull().sum()


# In[31]:


data.Married.fillna(np.random.randint(0,2),inplace=True)


# In[32]:


data.isnull().sum()


# In[33]:


## Filling with median
data.LoanAmount.fillna(data.LoanAmount.median(),inplace=True)


# In[34]:


## Filling with mean
data.Loan_Amount_Term.fillna(data.Loan_Amount_Term.mean(),inplace=True)


# In[35]:


data.isnull().sum()


# In[36]:


data.Gender.value_counts()


# In[37]:


## Filling Gender with random number between 0-2
from random import randint 
data.Gender.fillna(np.random.randint(0,2),inplace=True)


# In[38]:


data.Gender.value_counts()


# In[39]:


## Filling Dependents with median
data.Dependents.fillna(data.Dependents.median(),inplace=True)


# In[40]:


data.isnull().sum()


# In[41]:


corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)


# In[42]:


data.Self_Employed.fillna(np.random.randint(0,2),inplace=True)


# In[43]:


data.isnull().sum()


# In[44]:


data.head()


# In[45]:


data.isnull().sum()


# In[46]:


data.head()


# ### train-test-split

# In[47]:


train_X=data.iloc[:614,] ## all the data in X (Train set)
train_y=Loan_status  ## Loan status will be our Y


# In[48]:


from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y=train_test_split(train_X,train_y,test_size=0.2,random_state=0)


# In[49]:


#sc_f = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']
#sc_f = ['ApplicantIncome','CoapplicantIncome','LoanAmount']
train_X.head()


# In[50]:


train_X


# In[51]:


test_y


# ### training on 3 different ML models

# In[52]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[53]:


DTC=DecisionTreeClassifier()
DTC.fit(train_X,train_y)
DTC_pred=DTC.predict(test_X)
accuracy=accuracy_score(DTC_pred,test_y)
accuracy


# In[54]:


RFC=RandomForestClassifier()
RFC.fit(train_X,train_y)
RFC_pred=RFC.predict(test_X)
accuracy=accuracy_score(RFC_pred,test_y)
accuracy


# In[55]:


LR=LogisticRegression()
LR.fit(train_X,train_y)
pred=LR.predict(test_X)
accuracy=accuracy_score(pred,test_y)
accuracy


# In[56]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
print(confusion_matrix(test_y,pred))
print(classification_report(test_y,pred))


# In[57]:


print(pred)
pred.shape


# In[58]:


X_test=data.iloc[614:,] 
# X_test[sc_f]=SC.fit_transform(X_test[sc_f])


# In[59]:


X_test.head()


# In[60]:


pred_df=pd.DataFrame({'actual':test_y,'predicted':pred})


# In[68]:


pred_df.head(20)


# In[62]:


## TAken data from the dataset
t = LR.predict([[0.0,0.0,0.0,1,0.0,1811,1666.0,54.0,360.0,1.0,2]])


# In[63]:


print(t)


# In[64]:


import pickle
# now you can save it to a file
file = 'loan_prediction_system/Linear_Model.pkl'
with open(file, 'wb') as f:
    pickle.dump(LR, f)


# In[65]:


with open('loan_prediction_system/Linear_Model.pkl', 'rb') as f:
    k = pickle.load(f)


# In[66]:


new_data = [[0.0, 0.0, 0.0, 1, 0.0, 4230, 0.0, 112.0, 360.0, 1.0, 1]]
predictions = k.predict(new_data)
print(predictions)

