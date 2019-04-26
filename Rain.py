#!/usr/bin/env python
# coding: utf-8

# In[1]:


#PREPROCESSING DATA
import numpy as np #linear algebra
import pandas as pd #data processing
df=pd.read_csv(r'''C:\Users\91830\Desktop\weatherAUS.csv''')
print('Size of data frame is:',df.shape)


# In[2]:


# we see there are some columns with null values
#Before pre-processing, let's find out which have maximum null values
df.count().sort_values()


# In[3]:


#As we can see the first four columns have than 60% data,we can ignore them
del df['Sunshine']


# In[4]:


del df['Evaporation']


# In[5]:


del df['Cloud3pm']


# In[6]:


del df['Cloud9am']


# In[7]:


#we need to remove data because it do not affect model
del df['Date']


# In[8]:


#we don't need the location column
del df['Location']


# In[9]:


# We need to remove RISK_MM because we want to predict 'RainTomorrow' and RISK_MM can leak some info to our model
del df['RISK_MM']


# In[10]:


df.count().sort_values()


# In[11]:


#get rid of all null values
df = df.dropna(how='any')
df.shape


# In[12]:


from scipy import stats
z = np.abs(stats.zscore(df._get_numeric_data()))
print(z)
df = df[(z<3).all(axis=1)]
print(df.shape)


# In[13]:


df['RainToday'].replace({'No':0, 'Yes':1},inplace = True)


# In[14]:


df['RainTomorrow'].replace({'No':0, 'Yes':1},inplace = True)


# In[15]:


categorical_columns = ['WindGustDir', 'WindDir3pm','WindDir9am']
for col in categorical_columns:
    print(np.unique(df[col]))


# In[16]:


#transform the categorical columns using dummies
df = pd.get_dummies(df, columns = categorical_columns)
df.iloc[4:9]


# In[17]:


#standardize our data using MinMaxScaler
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df),index = df.index, columns = df.columns)
df.iloc[4:9]


# In[18]:


from sklearn.feature_selection import SelectKBest, chi2
X = df.loc[:,df.columns!='RainTomorrow']
y = df[['RainTomorrow']]
selector = SelectKBest(chi2, k=3)
selector.fit(X, y)
X_new = selector.transform(X)
print(X.columns[selector.get_support(indices=True)])


# In[19]:


df = df[['Humidity3pm','Rainfall','RainToday','RainTomorrow']]
X = df['Humidity3pm'] #let's use only one feature
y = df['RainTomorrow']


# In[20]:


#Logistic Regression 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

t0=time.time()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
clf_logreg = LogisticRegression(random_state=0)
clf_logreg.fit(X_train.values.reshape(-1,1),y_train.values.reshape(-1,1))
y_pred = clf_logreg.predict(X_test.values.reshape(-1,1))
score = accuracy_score(y_test,y_pred)
print('Accuracy:',score)
print('Time taken:' , time.time()-t0)


# In[21]:


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

t0=time.time()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
clf_rf = RandomForestClassifier(n_estimators=100, max_depth=4,random_state=0)
clf_rf.fit(X_train.values.reshape(-1,1),y_train.values.reshape(-1,1))
y_pred = clf_rf.predict(X_test.values.reshape(-1,1))
score = accuracy_score(y_test,y_pred)
print('Accuracy :',score)
print('Time taken :' , time.time()-t0)


# In[22]:


#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

t0=time.time()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
clf_dt = DecisionTreeClassifier(random_state=0)
clf_dt.fit(X_train.values.reshape(-1,1),y_train.values.reshape(-1,1))
y_pred = clf_dt.predict(X_test.values.reshape(-1,1))
score = accuracy_score(y_test,y_pred)
print('Accuracy :',score)
print('Time taken :' , time.time()-t0)


# In[23]:


from sklearn import svm
from sklearn.model_selection import train_test_split

t0=time.time()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
clf_svc = svm.SVC(kernel='linear')
clf_svc.fit(X_train.values.reshape(-1,1),y_train.values.reshape(-1,1))
y_pred = clf_svc.predict(X_test.values.reshape(-1,1))
score = accuracy_score(y_test,y_pred)
print('Accuracy :',score)
print('Time taken :' , time.time()-t0)


# In[ ]:




