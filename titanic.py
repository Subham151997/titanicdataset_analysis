#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#load test and train data
train=pd.read_csv("train.csv")


# In[3]:


test=pd.read_csv("test.csv")


# In[4]:


#knowing the train and test datsets summary statistics
train.describe()


# In[5]:


test.describe()


# In[6]:



train.shape


# In[7]:


test.shape


# In[8]:


#checking the datatype
train.dtypes


# In[9]:


#dealing the missing value
train["Age"].isnull().sum()


# In[10]:


train.head()


# In[11]:


avgAge=train["Age"].mean()
avgAge


# In[12]:


#setting the missing values in age as the average of the others
train["Age"]=train["Age"].fillna(value=avgAge)


# In[13]:


train.describe()


# In[14]:


train.head(10)


# In[15]:


#checking the importance of different features with respect to survivors
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[17]:


train[['Sex', 'Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[18]:


train[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[19]:


train[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[20]:


train[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[21]:


#visualising the data with different plots
train.boxplot(column='Age')


# In[22]:


train.boxplot(column='Fare')


# In[23]:


g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=30)


# In[24]:


grid = sns.FacetGrid(train, col='Survived', row='Pclass', height=3, aspect=2)
grid.map(plt.hist, 'Age', alpha=1, bins=20)
grid.add_legend();


# In[25]:


grid = sns.FacetGrid(train, col='Survived', row='Sex', height=3, aspect=2)
grid.map(plt.hist, 'Age', alpha=1, bins=20)
grid.add_legend();


# In[26]:


grid = sns.FacetGrid(train, row='Embarked', height=3, aspect=2)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()


# In[27]:


sns.barplot(x="Parch", y="Survived", data=train)
plt.show()
    


# In[28]:


train.describe()


# In[32]:


train['Embarked'].isnull().sum()


# In[30]:



train['Embarked']=train['Embarked'].fillna(value='others')


# In[35]:


train[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean().sort_values(by='Survived',ascending=True)


# In[37]:


#converting the imp features to interger value
Embarked_mapping = {"S": 0, "Q": 1,"C":2,"others":3}
train['Embarked'] = train['Embarked'].map(Embarked_mapping)
test['Embarked'] = test['Embarked'].map(Embarked_mapping)


# In[38]:


train.head()


# In[39]:


sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)


# In[40]:


train.head()


# In[41]:


train.Cabin.describe()


# In[42]:


train.Age.describe()


# In[50]:


train.head()


# In[52]:


X=train[["Pclass","Sex","Embarked","SibSp","Parch"]]
Y=train[["Survived"]]


# In[54]:


#importing imp ml libraries
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.22, random_state = 0)


# In[56]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)


# In[61]:


from sklearn.metrics import accuracy_score


# In[64]:


acc_logreg=round(accuracy_score(Y_pred,Y_test)*100,2)
print(acc_logreg )


# In[66]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(accuracy_score(Y_pred, Y_test) * 100, 2)
print(acc_knn)


# In[67]:


from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(X_train, Y_train)
Y_pred = decisiontree.predict(X_test)
acc_decisiontree = round(accuracy_score(Y_pred, Y_test) * 100, 2)
print(acc_decisiontree)


# In[69]:


from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(X_train, Y_train)
Y_pred = randomforest.predict(X_test)
acc_randomforest = round(accuracy_score(Y_pred, Y_test) * 100, 2)
print(acc_randomforest)


# In[71]:


#checking score wrt different models
#the best was the decision tree models
models = pd.DataFrame({
    'Model': ['LogisticRegression', 'KNN', 'Decisiontreee', 
              'Random Forest'],
    'Score': [acc_logreg, acc_knn, acc_decisiontree, 
              acc_randomforest]})
models.sort_values(by='Score', ascending=False)


# In[73]:


#building the model using decision tree 
    X_new=test[["Pclass","Sex","Embarked","SibSp","Parch"]]


# In[75]:


prediction=decisiontree.predict(X_new)


# In[76]:


pd.DataFrame({'PassengerId':test.PassengerId, 'Survived': prediction})


# In[ ]:




