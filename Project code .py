#!/usr/bin/env python
# coding: utf-8

# # EDA

# In[133]:


# Importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

# load the training and testing data files
train_df = pd.read_csv(r'D:\00RU\DS8015 - Machine learning\Project\train.csv')
test_df = pd.read_csv(r'D:\00RU\DS8015 - Machine learning\Project\test.csv')


# In[134]:


# check to see if the data files have any missing values
# RangeIndex: Gives dimension of training set
# Also tells us how many non-NA values for each feature
train_df.info()


# In[135]:


train_df.head()


# **Description of the features**
# 
# ---
# 
# PassengerId: Unique ID of a passenger
# 
# Survived:    0 - Not survived and 1 -  survived
# 
# Pclass:    Ticket class of passengers. It acts as a proxy for socio-economic status (SES). Pclass value is 1 for upper, 2 for middle and 3 for lower class.
# 
# Sex:    Sex     
# Age:    Age (in years). It is fractional if less than 1. If the age is estimated, is it in the form of xx.5    
# SibSp:    Number of siblings/spouse aboard the Titanic     
# Parch:    Number of parents / children aboard the Titanic     
# Ticket:    Ticket number     
# Fare:    Passenger fare     
# Cabin:    Cabin number     
# Embarked:   Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

# In[136]:


train_df.describe(include= 'all')


# In[137]:


train_df.hist(column = ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'], 
              bins=30, figsize=(8, 8))
plt.show()


# In[138]:


total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()*100).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', '%'])
missing_data.head(5)

# Cabin has high number of missing data hence dropping this feature is more logical than imputation 
# Age and Embarked still has acceptable number of missing entries and hence imputation can be performed here


# In[139]:


# We’ll need to fill the two missing values for Embarked. 
# Taking a quick look at the two passengers that don’t have values for Embarked
# Inner bracket gives boolean output and the outer train_df gives rows having null values as the output

print (train_df[train_df.Embarked.isnull()])


# Imputation in 'Embarked' feature - Approach 1

# In[140]:


# pivot table shows a breakdown by Sex, Pclass, Embarked, and shows the number of people from each subset that survived and embarked at a specific port
# This approach suggests that the imputation should be done with 'C' values as it is the most probable values for female passengers of class 1
print (train_df.pivot_table(values='Survived', index=['Sex', 'Pclass'], 
                     columns=['Embarked'], aggfunc='count'))


# In[141]:


bins = range(0,100,10)
df = train_df.copy()
df['Age1'] = pd.cut(df['Age'], bins)

#First filter the df of females who survived
#Create a pivot table on basis of age bins created before with column as embarkment
#This shows the missing data should be from S class 
# We will use 'S' for imputation as this approach is more logical and reliable


df1 = df[(df.Survived == 1) & (df.Sex == "female")]
print (df1.pivot_table(values='Survived', index=['Age1','Pclass'], 
                     columns=['Embarked'], aggfunc=['count']))


# In[142]:


# Finally, imputation of missing values by 'S'
(train_df.Embarked.iloc[61]) = 'S'
(train_df.Embarked.iloc[829]) = 'S'


# In[144]:


le_Sex = LabelEncoder()
train_df.Sex = le_Sex.fit_transform(train_df.Sex)
test_df.Sex = le_Sex.transform(test_df.Sex)

le_Embarked = LabelEncoder()
train_df.Embarked = le_Embarked.fit_transform(train_df.Embarked)
test_df.Embarked = le_Embarked.transform(test_df.Embarked)


# In[145]:


train_df.head()


# Dealing with missing entries of 'Age' feature

# In[102]:


# We can use the classical method such as imputation with mean 
# Another approach is generating a list of random numbers (with size = df['Age'].isnull() and values mean + std or mean - std) and filling NaN values with this list

# We can do even better by using the P_class feature!
# Pclass does not contain any missing data entries also we may see a relation of the passenger class with regards to the age of passengers
# As seen here the young people are more likely to travel in class 3 (cheapest)

sns.boxplot(x='Pclass',y='Age',data=train_df)
plt.show()


# In[103]:


train_df.groupby(by='Pclass').mean()['Age']
# These values can be imputed wherever the age is missing


# In[146]:


# Now, the aim is to fill in the Above values wherever age is missing. 
# Function can be made as shown here for this purpose

# ac = [['Age','Pclass']]
def impute_age(ac):
    Age = ac[0]
    Pclass = ac[0]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 38
        elif Pclass == 2:
            return 29
        else:
            return 25
    
    else:
        return Age


# In[147]:


data = [train_df,test_df]
for dataset in data:
  dataset['Age'] = dataset[['Age','Pclass']].apply(impute_age,axis=1).astype(int) 


# In[149]:


sns.heatmap(train_df.isnull())
# Now the only feature having missing values is Cabin but we are gonna drop this feature as it has 77% missing values!!
train_df = train_df.drop('Cabin',axis=1)
test_df = test_df.drop('Cabin',axis=1)


# In[107]:


sns.heatmap(train_df.isnull())


# Missing Values handled successfully! Now we will convert features

# In[150]:


sns.heatmap(train_df[['Survived',"Age","Sex","SibSp","Parch","Pclass",'Embarked']].corr(), annot = True)
plt.show()


# In[151]:


# Dropping name and ticket feature, as it is of less importance

train_df = train_df.drop(['Name','Ticket'],axis=1)


# In[152]:


#Looks good now
train_df.info()


# In[111]:


train_df.head(5)


# # FEATURE IMPORTANCE

# In[112]:


# Pclass - First of all we will check that pclass made any difference to survival 
# extracting required data from dataframe

pclass_survived = train_df[['Survived', 'Pclass']]        
pclass_survived.head()

#grouping data to check total & survivrd by pclass

A_survived = pclass_survived.groupby(['Pclass']).sum()
B_total = pclass_survived.groupby(['Pclass']).count()
B_total.rename(columns = {'Survived':'Total'}, inplace = True)

# data merging
pclass_status = pd.merge(A_survived, B_total, left_index=True, right_index=True) # merge by index
pclass_status["survived %"] = (pclass_status['Survived'] / pclass_status['Total']) * 100
pclass_status


# In[113]:


# Ploting the graph of result
X = pclass_status.index.values
A = pclass_status.Total
B = pclass_status.Survived

pht = plt.bar(X, A)
phs = plt.bar(X, B)

plt.xticks(X, X)
plt.xlabel('P-class')
plt.ylabel('Passenger')
plt.title('Survivors_by_Class')


plt.legend([pht,phs],['Died', 'Survived'])


# So, for above chart we can easily make conclusion that, Passengers in first class has heighest rate of survival followed by 2nd class and 3rd class passengers.

# In[114]:


# Gender - Now we will check which gender survived the most
# grouping based on sex
A_sex = train_df.groupby('Sex')

# calculating syrvival
sex_survived = A_sex['Survived'].sum()
sex_survived.name = 'Survived'

# calculating total
sex_total = A_sex['Survived'].size()
sex_total.name = 'Total'

# concating the result
merged_sex = pd.concat([sex_survived, sex_total], axis=1)

# Calculating percentage of survival
merged_sex['Survival %'] = (merged_sex['Survived'] / merged_sex['Total']) * 100
merged_sex


# In[115]:


# Ploting results

X = range(len(merged_sex.index.values))
T = merged_sex.Total
S = merged_sex.Survived

pht = plt.bar(X, T)
phs = plt.bar(X, S)

plt.xticks(X, merged_sex.index.values)
plt.xlabel('Gender')
plt.ylabel('Passengers')
plt.title('Survival based on Gender 0-Female 1-Male')

plt.legend([pht,phs],['Died', 'Survived'])


# From above results we can say that female had more chances of survival in terms of percentage it is around 74% which is around 4 times higher than the male survival rate (18.8 %)

# In[116]:


# Now let's check check that person who traveling with someone had more chaces of survival
Not_alone = (train_df.SibSp + train_df.Parch) >= 1
Not_alone_Passenger = train_df[Not_alone]

alone = (train_df.SibSp + train_df.Parch) == 0
alone_Passenger = train_df[alone]

print('Not alone - Statatics')
display(Not_alone_Passenger.describe())
print('Alone - Statatics')
display(alone_Passenger.describe())


# In[117]:


# Ploting the result
Not_alone_Passenger.Age.hist(label='Not_Alone')
alone_Passenger.Age.hist(label='Alone', alpha=0.5)

plt.xlabel('Age')
plt.ylabel('Passenger')
plt.legend(loc='best')
plt.title('Alone & Not Alone Passenger\'s Ages')


# So, from above chart we can conclude below:
# 
# in the range of 0-10, they are note alone and we can say that they are kids which make sense
# 537 passengers were alone at the same time 354 were not alone

# In[118]:


# let's reviw the survival
Alone_status = np.where((train_df.SibSp + train_df.Parch) >= 1, 'Not Alone', 'Alone')
Company_Status = train_df.groupby(Alone_status, as_index=False)['Survived'].agg([np.sum, np.size])
Company_Status.rename(columns={'sum':'Survived', 'size':'Total'},inplace=True)
Company_Status['Survival %'] = (Company_Status.Survived / Company_Status.Total) * 100
Company_Status


# In[119]:


# Ploting the results
X = range(len(Company_Status.index.values))
P = Company_Status.Total
Q = Company_Status.Survived

pht = plt.bar(X, P)
phs = plt.bar(X, Q)

plt.xticks(X, Company_Status.index.values)
plt.xlabel('Alone  & Not Alone Status')
plt.ylabel('Passengers')
plt.title('Survivors by Company Status')


plt.legend([pht,phs],['Died', 'Survived'])


# From above chart we can say that peoples who are not alone had better chances of survival

# In[120]:


# Now let's check which age group has better chance of survival
Age_Male = (train_df[train_df.Sex == 1])['Age']
Age_Male.describe()


# In[121]:


Age_Female = (train_df[train_df.Sex == 0])['Age']
Age_Female.describe()


# In[122]:


Age_Male.hist(label='Male')
Age_Female.hist(label='Female')

plt.xlabel('Age')
plt.ylabel('Passengers')
plt.title('Male & Female passenger ages')
plt.legend(loc='best')


# From above chart, we can conclude below:
# 
# no of males are higher than female in every age group
# The age of oldest male is around 80 and for female it is aroung 62

# In[123]:


# Age Group - Survival Analysis
def group_age(age):
    if age >= 80:
        return '80-89'
    if age >= 70:
        return '70-79'
    if age >= 60:
        return '60-69'
    if age >= 50:
        return '50-59'
    if age >= 40:
        return '40-49'
    if age >= 30:
        return '30-39'
    if age >= 20:
        return '20-29'
    if age >= 10:
        return '10-19'
    if age >= 0:
        return '0-9'
    
train_df['Age_Group'] = train_df.Age.apply(group_age)

Summary_Age_Group = train_df.groupby(['Age_Group'], as_index=False)['Survived'].agg([np.sum, np.size])
Summary_Age_Group = Summary_Age_Group.rename(columns={'sum':'Survived', 'size':'Total'})
Summary_Age_Group


# In[124]:


# Ploting the result
X = range(len(Summary_Age_Group.index.values))
M = Summary_Age_Group.Total
N = Summary_Age_Group.Survived

pht = plt.bar(X, M)
phs = plt.bar(X, N)

plt.xticks(X, Summary_Age_Group.index.values)
plt.xlabel('Age_groups')
plt.ylabel('Passenger')
plt.title('Age_Group Survival')


plt.legend([pht,phs],['Died', 'Survived'])


# In[45]:


agegroup_gender_summary = train_df.groupby(['Sex','Age_Group'], as_index=False)['Survived'].mean()
agegroup_gender_summary


# In[125]:


Age_Groups = train_df.Age_Group.unique()
age_labels = sorted(Age_Groups)
ax = sns.barplot(x='Age_Group', y='Survived', 
                 data=train_df, hue='Sex', order=age_labels ,ci= None)
ax.set_title('Survivors by Gender by Age groups')
plt.show()


# From above visulization we can say that the female and children were given preference in rescue operation and they had higher chances of survival

# In[47]:


# Let's check survival chances based on gender per class and age
# Plotinf the chart
sns.swarmplot(x='Pclass', y='Age', data=train_df, hue='Sex', 
              dodge=True).set_title('Male and Female Passenger Ages by Class')


# From above swarmplot we cam say that most no of passengers atr there in 3rd class specially male passenger. and that too in the range og 18-32 age group

# In[48]:


# Let's plot box plot in order to understand the age group distribution
sns.boxplot(x='Pclass', y='Age', data=train_df, hue='Sex').set_title('Male and Female Passenger Ages by Class')


# From above box plot we can say that in the third class passengers are younger than other classes

# In[126]:


#EMBARKED

g = sns.catplot(x="Embarked", y="Survived", hue='Sex', 
                data=train_df, height=4, 
                kind="bar", palette="muted", ci= None)
g.despine(left=True)
g.set_xticklabels(['Cherbourg', 'Queensland', 'Southampton'])
g.set_ylabels("survival probability")
plt.show()


# In[50]:


train_df = train_df.drop(columns = 'Age_Group')


# In[51]:


train_df = train_df.drop(columns = 'PassengerId')


# In[52]:


train_df.info()


# # FEATURE ENGINEERING

# In[58]:


# Function which checks if an individual has age less than 5 years and assigns it value 1 
# A new feature 'Child' is created in train and test data sets 

def is_child(x):
    if int(x) <= 5:
        return 1
    else:
        return 0

train_df['Child'] = train_df.Age.apply(is_child)


# In[59]:


(train_df['Child']).sum()


# In[61]:


g = sns.catplot(x="Child", y="Survived", data=train_df, height=6,kind="bar", palette="muted" , ci =None)
g.despine(left=True)
g.set_xticklabels(['Over 5yrs', 'Under 5yrs'])
g.set_ylabels("Survival Probability")
plt.show()

# Children have higher probability of survival


# In[63]:


# SIBSP & PARCH

train_df['FamSize'] = train_df.SibSp + train_df.Parch
test_df['FamSize'] = test_df.SibSp + test_df.Parch


# In[64]:


g = sns.catplot(x="FamSize", y="Survived", data=train_df, height=6, kind="bar", palette="muted" ,ci= None)
g.set_ylabels("survival probability")
plt.show()
# This shows that the survival probability for an individual with less or equal 3 relatives are high


# In[65]:


train_df['Sex'] = train_df['Sex'].replace(["Female","Male"],[0,1])


# In[66]:


train_df['Fare_Per_Person']=train_df['Fare']/(train_df['FamSize']+1)


# In[70]:


train_df['Fare_Per_Person'].describe()


# In[88]:


def farepp(fare):
    if fare >= 400:
        return '400+'
    if fare >= 300:
        return '300-400'
    if fare >= 150:
        return '150-300'
    if fare >= 100:
        return '100-150'
    if fare >= 50:
        return '50-100'
    if fare >= 0:
        return '0-50'
    
train_df['Fare_Group'] = train_df.Fare_Per_Person.apply(farepp)


# In[89]:


train_df


# In[90]:


g = sns.catplot(x="Fare_Group", y="Survived", data=train_df, height=6, kind="bar", palette="muted" ,order=train_df['Fare_Group'].value_counts().index, ci= None)
g.set_ylabels("survival probability")
plt.show()


# In[173]:


train_df = train_df.drop(columns = ['SibSp','Parch','Fare'])


# In[174]:


train_df


# # CLASSIFICATION MODELS

# In[175]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[176]:


# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 


# In[177]:


X = train_df.drop(['Survived'], axis=1)
Y = train_df.Survived

X_train, X_test, Y_train, Y_test = train_test_split(X, Y) # default split is 75% train and 25% test


# In[178]:


for name,method in [('SVM', SVC(kernel='linear',random_state=100)),

                    ('Logistic Regression',LogisticRegression()),

                    ('Decision Tree',DecisionTreeClassifier(random_state=100)),
                    ('Random Forest',RandomForestClassifier(n_estimators=100))]:

    method.fit(X_train, Y_train)

    predict = method.predict(X_test)

    target_names=['0','1']

    print('\nEstimator: {}'.format(name))

    print(confusion_matrix(Y_test,predict)) 

    print(classification_report(Y_test,predict,target_names=target_names)) 


# # K fold cross validation

# In[181]:


from sklearn.model_selection import cross_val_score
SVM = SVC(kernel='linear',random_state=100)
scores = cross_val_score(SVM, X_train, Y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


# In[180]:


# In this section we are confirming whether there is presence of multicolinearity!
# Using Variance Inflation Factor as shown here 

from statsmodels.stats.outliers_influence import variance_inflation_factor

X_variables = train_df.drop('Survived',axis=1)
vif_data = pd.DataFrame()

vif_data["feature"] = X_variables.columns
vif_data["VIF"] = [variance_inflation_factor(X_variables.values, i) for i in range(len(X_variables.columns))]

vif_data
#Variance Inflation Factor of all the features is less than 10 so there is no chance of the multicolinearity 
#amongst the independent variables 

