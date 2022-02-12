#!/usr/bin/env python
# coding: utf-8

# #### Introduction
# 
# Currently Rental bikes are introduced in many urban cities for the enhancement of mobility comfort. It is important to make the rental bike available and accessible to the public at the right time as it lessens the waiting time. Eventually, providing the city with a stable supply of rental bikes becomes a major concern. The crucial part is the prediction of bike count required at each hour for the stable supply of rental bikes.
# The dataset contains weather information (Temperature, Humidity, Windspeed, Visibility, Dewpoint, Solar radiation, Snowfall, Rainfall), the number of bikes rented per hour and date information.

# In[1]:


## Importing libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


## Check if the data is loaded properly

os.chdir("F:/Linear Regression/Dataset/bike-sharing-demand")
trainDF = pd.read_csv("train.csv")
testDF = pd.read_csv("test.csv")

trainDF.head()


# In[3]:


## Number of columns and rows
print(trainDF.shape)
print(testDF.shape)


# In[4]:


## Data cleaning 

## Separately store year, month, day, hour, and day values in columns
trainDF['datetime'] = pd.to_datetime(trainDF['datetime'])
testDF['datetime'] = pd.to_datetime(testDF['datetime'])

## extracting day, month, year and hour in separate columns
trainDF['year'] = trainDF['datetime'].apply(lambda x: x.year)
trainDF['month'] = trainDF['datetime'].apply(lambda x: x.month)
trainDF['day'] = trainDF['datetime'].apply(lambda x: x.day)
trainDF['hour'] = trainDF['datetime'].apply(lambda x: x.hour)

testDF['year'] = testDF['datetime'].apply(lambda x: x.year)
testDF['month'] = testDF['datetime'].apply(lambda x: x.month)
testDF['day'] = testDF['datetime'].apply(lambda x: x.day)
testDF['hour'] = testDF['datetime'].apply(lambda x: x.hour)


# In[5]:


testDF.head(10)


# In[6]:


trainDF.head(10)


# In[7]:


## as we can see the test data doesn't have the columns registered or casual and we don't need datetime as we have separated; 
## we will drop them

testDF.drop(['datetime'], axis = 1, inplace = True)
trainDF.drop(['datetime', 'casual', 'registered'], axis = 1, inplace = True)


# In[8]:


testDF.head(10)


# In[9]:


trainDF.head(10)


# In[10]:


## Check for missing values

print(testDF.isnull().sum())
print(trainDF.isnull().sum())


# In[10]:


## We can clearly see that there are no missing values; so lets make charts to understand each feature in our data set


# In[36]:


trainDF.groupby('year')['count'].mean()


# In[11]:


## EDA

fig = plt.figure(figsize=[12,10])
ax1 = fig.add_subplot(2,2,1)
ax1 = sns.barplot(x = 'year', y = 'count', data = trainDF.groupby('year')['count'].mean().reset_index())

ax2 = fig.add_subplot(2,2,2)
ax2 = sns.barplot(x = 'month', y = 'count', data = trainDF.groupby('month')['count'].mean().reset_index())

ax3 = fig.add_subplot(2,2,3)
ax3 = sns.barplot(x = 'day', y = 'count', data = trainDF.groupby('day')['count'].mean().reset_index())

ax4 = fig.add_subplot(2,2,4)
ax4 = sns.barplot(x = 'hour', y = 'count', data = trainDF.groupby('hour')['count'].mean().reset_index())


# From the above charts following assumptions can be made:
# 
# 1. The year 2012 saw a significant rise in rental counts; mostly because of increase in overall popularity. It is safe to assume that from 2012 onwards we can see a good rise.
# 2. The seasons in South Korea have a good impact in the counts. We can see that from March to May there is a slow climb in the counts; as there is a spring season and it is safe to assume for leisure people would love to ride their bicycles. From June till August we can see almost no change in counts as there is summer season and its good to head out fresh and its a great way to exercise and burn calories. The count is still stagnant through October and that marks the end or atleast towards the end of Autumn season and then we see a slow decline as winter hits Korea. 
# 3. The deviation in days is not far off, it might mean people who bicycle daily will continue to do so
# 4. We see spikes at 7, 17 & 18th hours, and that might be due to traffic rush

# In[12]:


fig = plt.figure(figsize=[12,10])
ax1 = fig.add_subplot(2,2,1)
ax1 = sns.barplot(x = 'workingday', y = 'count', data = trainDF.groupby('workingday')['count'].mean().reset_index())

ax2 = fig.add_subplot(2,2,2)
ax2 = sns.barplot(x = 'weather', y = 'count', data = trainDF.groupby('weather')['count'].mean().reset_index())

ax3 = fig.add_subplot(2,2,3)
ax3 = sns.barplot(x = 'holiday', y = 'count', data = trainDF.groupby('holiday')['count'].mean().reset_index())

ax4 = fig.add_subplot(2,2,4)
ax4 = sns.barplot(x = 'season', y = 'count', data = trainDF.groupby('season')['count'].mean().reset_index())


# From the above charts following conclusions can be drawn:
# 
# 1. Working day(1 = Working day, 0 = Not a working day) have little to no impact and this same goes for holidays. Assumption might be that people who have a habit of cycling will use it for their chores; if needed, or for leisure and/or exercising.
# 2. Winter can see the worst counts, while Autumn is the highest followed by Summer and then spring
# 3. Cycles rental counts in sunny weather is high as opposed to bad weathers

# In[13]:


## Cleaning data 

## There are no missing values; however there are windspeed = 0. This is not supposed to happen as there is always some windspeed
## while riding
## Plotting a graph to check the wind speeds

trainDF['windspeed'].value_counts()


# In[14]:


## Clearly can be seen 1313 entries have 0.00 Windspeed.

fig, axes = plt.subplots(nrows = 2)
fig.set_size_inches(15,15)

plt.sca(axes[0])
plt.xticks(rotation = 30, ha = 'right')
axes[0].set(ylabel = 'count', title = "Train Data Windspeed")
sns.countplot(data = trainDF, x = 'windspeed', ax = axes[0])

plt.sca(axes[1])
plt.xticks(rotation = 30, ha = 'right')
axes[1].set(ylabel = 'count', title = "Test Data Windspeed")
sns.countplot(data = testDF, x = 'windspeed', ax = axes[1])


# In[15]:


## to fix the above issue, which will hamper our model. We will replace 0.00 with the average of the windspeed column

trainDF.loc[trainDF['windspeed'] == 0, "windspeed"] = trainDF['windspeed'].mean()
testDF.loc[testDF['windspeed'] == 0, "windspeed"] = testDF['windspeed'].mean()


# In[16]:


## Checking the graph after implementing the above solution
fig, axes = plt.subplots(nrows = 2)
fig.set_size_inches(15,15)

plt.sca(axes[0])
plt.xticks(rotation = 30, ha = 'right')
axes[0].set(ylabel = 'count', title = "Train Data Windspeed")
sns.countplot(data = trainDF, x = 'windspeed', ax = axes[0])

plt.sca(axes[1])
plt.xticks(rotation = 30, ha = 'right')
axes[1].set(ylabel = 'count', title = "Test Data Windspeed")
sns.countplot(data = testDF, x = 'windspeed', ax = axes[1])


# In[17]:


## To check if 'count' in 'Train' is normally distributed or not. If the target variable (y) is normally distributed then it will help us make better model and hence get better results
## For instance, a skewed distribution will lead to high MSE values due to cases located on the other side of the distribution, while the MSE is limited if the data is transformed to a normal distribution.

sns.distplot(trainDF['count'])


# In[18]:


## The count is not normally distributed
## We will apply log transformation ny using numpy

sns.distplot(np.log1p(trainDF['count']))


# In[19]:


## Even though its not a normally distributed graph, the skewness is reduced, which will intern help out model

## the value in column 'count' before transformation:

print(trainDF['count'])


# In[20]:


## ## The value after apply log transformation

trainDF['count'] = np.log1p(trainDF['count'])
print(trainDF['count'])


# Feature Engineering

# In[21]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression


# In[22]:


## sampling

trainX = trainDF.drop(['count'], axis = 1).copy()
trainY = trainDF['count'].copy()

print(trainX.shape)
print(trainY.shape)


# In[37]:


## VIF - To check multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor

tempMaxVIF = 5 # The VIF which will be calculated at every iteration of the loop
maxVIFCutoff = 5 # 5 is the recommended cutoff value for VIF
trainXCopy = trainX.copy()
counter = 1
highVIFcolumnNames = []

while (tempMaxVIF >= maxVIFCutoff):
    print(counter)
    
    #create an empty dataframe to store VIF values
    tempVIFDf = pd.DataFrame()
    
    #calculate VIF using list comprehension
    tempVIFDf['VIF'] = [variance_inflation_factor(trainXCopy.values, i) for i in range(trainXCopy.shape[1])]
    
    #Create a new column 'Column_Name' to store column names against VIF values from list comprehension
    tempVIFDf['Column_Name'] = trainXCopy.columns
    
    #Drop NA rows from DF - If there is some calculation errors resulting in NA
    tempVIFDf.dropna(inplace=True)
    
    #Sort the df based on VIF values, then pick the most column names (Which has the highest values)
    tempColumnName = tempVIFDf.sort_values(['VIF'], ascending = False).iloc[0,1]
    
    #Store the max VIF value in tempMaxVIF
    tempMaxVIF = tempVIFDf.sort_values(['VIF'], ascending = False).iloc[0,0]
    
    print(tempColumnName)
    
    if (tempMaxVIF >= maxVIFCutoff): #This condition will ensure that columns having VIF more than 5 are not dropped
        trainXCopy = trainXCopy.drop(tempColumnName, axis = 1)
        highVIFcolumnNames.append(tempColumnName)
        
    counter +=1
    
highVIFcolumnNames


# In[24]:


## Since Weather's VIF is the highest, will drop that column. 

trainX_noWeather = trainX.drop(['weather'], axis = 1)


# In[29]:


## Model Building ##

from statsmodels.api import OLS
m1modelDef = OLS(trainY, trainX_noWeather) #Model definition
m1modelBuild = m1modelDef.fit() #Model building
m1modelBuild.summary()


# In[26]:


## Extracting or Identifying p-values from the models

dir(m1modelBuild)
m1modelBuild.pvalues


# In[30]:


## Linear regression if we don't remove weather ##

m2modelDef = OLS(trainY, trainX)
m2modelBuild = m1modelDef.fit()
m2modelBuild.summary()


# In[35]:


## model optimization ##

tempMaxpValue = 0.1
maxPvalueCutoff = 0.1
trainXCopy = trainX.copy()
counter = 1
highPvaluecolumnNames = []

while (tempMaxpValue >= maxPvalueCutoff):
    print(counter)
    
    tempmodelDf = pd.DataFrame()
    model = OLS(trainY, trainXCopy).fit()
    tempmodelDf['PValue'] = model.pvalues
    tempmodelDf['Column_Name'] = trainXCopy.columns
    tempmodelDf.dropna(inplace = True) ## if there is some calculation error then they will come as NA, so we will drop them
    tempColumnName = tempmodelDf.sort_values(['PValue'], ascending = False).iloc[0,1]
    tempMaxpValue = tempmodelDf.sort_values(['PValue'], ascending = False).iloc[0,0]
    
    if (tempMaxpValue >= maxPvalueCutoff): # This condition will ensure that ONLY columns having p-value lower than 0.1 are NOT dropped
        print(tempColumnName, tempMaxpValue)
        trainXCopy = trainXCopy.drop(tempColumnName, axis = 1)
        highPvaluecolumnNames.append(tempColumnName)
    
    counter +=1
    
highPvaluecolumnNames


# In[38]:


## Check final model summary ##

model.summary()
trainX = trainX.drop(highPvaluecolumnNames, axis = 1) 


# In[39]:


model.predict(testDF)


# In[ ]:




