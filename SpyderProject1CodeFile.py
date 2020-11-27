# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 19:31:40 2020

@author: sushm
"""


#Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt

#Reading data from files
data = pd.read_csv("advertising.csv")
data.head()

#To visualize data
fig , axs = plt.subplots(1,3,sharey = True)
data.plot(kind='scatter',x='TV',y='Sales',ax=axs[0],figsize=(14,7))
data.plot(kind='scatter',x='Radio',y='Sales',ax=axs[1])
data.plot(kind='scatter',x='Newspaper',y='Sales',ax=axs[2])          


#Creating X & Y for linear regression
feature_cols = ['TV']
X = data[feature_cols]
Y = data.Sales

 
#importing linear regression algorithm for simple linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,Y)

print(lr.intercept_)
print(lr.coef_)



result = 6.9748214882298925+0.05546477*50
print(result)

#Create a DataFrame with min and max value of a table
X_new = pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
X_new.head()


preds = lr.predict(X_new)
preds


data.plot(kind='scatter',x='TV',y='Sales')

plt.plot(X_new,preds,c='red',linewidth = 4)




import statsmodels.formula.api as smf
lr = smf.ols(formula = 'Sales ~ TV',data = data).fit()
lr.conf_int()



#importing probability values
lr.pvalues



#finding the r-squared values
lr.rsquared



#multi linear regression
feature_cols = ['TV','Radio','Newspaper']
X = data[feature_cols]
Y = data.Sales


lm = LinearRegression()
lm.fit(X,Y)


print(lm.intercept_)
print(lm.coef_)


lm = smf.ols(formula = 'Sales ~ TV+Radio+Newspaper',data = data).fit()
lm.conf_int()
lm.summary()



lm = smf.ols(formula = 'Sales ~ TV+Radio',data = data).fit()
lm.conf_int()
lm.summary()






























