import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()

#load data in variable
data = pd.read_csv('real_estate_price_size.csv')
data.head()

data.describe()

#create regression
#decleration dependent/independent variable.
y = data['price']
x1 = data['size']

#scatter plot
plt.scatter(x1,y)
plt.xlabel('Size',fontsize=20)
plt.ylabel('Price',fontsize=20)
#plt.show()

#regression itself OLS
x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()     # ordenary least squares
results.summary()

#plot regressionline
plt.scatter(x1,y)
yhat = x1*223.1787+101900
fig = plt.plot(x1,yhat, lw=4, c='orange', label ='regression line')
plt.xlabel('Size', fontsize = 20)
plt.ylabel('Price', fontsize = 20)
plt.show()