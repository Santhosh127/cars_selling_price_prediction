import pandas as py
from sklearn import linear_model
import matplotlib.pyplot as plt
df= py.read_csv("car.csv")
print(df.head())
X=df[['year','km_driven']]
y=df['selling_price']

regr=linear_model.LinearRegression()
regr.fit(X,y)
predictedselling=regr.predict([[2021,100000]])
print(predictedselling)
ax = df.plot.bar(x='km_driven', y='year', rot=0)
ax = df.plot.bar(stacked=True)
df.describe()