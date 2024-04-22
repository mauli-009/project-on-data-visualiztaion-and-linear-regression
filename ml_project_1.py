
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data=pd.read_excel("/content/Dataset_psp1.xlsx")
data
data_cleaned = data.dropna()
print(data_cleaned)

#PSP DATASET
#q1.The mean, median, mode and quartile of each Independent and dependent parameter

#1.for the independent variables
print("this is mean,median,quartile and mode for the independent data in this datset ")
ind_stat = {}
ind_var=['AT','V','AP','RH']
dep_var='PE'

for var in ind_var:

  ind_data=data_cleaned[var].values.astype(int)
  mean=np.mean(ind_data)

  median=np.median(ind_data)

  q1,q2,q3=np.percentile(ind_data,[25,50,75])

  ind_stat[var]={"mean":mean,"median":median,"q1":q1,"q2":q2,"q3":q3}

for variable, stats in ind_stat.items():
    print(variable)
    for key, value in stats.items():
        print(f"{key}: {value}")
    print()


#2.for the dependent variables
print("this is mean,median,quartile and mode for the dependent data in this datset")
print(f"mean: {np.mean(data_cleaned[dep_var])}")
print(f"median: {np.median(data_cleaned[dep_var])}")
print(f"quartile: {np.percentile(data_cleaned[dep_var],[25,50,75])}")

#q2.Determine the corelation of each independent parameter with the dependent parameter

correlation=data_cleaned[ind_var+[dep_var]].corr()

 #we have to connvert dependent variable in the list hence we used a bracket for the dependent varaiable

print(correlation)

#q3.Draw a Histogram of any Independent parameters with the dependent variables
for i in ind_var:
  plt.hist2d(data_cleaned[i],data_cleaned[dep_var],bins=(20,20), cmap=plt.cm.viridis)
  plt.xlabel(i)
  plt.ylabel(dep_var)
  plt.title("relation between independent and dependent variales in datta")
  plt.show()

#q4.Determine the Regression coefficients and the Intercept.
x=data_cleaned[['AT','V','AP','RH']]
y=data_cleaned['PE']
lr=LinearRegression()
lr.fit(x,y)
print(f"regression coefficient:{lr.coef_} ")
print()
print(f"intersept:{lr.intercept_}")

#q5.Calculate the corelation corelation between Actual Dependent parameter s and the Predicted Dependent values (In testing/validation phase)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)
lr.fit(X_train,y_train)
y_predict=lr.predict(X_test)
correlation = y_test.corr(pd.Series(y_predict))
print(f"correalation between actual values and predicted values is :{correlation}")

#q6.Draw a scatter plot between Actual Dependent parameter s and the Predicted Dependent values (In testing/validation phase)
plt.scatter(y_test,y_predict,color="red")
plt.xlabel("actual data")
plt.ylabel("predicted data")
plt.title("relation between actual and predicted data using sactter plot")
plt.show()

