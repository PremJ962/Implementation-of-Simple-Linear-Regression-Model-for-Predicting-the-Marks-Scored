# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. To implement the linear regression using the standard libraries in the python.
2. 2.Use slicing function() for the x,y values.
3. 3.Using sklearn library import training , testing and linear regression modules.
4. 4.Predict the value for the y.
5.  5.Using matplotlib library plot the graphs.
6.  6.Use xlabel for hours and ylabel for scores.
7.  7.End the porgram.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: 
RegisterNumber: 



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('student_scores.csv')
dataset.head()
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred = regressor.predict(X_test)
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Score")
plt.show()
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Hours vs scores (Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
*/  
*/
```

## Output:
![image](https://github.com/PremJ962/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/161425730/9d1a6d59-409e-49c9-9a2e-cdabd7ce4874)

![image](https://github.com/PremJ962/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/161425730/4593e28c-71b7-4e9a-976c-9970be0e4c20)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
