# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 13:59:02 2022

@author: HP
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as metric
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.preprocessing import StandardScaler

df=pd.read_csv(r'C:\Users\HP\Downloads\Lesson1\Social_Network_Ads.csv')
print(df.head())
#datasetin özelliklerine bakıyoruz
print(df.describe().T)

#eksik veri kontrolü yaparız
print(df.isnull().sum()) # eksik veri yok

#aykırı gözlem kontrolü bu yüzden numeric değerleri alırım
df1 = df.select_dtypes(include = ['float64', 'int64']) 

df1=df1.drop("User ID",axis=1) #ID olduğu içn aykırı gözlem kısmına dahil etmedim

df1_age=df1["Age"]
sns.boxplot(x=df1_age) #aykırı değer yok

df1_es=df1["EstimatedSalary"]
sns.boxplot(x=df1_es) #aykırı değer yok

df1_purchase=df1["Purchased"]
sns.boxplot(x=df1_purchase) #aykırı değer yok

#datasetteki sütunların tipine bakıp kategorik değer varsa işleme tabi tutarız
print(df.dtypes) #gender değişkeni kategorik

#Burada değişkenler sıralı olmadığı için One Hot Encoding yöntemini kullandım.
pd.get_dummies(df, columns=["Gender"])
df=pd.get_dummies(df, columns=["Gender"], drop_first=True) 

x = df.drop("User ID",axis=1) #user ID değişkenini prediction yaparken kullanamayacağım için sildim
x = df.drop("Purchased",axis=1) #Purchased bağımlı değişken olduğu için bağımsız değişkenlerin olduğu kısımda yer almamalı
y=df["Purchased"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)
sc = StandardScaler()

x_train = sc.fit_transform(x_train) 
x_test = sc.fit_transform(x_test)

model=LinearRegression()
model.fit(x_train,y_train)
pred=model.predict(x_test)
print("MAE",(metric.mean_absolute_error(pred, y_test)))
print("MSE",(metric.mean_squared_error(pred,y_test)))
print("R2",(metric.r2_score(pred,y_test)))

classifier= RandomForestClassifier(n_estimators= 10, criterion='gini') #araştırdığımda criterion için mse ve mae değişkenleri de vardı fakat uygulama sadce gini'yi kabul etti
classifier.fit(x_train, y_train)  
y_pred= classifier.predict(x_test)
print(accuracy_score(y_test,y_pred))  



