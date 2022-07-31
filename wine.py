# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 12:56:31 2022

@author: HP
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as metric
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

veri=pd.read_csv(r'C:\Users\HP\Downloads\Lesson1\Wine.csv')
print(veri.head())
#datasetin özelliklerine bakıyoruz
print(veri.describe().T)

#eksik veri kontrolü yaparız
print(veri.isnull().sum()) # eksik veri yok
#datasetteki sütunların tipine bakıp kategorik değer varsa işleme tabi tutarız
print(veri.dtypes) #tüm veriler float ve integer değerden oluşmaktadır
#burada customer segmentte olan değerlere bakıyoruz çünkü tahmin edilmesi gerekn sütun orası
print(veri["Customer_Segment"].unique()) #1 2 3 değerlerine sahip
#veri setini customer_segmentten ayırıyoruz
x=veri.drop("Customer_Segment",axis=1)
y=veri["Customer_Segment"]

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



