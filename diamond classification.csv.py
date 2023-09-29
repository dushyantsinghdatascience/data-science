#IMPORTING LIBS
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as mlt
#IMPORTING DATA SET
data=pd.read_csv('diamonds.csv')
#PERFORMING BASIC EDA
pd.set_option('display.max_columns',50)
data.head()
data.info()
data.isna().sum()
data1=pd.read_csv('diamonds.csv',na_values=[0])
data1.isna().sum()
columns=['x','y','z']
for column in columns:
    mean=data[column].mean()
    data[column]=data[column].replace(0,mean)
data.isna().sum()
data.describe()
 #converting caterigorical to numerical
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
columns=['cut','color','clarity']
for column in columns:
    data[column]=encoder.fit_transform(data[column])
data['cut'].unique()    
data['cut'].value_counts()

data['color'].unique()
data['color'].value_counts()

data['cut'].unique()
data['cut'].value_counts()
data.describe()
#SEGREGATING DATA IN INPUT AND OUTPUT
x=data.drop(['price'],axis=1)
y=data['price']
#FINDING CORR() AND PLOTING GRAPHS
mlt.figure(figsize=(10,8))
sns.pairplot(data)
    
mlt.figure(figsize=(10,8))
sns.heatmap(data.corr(),annot=True)
#SPLTING INTO TRAIN AND TEST
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               test_size=0.20,
                                               random_state=0)
#giving training
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
regressor.coef_
regressor.intercept_
#TESTING
y_pred=regressor.predict(x_test)
y_pred
#QUALITY CHECK
from sklearn import metrics
metrics.mean_squared_error(y_test, y_pred)
metrics.r2_score(y_test,y_pred)

