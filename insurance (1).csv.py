
#IMPORTING LIB.
import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as mlt
#IMPORYING THE DATA FRAME 
data=pd.read_csv('insurance (1).csv')
#PERFORMING EDA
data.shape
data.head()
data.tail()
data.columns
data.info()
data.isna()
data.describe()
#FINDING CORR()AND PLOTINGT GRARPHS
sns.heatmap(data.corr(),annot=True)
mlt.scatter(x=data['age'],y=data['charges'],color='red')
mlt.scatter(x=data['bmi'],y=data['charges'],color='darkgray')
mlt.scatter(x=data['children'],y=data['charges'],color='indigo')
#FINDING UNIQUES VALUE AND TURNING CATEGORICAL COLUMN INTO NUMERICAL COLUMNS
data['sex'].unique()
data['sex'].value_counts()

data['smoker'].unique()
data['smoker'].value_counts()

data['region'].unique()
data['region'].value_counts()

from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
data['sex']=encoder.fit_transform(data['sex'])
data['smoker']=encoder.fit_transform(data['smoker'])
data['region']=encoder.fit_transform(data['region'])
#SEGREGATING DATA INTO INPUT AND OUTPUT
x=data.drop(['charges'],axis=1)
y=data['charges']
#SPLITING THE DATA INTO TARIN AND TEST
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
#GIVING TRAING TO THE SYSTEM
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()

regressor.fit(x_train,y_train)
regressor.coef_
regressor.intercept_
#TESTING 
y_pred=regressor.predict(x_test)
y_pred
#QUALITY TESTING
from sklearn import metrics
metrics.mean_squared_error(y_test,y_pred)
np.sqrt(metrics.mean_squared_error(y_test,y_pred))
metrics.mean_absolute_error(y_test,y_pred)
metrics.r2_score(y_test,y_pred)
#END OF REPORT
#thankyou



