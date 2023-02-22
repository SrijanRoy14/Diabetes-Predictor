#importing the dependencies
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
#data collection and analysis
#loading the dataset
diabetes_ds=pd.read_csv('C:/Users/SRIJAN/Dropbox/PC/Documents/My_C/diabetes.csv')
#seperating data and labels
x=diabetes_ds.drop(columns='Outcome',axis=1)
y=diabetes_ds['Outcome']
#data standardization
scaler=StandardScaler()
scaler.fit(x)
standardized_data=scaler.transform(x)
X=standardized_data
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,stratify=y,random_state=2)
classifier=svm.SVC(kernel='linear')
classifier.fit(x_train,y_train)
#model evaluation
#accuracy score
x_train_prediction=classifier.predict(x_train)
trainningdataaccuracy=accuracy_score(x_train_prediction,y_train)
print('Accuracy score of the training data: ',trainningdataaccuracy)
x_test_prediction=classifier.predict(x_test)
testdataaccuracy=accuracy_score(x_test_prediction,y_test)
print('Accuracy score of the training data: ',testdataaccuracy)
#making a predictive system
input_data=(5,166,72,19,175,25.8,0.587,51)
ip_data_np_array=np.asarray(input_data)
#reshaping the array as we are predicting for one instance
ip_data_reshaped=ip_data_np_array.reshape(1,-1)
#standardizing the data
std_data=scaler.transform(ip_data_reshaped)
#prediction
prediction=classifier.predict(std_data)
if prediction[0]==1:
  print("The patient is diabetic")
else:
  print("The patient is not diabetic")