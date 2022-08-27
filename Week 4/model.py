import numpy as np 
import pandas as pd
import pickle 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('E:/solo projects/Data_Glacier_virtual_internship/Data_Glacier_virtual_internship/week 4/IRIS.csv')


X = data[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
Y = data["species"]

# split data to train and test
X_train,X_test , Y_train , Y_test = train_test_split(X, Y, test_size=0.2)



regressor =RandomForestClassifier()

#fit the model
regressor.fit(X_train,Y_train)

# make pickle file to the model 

pickle.dump(regressor, open('model.pkl','wb'))


# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

