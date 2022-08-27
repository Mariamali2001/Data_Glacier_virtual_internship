import numpy as np 
import pandas as pd
import pickle 

data = pd.read_csv('E:/solo projects/Data_Glacier_virtual_internship/Data_Glacier_virtual_internship/week 4/toy_dataset.csv')



X=data.iloc[:,:4]
Y=data.iloc[:,-1]

from sklearn.linear_model import LinearRegression
regressor =LinearRegression() 

regressor.fit(X,Y)

pickle.dump(regressor, open('model.pkl','wb')) 


model =pickle.load(open('model.pkl','rb')) 
print(model.predict([[2, 2200, 5]]))