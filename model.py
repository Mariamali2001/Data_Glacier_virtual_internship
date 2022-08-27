import numpy as np 
import pandas as pd
import pickle 

data = pd.read_csv('toy_dataset.csv')

data = data.replace(['Yes', 'No'], ['0', '1'], 'illness')
data = data.withColumn("ill",data.illness.cast('int'))
data.show()

# X=data.iloc[:,:4]
# Y=data.iloc[:,-1]

# from sklearn.linear_model import LinearRegression
# regressor =LinearRegression() 

# regressor.fit(X,Y)

# pickle.dump(regressor, open('model.pkl','wb')) 


# model =pickle.load(open('model.pkl','rb')) 
# print(model.predict([[2, 2200, 5]]))