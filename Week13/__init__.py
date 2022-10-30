from unittest import result
from flask import Flask , request,render_template
import numpy as np
import pickle

app=Flask(__name__, template_folder='E:/solo projects/Data_Glacier_virtual_internship/Data_Glacier_virtual_internship/Week13/templates/index.html')
model = pickle.load(open('E:/solo projects/Data_Glacier_virtual_internship/Data_Glacier_virtual_internship/Week13/Model/model_gbm.pk','rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    


    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)] 
    prediction = model.predict(final_features)
    
    return render_template('index.html', prediction_text='The Presistent_Flag is {}'.format(prediction))

if __name__ == "__main__":
    # app.run(port =5000,debug=True)
    app.run(host='127.0.0.1', port =5000,debug=True, use_reloader=False)