from flask import Flask , request,render_template
import numpy as np
import pickle 


app=Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)] 
    prediction = model.predict(final_features) 
    return render_template('index.html', prediction_text='The flower species is$ {}'.format(prediction))

if __name__ == '__main__':
     app.run(debug=True)