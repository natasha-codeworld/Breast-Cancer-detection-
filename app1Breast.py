#import libraries
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

#Initialize the flask App
app = Flask(__name__)
with open('model.pkl','rb') as file:
    clf = pickle.load(file)
model = pickle.load(open('Breast_Cancer.sav', 'rb'))

#default page of our web-app
@app.route('/')
def home():
    return render_template('index1.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST','GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    features_name = ['mean_radius', 'mean_texture','mean_perimeter']

    output = clf.predict([features_name])
    if output == 1:
        res_val = "breast cancer"
    else:
        res_val = "no breast cancer"
    return render_template('index1.html', prediction_text='Patient has :{}'.format(res_val))
    
if __name__ == "__main__":
    app.run(debug=True)