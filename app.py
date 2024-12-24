import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
regmodel=pickle.load(open('linear_regression_model.pkl','rb'))

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict_api',methods=['post'])
def predict_api():
    data=request.json['data']
    print(data)
    input=np.array(list(data.values())).reshape(1,-1)
    output=regmodel.predict(input)
    print(output[0])
    return jsonify(output[0])

if __name__=="__main__":
    app.run(debug=True)