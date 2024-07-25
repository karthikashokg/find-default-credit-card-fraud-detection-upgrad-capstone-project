import json
import pickle
import os

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
# Load the scaler object from the pickle file
scaler_path = os.path.join('model', 'scaler.pkl')
with open(scaler_path, 'rb') as f:
    scalar = pickle.load(f)
# Load the XGBoost model from its pickle file
xgb_model_path = os.path.join('model', 'best_xgb_os.pkl')
with open(xgb_model_path, 'rb') as f:
    xgb_model = pickle.load(f)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    
    time = float(data['Time'])
    amount = float(data['Amount'])
    
    # Scale 'Time' and 'Amount' features
    scaled_time = scalar.transform([[time]])[0][0]
    scaled_amount = scalar.transform([[amount]])[0][0]
    
    # Update the 'Time' and 'Amount' values with the scaled values
    data['Time'], data['Amount'] = scaled_time, scaled_amount
    # Make predictions using the trained model
    new_data = np.array(list(data.values())).reshape(1, -1)
    output = xgb_model.predict(new_data)
    print(output[0])
    return jsonify(str(output[0]))

# Define the column order expected by the model
column_order = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

@app.route('/predict',methods=['POST'])
def predict():
    # Extract 'Time' and 'Amount' from the request form
    time = float(request.form['Time'])
    amount = float(request.form['Amount'])
    
    # Scale 'Time' and 'Amount' using the loaded scalar object
    scaled_time = scalar.transform([[time]])[0][0]
    scaled_amount = scalar.transform([[amount]])[0][0]
    
    # Create the final input array with scaled 'Time' and 'Amount' and other features in the correct order
    final_input = [scaled_time] + [float(request.form[col]) for col in column_order[1:29]] + [scaled_amount]  
    
    # Make predictions using the trained XGBoost model
    output = xgb_model.predict(np.array(final_input).reshape(1, -1))[0]
    
    # Map prediction output to transaction label
    transaction_label = "Non-Fraudulent" if output == 0 else "Fraudulent"
    
    return render_template("home.html", prediction_text="The Transaction is {}".format(transaction_label))


    
if __name__ == '__main__':
    app.run(debug=True)
    