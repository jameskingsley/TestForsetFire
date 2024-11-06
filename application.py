import pickle
from flask import Flask, render_template,request  # Removed unused imports: jsonify
import numpy as np  # This import is currently unused but retained if you plan to use it
import pandas as pd  # This import is currently unused but retained if you plan to use it
from sklearn.preprocessing import StandardScaler  # Retained in case you plan to use it later

# Import lasso regressor and standard scaler pickle
lasso_model = pickle.load(open('Models/lasso.pk1', 'rb'))
standard_scaler = pickle.load(open('Models/scaler.pk1', 'rb'))

# Initializing Flask app
application = Flask(__name__)
app = application

#@app.route("/")
#def index():
    #return render_template("index.html")
    
@app.route("/", methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        WS = float(request.form.get('WS'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled = standard_scaler.transform([[Temperature,RH,WS,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = lasso_model.predict(new_data_scaled)
        
        return render_template('home.html', results=result[0])
    else:
        return render_template('home.html')



if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 8080, debug=True)
