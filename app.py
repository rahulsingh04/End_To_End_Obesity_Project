from flask import Flask,request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__)
app = application

### Route of The Home Page 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            Gender=request.form.get('Gender'),
            Age=request.form.get('Age'),
            Height=request.form.get('Height'),
            Weight=request.form.get('Weight'),
            family_history_with_overweight=request.form.get('family_history_with_overweight'),
            FAVC=request.form.get('FAVC'),
            FCVC=request.form.get('FCVC'),
            NCP=request.form.get('NCP'),
            CAEC=request.form.get('CAEC'),
            SMOKE=request.form.get('SMOKE'),
            CH2O=request.form.get('CH2O'),
            SCC=request.form.get('SCC'),
            FAF=request.form.get('FAF'),
            TUE=request.form.get('TUE'),
            CALC=request.form.get('CALC'),
            MTRANS=request.form.get('MTRANS')        
            )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("BEFORE PREDICTION STAGE :----->")
        predict_pipeline = PredictPipeline()
        print("MID PREDICTION STAGE :--->")
        results = predict_pipeline.predict(pred_df)
        print("AFTER PREDICTION STAGE :-->")
        print("result is :--->",results)
        dict_ = {'Insufficient_Weight': 0, 'Normal_Weight': 1, 'Obesity_Type_I': 2, 'Obesity_Type_II': 3, 'Obesity_Type_III': 4, 'Overweight_Level_I': 5, 'Overweight_Level_II': 6}
        
        results = [k for k,v in dict_.items() if v==results]
        print("Result is :-->",results)

        return render_template("home.html", results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
