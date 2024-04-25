from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction import CustomData,PredictPipeline
import pandas as pd
from src.components.Model_trainer import ModelTrainer

import json



application=Flask(__name__)

app=application



@app.route('/',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        custom_data_input_dict={
                'Age': int(request.form['Age']),
                'Workclass': request.form['Workclass'],
                'Fnlwgt': int(request.form['Fnlwgt']),
                'Education-num': int(request.form['Education-num']),
                'Occupation': request.form['Occupation'],
                'Relationship': request.form['Relationship'],
                'Race': request.form['Race'],
                'Sex': request.form['Sex'],
                'Capital-gain': int(request.form['Capital-gain']),
                'Capital-loss': int(request.form['Capital-loss']),
                'Hours-per-week': int(request.form['Hours-per-week']),
                'Native-country': request.form['Native-country']
            }
        
        final_new_data=pd.DataFrame(custom_data_input_dict,index=[0])
        predict_pipeline=PredictPipeline()
        print(final_new_data)
        pred=predict_pipeline.predict(final_new_data)
        print(pred)
        results = "<=50K" if pred[0]==0 else ">50K"
            # You can then load variables from the same location
        
        return render_template('result.html',results=results)

@app.route('/admin')
def admin():
    with open("variables/Best_model_info.json", 'r') as f:
        loaded_variables = json.load(f)
    models = loaded_variables['models']
    best_model= loaded_variables['best_model_str']
    best_model_score = loaded_variables['best_model_score_str']
    return render_template('admin.html',models=models,best_model=best_model,best_model_score=best_model_score)


if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)

