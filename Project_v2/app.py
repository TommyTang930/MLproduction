import pandas as pd
from flask import Flask, render_template, request
import numpy as np
import logging
import joblib
import sys
import json

app = Flask(__name__, static_folder="static", template_folder="templates")

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

@app.route("/")
def home():
    return render_template('index.html')


@app.route("/prediction", methods=['POST', 'GET'])
def predict():
    if request.method == "POST":
        # 获取表单数据
        try:
            name = request.form['name']
            gender = request.form['gender']
            education = request.form['education']
            self_employed = request.form['self_employed']
            martial_status = request.form['martial_status']
            dependents = request.form['dependents']
            applicant_income = request.form['applicant_income']
            coapplicant_income = request.form['coapplicant_income']
            loan_amount = request.form['loan_amount']
            loan_term = request.form['loan_term']
            credit_history = request.form['credit_history']
            property_area = request.form['property_area']
        except ValueError:
            return render_template("error.html", prediction="Error")
        # 合并
        with open("data/columns_set.json", "r") as file:
            cols = json.loads(file.read())
        columns = cols['data_columns']
        # dependents
        try:
            col = ('Dependents_' + str(dependents))
            if col in columns.keys():
                columns[col] = 1
        except:
            return "Value Error"

        # property area
        try:
            col = ('Property_Area_' + str(property_area))
            if col in columns.keys():
                columns[col] = 1
        except:
            return "Value Error"

        # Gender
        try:
            col = ('Gender_' + gender)
            if col in columns.keys():
                columns[col] = 1
        except:
            return "Value Error"

        # Married
        try:
            col = ('Married_' + martial_status)
            if col in columns.keys():
                columns[col] = 1
        except:
            return "Value Error"

        # Education
        try:
            col = ('Education_' + education)
            if col in columns.keys():
                columns[col] = 1
        except:
            return "Value Error"

        # Self_Employed
        try:
            col = ('Self_Employed_' + self_employed)
            if col in columns.keys():
                columns[col] = 1
        except:
            return "Value Error"

        df = pd.DataFrame(
            data = {k: [v] for k,v in columns.items()},
            dtype=float
        )

        df.fillna(value=0, inplace=True)

        print(df)

        # 加载模型
        clf_model = joblib.load(open("bin/xgboostModel.pkl", "rb"))

        # 预测
        result = clf_model.predict(df)

        print("result: ", result)

        predict_label = "审核通过" if result[0] == 1 else "审核失败"

        return render_template("prediction.html", prediction=predict_label)

if __name__ == "__main__":
    app.run("0.0.0.0", debug=True)
