from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import sklearn

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == "POST":
        print(request.form.get("v1"))
        print(request.form.get("v2"))
        print(request.form.get("v3"))
        print(request.form.get("v4"))
        try:
            v1 = float(request.form['v1'])
            v2 = float(request.form['v2'])
            v3 = float(request.form['v3'])
            v4 = float(request.form['v4'])
            input_features = np.array([v1, v2, v3, v4])
            print("input_features: ", input_features)
            input_features = input_features.reshape(1, -1)
            model = open("linear_regression_model.pkl", "rb")
            clf_model = joblib.load(model) # 加载模型
            model_prediction = clf_model.predict(input_features) # 模型预测
            print("model_prediction: ", model_prediction)
            model_prediction = round(float(model_prediction), 2) # 取小数点后两位
        except ValueError:
            return "input feature value error"

    return render_template("predict.html", prediction=model_prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
