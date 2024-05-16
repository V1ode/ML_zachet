import pickle

import numpy as np
import pandas as pd
import math
from flask import Flask, render_template, url_for, request
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

menu = [{"name": "Лаба 1", "url": "p_knn"},
        {"name": "Лаба 2", "url": "p_LogR"},
        {"name": "Лаба 3", "url": "p_LinR"},
        {"name": "Лаба 4", "url": "p_BT"}]

label_encoder=LabelEncoder()
iris_df=pd.read_csv("model/IRIS.csv")
X=iris_df.drop(["species"],axis=1)
Y=iris_df["species"]
feet_data = pd.read_excel("model/Feet.xlsx")
linear_X = feet_data.drop("Размер обуви", axis=1).drop("Номер", axis=1)
linear_X["Пол"] = label_encoder.fit_transform(linear_X["Пол"])
linear_Y = feet_data["Размер обуви"]
wine_data = pd.read_excel("model/WineSet.xlsx")
classify_X = wine_data.drop("Страна", axis=1)
classify_Y = wine_data["Страна"]
classify_transform_Y = label_encoder.fit_transform(classify_Y)


@app.route("/")
def index():
    return render_template('index.html', title="Лабораторные работы, выполненные Терешиным Р.П.", menu=menu)



@app.route("/p_knn", methods=['POST', 'GET'])
def f_lab1():
    with open('model/KNN', 'rb') as pkl:
        model_knn = pickle.load(pkl)

    if request.method == 'GET':
        return render_template('lab1.html', title="Метод k ближайших соседей (KNN)", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2'])]])
        pred = model_knn.predict(X_new)
        ac_score = f"{math.ceil(accuracy_score(model_knn.predict(classify_X), classify_Y)*100)}%"
        return render_template('lab1.html', title="Метод k ближайших соседей (KNN)", menu=menu,
                               class_model="Это: " + pred[0], accuracy_score=ac_score)

    pkl.close()

@app.route("/p_LogR", methods=['POST', 'GET'])
def f_lab2():
    label_encoder.fit_transform(classify_Y)
    with open('model/LogR', 'rb') as pkl:
        model_LogR = pickle.load(pkl)

    if request.method == 'GET':
        return render_template('lab2.html', title="Логистическая регрессия", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2'])]])
        pred = model_LogR.predict(X_new)
        ac_score = f"{math.ceil(accuracy_score(model_LogR.predict(classify_X), classify_transform_Y)*100)}%"
        return render_template('lab2.html', title="Логистическая регрессия", menu=menu,
                               class_model="Это: " + label_encoder.classes_[pred[0]], accuracy_score=ac_score)

    pkl.close()


@app.route("/p_LinR", methods=['POST', 'GET'])
def f_lab3():
    label_encoder.fit_transform(linear_X["Пол"])
    with open('model/LinR', 'rb') as pkl:
        model_LinR = pickle.load(pkl)

    if request.method == 'GET':
        return render_template('lab3.html', title="Линейная регрессия", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[label_encoder.fit_transform([request.form['list1']])[0],
                            float(request.form['list2']),
                            float(request.form['list3'])]])

        pred = model_LinR.predict(X_new)
        all_pred = model_LinR.predict(linear_X)
        new_pred = []
        for i in all_pred:
            new_pred.append(round(all_pred[0][0]))

        ac_score = f"{math.ceil(accuracy_score(np.array(new_pred), np.array(linear_Y))*100)}%"
        return render_template('lab3.html', title="Линейная регрессия", menu=menu,
                               class_model="Размер обуви: " + str(round(pred[0][0])), accuracy_score=ac_score)

    pkl.close()


@app.route("/p_BT", methods=['POST', 'GET'])
def f_lab4():
    with open('model/BT', 'rb') as pkl:
        model_BT = pickle.load(pkl)

    if request.method == 'GET':
        return render_template('lab4.html', title="Дерево решений", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2'])]])
        pred = model_BT.predict(X_new)
        ac_score = f"{math.ceil(accuracy_score(model_BT.predict(classify_X), classify_Y)*100)}%"
        return render_template('lab4.html', title="Дерево решений", menu=menu,
                               class_model="Это: " + pred[0], accuracy_score=ac_score)

    pkl.close()


if __name__ == "__main__":
    app.run(debug=True)
