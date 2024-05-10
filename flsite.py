import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template, url_for, request
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

menu = [{"name": "Лаба 1", "url": "p_knn"},
        {"name": "Лаба 2", "url": "p_lab2"},
        {"name": "Лаба 3", "url": "p_lab3"}]

iris_df=pd.read_csv("model/IRIS.csv")
X=iris_df.drop(["species"],axis=1)
Y=iris_df["species"]
model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(X, Y)


@app.route("/")
def index():
    return render_template('index.html', title="Лабораторные работы, выполненные Терешиным Р.П.", menu=menu)


@app.route("/p_knn", methods=['POST', 'GET'])
def f_lab1():
    if request.method == 'GET':
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           float(request.form['list4'])]])
        pred = model_knn.predict(X_new)
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu,
                               class_model="Это: " + pred)

@app.route("/p_lab2")
def f_lab2():
    return render_template('lab2.html', title="Логистическая регрессия", menu=menu)


@app.route("/p_lab3")
def f_lab3():
    return render_template('lab3.html', title="Логистическая регрессия", menu=menu)


if __name__ == "__main__":
    app.run(debug=True)
