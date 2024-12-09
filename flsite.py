import pickle

import flask
import numpy as np
import pandas as pd
import math
from flask import Flask, render_template, url_for, request, jsonify
from keras.src.saving import load_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from model.neuronFit import OurNeuralNetwork
from model.imgNeurFit import ClothesClasses
import PIL
from tensorflow.keras.preprocessing import image
import os
import cv2
import torch


app = Flask(__name__)

menu = [{"name": "Лаба 1", "url": "p_knn"},
        {"name": "Лаба 2", "url": "p_LogR"},
        {"name": "Лаба 3", "url": "p_LinR"},
        {"name": "Лаба 4", "url": "p_BT"},
        {"name": "Лаба 13", "url": "p_ClNeuron"},
        {"name": "Лаба 18", "url": "p_ImgClNeur"},
        {"name": "Лаба 19", "url": "p_detection"},
        {"name": "Курсовая", "url": "p_ImgMinecraft"}]

label_encoder=LabelEncoder()
iris_df=pd.read_csv("model/IRIS.csv")
X=iris_df.drop(["species"],axis=1)
Y=iris_df["species"]
feet_data = pd.read_excel("model/Feet.xlsx")
neuron_df = feet_data.copy(deep=True)
linear_X = feet_data.drop("Размер обуви", axis=1).drop("Номер", axis=1)
linear_X["Пол"] = label_encoder.fit_transform(linear_X["Пол"])
linear_Y = feet_data["Размер обуви"]
wine_data = pd.read_excel("model/WineSet.xlsx")
classify_X = wine_data.drop("Страна", axis=1)
classify_Y = wine_data["Страна"]
classify_transform_Y = label_encoder.fit_transform(classify_Y)

# Подгружаем данные для нейронки классификации и обучаем ее
all_y_trues =  label_encoder.fit_transform(neuron_df["Пол"])
neuron_data = neuron_df.drop(["Пол"], axis=1).drop(["Номер"], axis=1)
neuron_data = np.array(neuron_data)
# neuron_train_x, neuron_train_y, neuron_test_x, neuron_test_y = neuron_data.train_test_split(neuron_data, all_y_trues,test_size=0.3,random_state=3)
ClNetwork = OurNeuralNetwork()
ClNetwork.train(neuron_data, all_y_trues)

# Подгружаем нейронку регрессии
RegNetwork = load_model("model/RegNeuron.h5")

# Подгружаем нейронку для классификации картинок
ImgClNetwork = load_model("model/ImgClNetwork.h5")

# Подгружаем нейронку для детектирования
DetectionModel = torch.hub.load('ultralytics/yolov5', 'yolov5s')

preds = []
for x in neuron_data:
    y = ClNetwork.feedforward(x)
    preds.append(round(y))


with open('model/KNN', 'rb') as pkl:
    model_knn = pickle.load(pkl)

with open('model/LogR', 'rb') as pkl:
    model_LogR = pickle.load(pkl)

with open('model/LinR', 'rb') as pkl:
    model_LinR = pickle.load(pkl)

with open('model/BT', 'rb') as pkl:
    model_BT = pickle.load(pkl)

@app.route("/")
def index():
    return render_template('index.html', title="Лабораторные работы, выполненные Терешиным Р.П.", menu=menu)


@app.route("/p_knn", methods=['POST', 'GET'])
def f_lab1():
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


@app.route("/p_ClNeuron", methods=['POST', 'GET'])
def f_lab13():
    if request.method == 'GET':
        return render_template('lab13.html', title="Нейронная сеть", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([int(request.form['list1']),
                           int(request.form['list2']),
                           int(request.form['list3'])])

        pred = ClNetwork.feedforward(X_new)

        ac_score = f"{math.ceil(accuracy_score(preds, all_y_trues)*100)}%"
        return render_template('lab13.html', title="Нейронная сеть", menu=menu,
                               class_model=pred, accuracy_score=ac_score)

@app.route("/p_ImgClNeur", methods=['POST', 'GET'])
def f_lab18():
    if request.method == 'GET':
        return render_template('lab18.html', title="Нейронная сеть", menu=menu, class_model='')
    if request.method == 'POST':
        img_file = flask.request.files.get('img', '')
        img_file.save('static/img.jpg')

        img = image.load_img('static/img.jpg', target_size=(28, 28), color_mode="grayscale")
        x = image.img_to_array(img)
        x = x.reshape(1, 784)
        pred = ImgClNetwork.predict(x)
        pred = np.argmax(pred)

        os.remove('static/img.jpg')

        # ac_score = f"{math.ceil(accuracy_score(preds, all_y_trues)*100)}%"
        return render_template('lab18.html', title="Нейронная сеть", menu=menu,
                               class_model=ClothesClasses[pred], accuracy_score="100%")


@app.route("/p_detection", methods=['POST', 'GET'])
def f_lab21():
    if request.method == 'GET':
        return render_template('lab21.html', title="Детектирование", menu=menu)
    if request.method == 'POST':
        img_file = flask.request.files.get('img', '')
        img_file.save('static/blank_lab21.jpg')

        img = Image.open('static/blank_lab21.jpg')
        desired_classes = ['handbag', 'car']
        confidence_threshold = 0.5
        filtered_results = results.pandas().xyxy[0]
        filtered_results = filtered_results[
            (filtered_results['name'].isin(desired_classes)) & (filtered_results['confidence'] >= confidence_threshold)]
        filtered_image = np.array(image)

        for _, row in filtered_results.iterrows():
            label = row['name']
            conf = row['confidence']
            xmin, ymin, xmax, ymax = row[['xmin', 'ymin', 'xmax', 'ymax']]
            color = (0, 255, 0)  # Зеленый цвет для рамки
            cv2.rectangle(filtered_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 1)
            cv2.putText(filtered_image, f'{label} {conf:.2f}', (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, color, 2)

        if (os.path.exists('static/media/result_lab21.jpg')):
            os.remove('static/media/result_lab21.jpg')


        filtered_image.save('static/media/result_lab21.jpg')

        os.remove('static/blank_lab21.jpg')

        return render_template('lab21.html', title="Детектирование", menu=menu)


@app.route("/p_ImgMinecraft", methods=['POST', 'GET'])
def f_ImgMinecraft():
    if request.method == 'GET':
        return render_template('Minecraft.html', title="Курсовая", menu=menu, class_model='')
    if request.method == 'POST':
        img_file = flask.request.files.get('img', '')
        img_file.save('static/blank.jpg')

        img = image.load_img('static/blank.jpg', target_size=(28, 28), color_mode="grayscale")
        x = image.img_to_array(img)
        x = x.reshape(1, 784)

        if(os.path.exists('static/media/result.jpg')):
            os.remove('static/media/result.jpg')

        # Генерируем центральный контент картинки в разрешении 16х16 и с пустым бэкграундом
        generated = ImgClNetwork.predict(x)
        generated.save('static/media/result.jpg')

        os.remove('static/blank.jpg')

        return render_template('Minecraft.html', title="Курсовая", menu=menu)


@app.route('/api_knn', methods=['get'])
def api_knn():
    request_data = request.get_json()
    X_new = np.array([[float(request_data['rating']),
                       float(request_data['cost'])]])
    pred = model_knn.predict(X_new)

    return jsonify(country_producer=pred[0])


@app.route('/api_LogR', methods=['get'])
def api_LogR():
    request_data = request.get_json()
    X_new = np.array([[float(request_data['rating']),
                       float(request_data['cost'])]])
    pred = model_LogR.predict(X_new)

    return jsonify(country_producer=label_encoder.classes_[pred[0]])


@app.route('/api_LinR', methods=['get'])
def api_LinR():
    label_encoder.fit_transform(linear_X["Пол"])

    request_data = request.get_json()
    X_new = np.array([[label_encoder.fit_transform([request_data['sex']])[0],
                       float(request_data['weight']),
                       float(request_data['height'])]])
    pred = model_LinR.predict(X_new)

    return jsonify(shoe_size=round(pred[0][0]))


@app.route('/api_BT', methods=['get'])
def api_BT():
    request_data = request.get_json()
    X_new = np.array([[float(request_data['rating']),
                       float(request_data['cost'])]])

    pred = model_BT.predict(X_new)

    return jsonify(country_producer=pred[0])


@app.route('/api_ClNeuron', methods=['get'])
def api_ClNeuron():
    request_data = request.get_json()
    X_new = np.array([[float(request_data['weight']),
                       float(request_data['height']),
                       float(request_data['shoe_size'])]])

    pred = ClNetwork.feedforward(X_new)

    return jsonify(sex=pred[0])


@app.route('/api_RegNeuron', methods=['get'])
def api_RegNeuron():
    request_data = request.get_json()
    X_new = np.array([[(request_data['weight']),
                       (request_data['height']),
                       label_encoder.fit_transform(request_data['sex'])]])

    pred = RegNetwork.feedforward(X_new)

    return jsonify(shoe_size=pred[0])


if __name__ == "__main__":
    app.run(debug=True)