import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def sigmoid(x):
  # Сигмоидная функция активации: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # Производная сигмоиды: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

def mse_loss(y_true, y_pred):
  # y_true и y_pred - массивы numpy одинаковой длины.
  return ((y_true - y_pred) ** 2).mean()

class OurNeuralNetwork:
  label_encoder = LabelEncoder()

  def __init__(self):
    # Веса
    self.w1 = np.random.normal()
    self.w2 = np.random.normal()
    self.w3 = np.random.normal()
    self.w4 = np.random.normal()
    self.w5 = np.random.normal()
    self.w6 = np.random.normal()
    self.w7 = np.random.normal()
    self.w8 = np.random.normal()
    self.w9 = np.random.normal()
    self.we1 = np.random.normal()
    self.we2 = np.random.normal()
    self.we3 = np.random.normal()

    # Пороги
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()
    self.b4 = np.random.normal()

  def encode_labels(self, x):
    return label_encoder.fit_transform((x))

  def feedforward(self, x):
    # x is a numpy array with 3 elements.
    h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2] + self.b1)
    h2 = sigmoid(self.w4 * x[0] + self.w5 * x[1] + self.w6 * x[2] + self.b2)
    h3 = sigmoid(self.w7 * x[0] + self.w8 * x[1] + self.w9 * x[2] + self.b3)
    o1 = sigmoid(self.we1 * h1 + self.we2 * h2 + self.we3 * h3 + self.b4)
    return o1

  def train(self, data, all_y_trues):
    '''
    - data - массив numpy (n x 2) numpy, n = к-во наблюдений в наборе.
    - all_y_trues - массив numpy с n элементами.
      Элементы all_y_trues соответствуют наблюдениям в data.
    '''
    learn_rate = 0.1
    epochs = 1000 # сколько раз пройти по всему набору данных

    for epoch in range(epochs):
      for x, y_true in zip(data, all_y_trues):
        # --- Прямой проход (эти значения нам понадобятся позже)
        sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2] + self.b1
        h1 = sigmoid(sum_h1)

        sum_h2 = self.w4 * x[0] + self.w5 * x[1] + self.w6 * x[2] + self.b2
        h2 = sigmoid(sum_h2)

        sum_h3 = self.w7 * x[0] + self.w8 * x[1] + self.w9 * x[2] + self.b3
        h3 = sigmoid(sum_h3)

        sum_o1 = self.we1 * h1 + self.we2 * h2 + self.we3 * h3 + self.b4
        o1 = sigmoid(sum_o1)
        y_pred = o1

        # --- Считаем частные производные.
        # --- Имена: d_L_d_w1 = "частная производная L по w1"
        d_L_d_ypred = -2 * (y_true - y_pred)

        # Нейрон o1
        d_ypred_d_we1 = h1 * deriv_sigmoid(sum_o1)
        d_ypred_d_we2 = h2 * deriv_sigmoid(sum_o1)
        d_ypred_d_we3 = h3 * deriv_sigmoid(sum_o1)
        d_ypred_d_b4 = deriv_sigmoid(sum_o1)

        d_ypred_d_h1 = self.we1 * deriv_sigmoid(sum_o1)
        d_ypred_d_h2 = self.we2 * deriv_sigmoid(sum_o1)
        d_ypred_d_h3 = self.we3 * deriv_sigmoid(sum_o1)

        # Нейрон h1
        d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
        d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
        d_h1_d_w3 = x[2] * deriv_sigmoid(sum_h1)
        d_h1_d_b1 = deriv_sigmoid(sum_h1)

        # Нейрон h2
        d_h2_d_w4 = x[0] * deriv_sigmoid(sum_h2)
        d_h2_d_w5 = x[1] * deriv_sigmoid(sum_h2)
        d_h2_d_w6 = x[2] * deriv_sigmoid(sum_h2)
        d_h2_d_b2 = deriv_sigmoid(sum_h2)

        # Нейрон h3
        d_h3_d_w7 = x[0] * deriv_sigmoid(sum_h3)
        d_h3_d_w8 = x[1] * deriv_sigmoid(sum_h3)
        d_h3_d_w9 = x[2] * deriv_sigmoid(sum_h3)
        d_h3_d_b3 = deriv_sigmoid(sum_h3)

        # --- Обновляем веса и пороги
        # Нейрон h1
        self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
        self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
        self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w3
        self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

        # Нейрон h2
        self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
        self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w5
        self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w6
        self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

        # Нейрон h3
        self.w7 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w7
        self.w8 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w8
        self.w9 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w9
        self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_b3

        # Нейрон o1
        self.we1 -= learn_rate * d_L_d_ypred * d_ypred_d_we1
        self.we2 -= learn_rate * d_L_d_ypred * d_ypred_d_we2
        self.we3 -= learn_rate * d_L_d_ypred * d_ypred_d_we3
        self.b4 -= learn_rate * d_L_d_ypred * d_ypred_d_b4

      # --- Считаем полные потери в конце каждой эпохи
      if epoch % 10 == 0:
        y_preds = np.apply_along_axis(self.feedforward, 1, data)
        loss = mse_loss(all_y_trues, y_preds)
        print("Epoch %d loss: %.3f" % (epoch, loss))

  def save_weights(self):
    with open("weights", 'wb') as pkl:
      pickle.dump(
        [self.w1, self.w2, self.w3, self.w4, self.w5, self.w6, self.w7, self.w8, self.w9, self.we1, self.we2, self.we3,
         self.b1, self.b2, self.b3, self.b4], pkl)

  def set_weights(self, weights):
      self.w1 = weights[0]
      self.w2 = weights[1]
      self.w3 = weights[2]
      self.w4 = weights[3]
      self.w5 = weights[4]
      self.w6 = weights[5]
      self.w7 = weights[6]
      self.w8 = weights[7]
      self.w9 = weights[8]
      self.we1 = weights[9]
      self.we2 = weights[10]
      self.we3 = weights[11]
      self.b1 = weights[12]
      self.b2 = weights[13]
      self.b3 = weights[14]
      self.b4 = weights[15]


# Определим набор данных
df = pd.read_excel("model/Feet.xlsx")
label_encoder = LabelEncoder()
network = OurNeuralNetwork()
all_y_trues = network.encode_labels(df["Пол"])
data = df.drop(["Пол"], axis=1).drop(["Номер"], axis=1)
data = np.array(data)

# Обучаем нашу нейронную сеть!
network.train(data, all_y_trues)

network.save_weights()