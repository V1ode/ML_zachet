from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

df = pd.read_excel("Feet.xlsx")
label_encoder=LabelEncoder()
all_y_trues =  label_encoder.fit_transform(df["Размер обуви"])
data = df.drop(["Размер обуви"], axis=1).drop(["Номер"], axis=1)
data = np.array(data.apply(label_encoder.fit_transform))

l0 = tf.keras.layers.Dense(units=3, input_shape=[3,])
l1 = tf.keras.layers.Dense(units=1, activation='linear')
model = tf.keras.Sequential([l0, l1])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
model.fit(data, all_y_trues, epochs=500, verbose=False)

model.save('RegNeuron.h5')