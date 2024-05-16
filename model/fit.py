import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

iris_df = pd.read_csv("IRIS.csv")
X = iris_df.drop(["species"], axis=1)
Y = iris_df["species"]

wine_data = pd.read_excel("WineSet.xlsx")
classify_X = wine_data.drop("Страна", axis=1)
classify_Y = wine_data["Страна"]
model_BT = DecisionTreeClassifier(criterion="entropy")
model_BT.fit(classify_X, classify_Y)

with open("BT", 'wb') as pkl:
    pickle.dump(model_BT, pkl)

X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X, Y, test_size=0.3, random_state=3)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train1, Y_train1)
with open('Iris_pickle_file', 'wb') as pkl:
    pickle.dump(model, pkl)
