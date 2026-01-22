

import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

data=pd.read_csv(r"/content/kyphosis.csv")
print(data)

print(data.head())
print(data["Kyphosis"].value_counts())

b = LabelEncoder()
data["Kyphosis"] = b.fit_transform(data["Kyphosis"])

print(data.head())

X=data.drop("Kyphosis",axis=1)
Y=data["Kyphosis"]

print(X.head())
print(Y.head())

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

print(X_train.shape)
print(X_test.shape)

model=LogisticRegression()
model.fit(X_train,Y_train)
print("trained")

y_pred=model.predict(X_test)
print(accuracy_score(Y_test,y_pred))
print(confusion_matrix(Y_test,y_pred))
print(classification_report(Y_test,y_pred))

c=DecisionTreeClassifier()
c.fit(X_train,Y_train)
print("trained")

d=RandomForestClassifier(n_estimators=100,
                         random_state=42
                         )
d.fit(X_train,Y_train)
d_pred=d.predict(X_test)
print(accuracy_score(Y_test,d_pred))
print(confusion_matrix(Y_test,d_pred))
print(classification_report(Y_test,d_pred))
print("trained")

dt_pred = c.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(Y_test, y_pred))
print("Decision Tree Accuracy:", accuracy_score(Y_test, dt_pred))
print("Random Forest Accuracy:", accuracy_score(Y_test, d_pred))



with open("kyphosis_model.pkl", "wb") as f:
    pickle.dump(d, f)

print("Model saved successfully!")

with open("kyphosis_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

test_sample = [[100, 5, 12]]
result = loaded_model.predict(test_sample)

print("\nLoaded Model Prediction:")
print("Kyphosis Present" if result[0] == 1 else "Kyphosis Absent")


