import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('train.csv')

df.isnull().sum()

df['Age'].fillna(df['Age'].median(), inplace=True)

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'Q': 0, 'S': 1, 'C' : 2})

df = df.drop(columns=['Name', "Cabin", "Ticket"])

df = df.dropna()

df['Age'] = df['Age'].astype("Int64")
df['Fare'] = df['Fare'].astype("int64")

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(penalty='l2', C=0.1, solver='lbfgs', max_iter=1000, random_state=42)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy Score: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

prob = model.predict_proba(X_test_scaled)[:5]
print("Predicted Probabilities for first 5 test samples:")
for i, prob in enumerate(prob):
    print(f"Sample {i+1}: Didn't Survive: {prob[0]:.4f}, Survived: {prob[1]:.4f}")
