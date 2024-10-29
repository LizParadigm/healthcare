from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
import pandas as pd
import pathlib

df = pd.read_csv(pathlib.Path('data/healthcare.csv'))

y = df['stroke'] 
X = df.drop(columns=['id', 'stroke'])

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

pathlib.Path('model').mkdir(parents=True, exist_ok=True)

print('Entrenando modelo...')
clf = RandomForestClassifier(n_estimators=100, random_state=2)
clf.fit(X_train, y_train)

dump(clf, pathlib.Path('model/stroke_model.joblib'))
print("Modelo guardado correctamente en 'model/stroke_model.joblib'")

# Evaluar el modelo
y_pred = clf.predict(X_test)
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))
