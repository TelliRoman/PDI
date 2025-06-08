import os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

X = []
y = []

# Leer archivos
for fname in os.listdir("Trabajo Final de PDI/datos_dolor/"):
    if fname.endswith(".csv"):
        label = "no_dolor" if "no_dolor" in fname.lower() else "dolor"
        vec = np.loadtxt(os.path.join("Trabajo Final de PDI/datos_dolor/", fname), delimiter=",")
        # Elegí una de las dos líneas siguientes:
        features = vec
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

# Entrenar modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)

# Evaluar
print(classification_report(y_test, clf.predict(X_test)))

# Guardar modelo
with open("Trabajo Final de PDI/modelo_dolor.pkl", "wb") as f:
    pickle.dump(clf, f)