import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import numpy as np

# Cargar el dataset
df = pd.read_csv("data/Airline_customer_satisfaction.csv")
y_true = df["satisfaction"].apply(lambda x: 1 if x == "satisfied" else 0)

columnas_utiles = ["Age", "Type of Travel", "Class", "Flight Distance", 
                   "Inflight entertainment", "On-board service", "Cleanliness", 
                   "Arrival Delay in Minutes", "Departure Delay in Minutes"]

# Preparar X original
X_original = df[columnas_utiles].copy()

# Acumuladores de probabilidades
acumulador_proba = np.zeros(len(df))

# Contador de modelos válidos
n_modelos = 0

for model_file in os.listdir("models"):
    model_id = model_file.split("_")[1].split(".")[0]
    try:
        model = joblib.load(f"models/{model_file}")
        encoders = joblib.load(f"encoders/encoder_{model_id}.pkl")

        X = X_original.copy()
        for col in X.select_dtypes(include='object').columns:
            le = encoders.get(col)
            if le:
                X[col] = le.transform(X[col])

        y_prob = model.predict_proba(X)[:, 1]
        acumulador_proba += y_prob
        n_modelos += 1

    except Exception as e:
        print(f"❌ Error al procesar modelo {model_id}: {e}")

if n_modelos == 0:
    raise ValueError("No se pudo cargar ningún modelo correctamente.")

# Promediar las probabilidades
prob_final = acumulador_proba / n_modelos
y_pred_final = (prob_final >= 0.5).astype(int)

# 📊 Matriz de confusión
cm = confusion_matrix(y_true, y_pred_final)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Matriz de Confusión acumulada (promedio de modelos)")
plt.show()

# 📈 Curva ROC
fpr, tpr, _ = roc_curve(y_true, prob_final)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC promedio = {roc_auc:.2f}", color='darkorange')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Curva ROC (promedio de todos los entrenamientos)')
plt.legend(loc="lower right")
plt.grid()
plt.show()
