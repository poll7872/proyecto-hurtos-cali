import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Cargar el dataset limpio
df = pd.read_csv('../data/HURTO_PERSONAS_CALI.csv')

# Convertir la columna 'FECHA' a datetime
df['FECHA'] = pd.to_datetime(df['FECHA'])

# Si no existen las columnas de día, mes y año, crearlas
if 'DIA' not in df.columns:
    df['DIA'] = df['FECHA'].dt.day
if 'MES' not in df.columns:
    df['MES'] = df['FECHA'].dt.month
if 'AÑO' not in df.columns:
    df['AÑO'] = df['FECHA'].dt.year


# Definir las características (X) y la variable objetivo (y)
X = df[['DIA', 'MES', 'AÑO']]
y = df['CANTIDAD']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Hacer predicciones
y_pred = modelo.predict(X_test)

# Evaluar el modelo
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Imprimir los resultados
print("Modelo entrenado con éxito")
print(f"MAE (Error absoluto medio): {mae:.2f}")
print(f"RMSE (Raíz del error cuadrático medio): {rmse:.2f}")
print(f"R2 (Coeficiente de determinación): {r2:.2f}")

# Importancia de las variables
importancias = modelo.feature_importances_
for col, imp in zip(X.columns, importancias):
    print(f"Importancia de {col}: {imp:.4f}")