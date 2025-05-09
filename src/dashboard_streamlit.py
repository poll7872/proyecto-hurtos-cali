import streamlit as st
import pandas as pd
import numpy as np
import datetime 
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# ---Configuración Inicial---
st.set_page_config(
    page_title="Dashboard de HURTO_PERSONAS_CALI",
    page_icon=":guardsman:"
)

## -- Titulo del Dashboard --
st.title("Prediccion de Hurto de Personas en Cali")
st.markdown("Esté prototipo tiene como objetivo predecir la cantidad de hurto de personas en la ciudad de Cali, basado en los datos históricos de los delitos cometidos. Utilizando el modelo de Random.")


## -- Cargar el modelo y entrenar (rapido por que es un prototipo) --
def cargar_modelo_y_datos():
    df = pd.read_csv("../data/HURTO_PERSONAS_CALI.csv", encoding="UTF-8")
    df['FECHA'] = pd.to_datetime(df['FECHA'])
    df['DIA'] = df['FECHA'].dt.day
    df['MES'] = df['FECHA'].dt.month
    df['AÑO'] = df['FECHA'].dt.year
    X = df[['DIA', 'MES', 'AÑO']]
    y = df['CANTIDAD']
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X, y)
    return modelo

modelo = cargar_modelo_y_datos()

# -- Entrada del usuario --
st.subheader("📅 Ingrese una fecha")
fecha = st.date_input("Fecha", value=datetime.date(2025, 5, 1))
anio = fecha.year
mes = fecha.month   
dia = fecha.day

# -- Prediccion --
entrada = pd.DataFrame([[dia, mes, anio]], columns=['DIA', 'MES', 'AÑO'])
prediccion = modelo.predict(entrada)[0]


st.success(f"Predicción estimada de hurtos para el {fecha.strftime('%d/%m/%Y')}: **{int(prediccion)} casos**", icon="✅")

# Descargar reporte simple
st.subheader("📄 Generar reporte")
reporte_simple = f"""
REPORTE DE PREDICCIÓN - HURTO DE PERSONAS

Fecha: {fecha.strftime('%d/%m/%Y')}
Predicción: {int(prediccion)} hurtos

Modelo utilizado: Random Forest
Entrenado con datos abiertos de hurtos desde datos.gov.co

"""

st.download_button(
    label="⬇️ Descargar reporte (.txt)",
    data=reporte_simple,
    file_name=f"reporte_hurtos_{fecha.strftime('%Y%m%d')}.txt",
    mime="text/plain"
)