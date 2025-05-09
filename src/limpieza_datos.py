import pandas as pd

#Cargar el dataset
file_path = "../data/HURTO_PERSONAS.csv"
try:
    df = pd.read_csv(file_path, encoding="UTF-8")
except FileNotFoundError:
    df = pd.read_csv(file_path, encoding="latin1")
    

# Normalizar nombres de las columnas (Eliminar espacios si los hay)
df.columns = df.columns.str.strip().str.upper()

# Filtrar solo datos de la ciudad de Cali
df_cali = df[df['MUNICIPIO'].str.upper() == 'CALI'].copy()

# Convertir la columna 'FECHA' a tipo datetime
df_cali['FECHA'] = pd.to_datetime(df_cali['FECHA HECHO'], format='%d/%m/%Y', errors='coerce')

# Eliminar filas con fechas no válidas
df_cali = df_cali.dropna(subset=['FECHA HECHO'])

# Crear columnas adicionales para año, mes y día
df_cali['AÑO'] = df_cali['FECHA'].dt.year
df_cali['MES'] = df_cali['FECHA'].dt.month
df_cali['DIA'] = df_cali['FECHA'].dt.day

# Guardar el dataset limpio
df_cali.to_csv('../data/HURTO_PERSONAS_CALI.csv', index=False, encoding='utf-8')

print("Dataset limpio guardado en '/data/HURTO_PERSONAS_CALI.csv'")

