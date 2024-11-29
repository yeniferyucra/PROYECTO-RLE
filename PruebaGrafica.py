#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo y el LabelEncoder
modelo = joblib.load('modelo_clasificacion.pkl')
label_encoder = joblib.load('label_encoder.pkl')

st.title('Predicción de Perfil de Lealtad de Clientes para el NPS')

# Cargar datos nuevos
archivo_subido = st.file_uploader("Sube un archivo CSV para hacer predicciones", type=["csv"])

if archivo_subido is not None:
    # Leer los datos subidos
    datos_nuevos = pd.read_csv(archivo_subido)
    st.write("Datos cargados:")
    st.write(datos_nuevos)
    
    # Realizar predicciones
    predicciones_numericas = modelo.predict(datos_nuevos)
    
    # Convertir las predicciones numéricas a las etiquetas originales
    predicciones_etiquetas = label_encoder.inverse_transform(predicciones_numericas)
    
    # Agregar la columna de predicciones al DataFrame original
    datos_nuevos['Predicción'] = predicciones_etiquetas
    
    st.write("Datos con predicciones:")
    st.write(datos_nuevos)

    # Permitir descargar el archivo con las predicciones
    csv = datos_nuevos.to_csv(index=False).encode('utf-8')
    st.download_button(label="Descargar Predicciones", data=csv, file_name='predicciones.csv', mime='text/csv')


# In[ ]:




