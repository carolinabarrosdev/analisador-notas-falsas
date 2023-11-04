# Importando bibliotecas
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Carregando o modelo KMeans previamente treinado
kmeans_model = KMeans(n_clusters=2)
df_fakebills = pd.read_csv('fake_bills_unsuperv.csv', index_col=0)
df_fakebills.dropna(inplace=True) # Limpeza dos registros NA
kmeans_model.fit(df_fakebills)

# Título do aplicativo
st.title("Detecção de Cédulas Falsas")

# Descrição do aplicativo
st.write("Este aplicativo permite que você insira os valores das features e preveja se a cédula é verdadeira ou falsa com base em um modelo KMeans previamente treinado.")

# Entrada das features
feature1 = st.number_input("Comprimento da cédula:", min_value=df_fakebills['length'].min(), max_value=df_fakebills['length'].max())
feature2 = st.number_input("Margem inferior da cédula:", min_value=df_fakebills['margin_low'].min(), max_value=df_fakebills['margin_low'].max())

# Botão para executar a inferência
if st.button("Prever"):
    # Criando um DataFrame com as features inseridas pelo usuário
    user_data = pd.DataFrame({
            'diagonal': [0],
            'height_left': [0],
            'height_right': [0],
            'margin_low': [feature2],
            'margin_up': [0],
            'length': [feature1] 
        })

    # Realizando a previsão com o modelo KMeans
    prediction = kmeans_model.predict(user_data)

    # Exibindo o resultado
    if prediction[0] == 0:
        st.write("A cédula é verdadeira.")
    else:
        st.write("A cédula é falsa.")
