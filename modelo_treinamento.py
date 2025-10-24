import pandas as pd
import joblib
import os
from sklearn import model_selection, preprocessing, pipeline, linear_model, metrics

def carregar_dados(caminho_arquivo = "historicoAcademico.csv"):
    try:
        if os.path.exists(caminho_arquivo): 
            df = pd.read_csv(caminho_arquivo, encoding="latin1", sep=",")

            print("o arquivo foi carregado")
            return df
        else:
            print("o arquivo não foi encontrado")
            return None
    except Exception as e:
        print("erro inesperado ao carregar o arquivo: " , e)
        return None
        

dados = carregar_dados()
print(dados)