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



# etapa 2 preparação e divisão dos dados
# definição de X(features) e Y(target)

if dados is not None:
    print(f"\nTotal de registros carregados: {len(dados)}")
    print("iniciando pipeline de treinamento")

    TARGET_COLUMN = "Status_Final"
    

    #etapa 2.1 - definição das features e target
    try:
        X = dados.drop(TARGET_COLUMN, axis=1)
        y = dados[TARGET_COLUMN]
        
        print(f"Features (X) definidas: {list(X.columns)}")
        print(f"Features (Y) definidas: {TARGET_COLUMN} ")
    except KeyError:
        print("erro epico")
        print(f"a coluna {TARGET_COLUMN} não foi encontrado no CSV")
        print(f"colunas disponiveis: {list(dados.columns)}")
        print(f"por favor ajuste a variavel 'TARGET_COLUMN' e tente novamente")
        #se o target n for games o codigo para moments
        exit()


#etapa 2.2 - divisão entre treino e teste
    print("\n dividindo dados em treino e teste")
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y,
        test_size= 0.2,
        random_state= 42, #garantir a reprodutiblidade
        stratify=y #manter a proporção de aprovados e reprovados
    )

    print(f"dados de treino: {len(X_train)} | dados de teste: {len(X_test)}")

    print("criando a pipeline de ML")
    pipeline_model = pipeline.Pipeline([
        ('scaler', preprocessing.StandardScaler()),
        ('model', linear_model.LogisticRegression(random_state=42))
    ])

    #ETAPA 4: treinamento e avaliação dos dados/modelo

    print("\n treinamento do modelo")

    pipeline_model.fit(X_train, y_train)

    print("modelo treinado. avaliando com os dados de test")
    y_pred = pipeline_model.predict(X_test)

    #AVALIAÇÃO DE DESEMPENHO
    Accuracy = metrics.accuracy_score(y_test, y_pred)
    report = metrics.classification_report(y_test, y_pred)

    print("\n relatorio de avaliação geral")
    print(f"acuracia geral: {Accuracy * 100:.2f}%")
    print("\nrelatorio de classificação bigodado:")
    print(report)

    #ETAPA 5 - salvando o modelo
    model_filename = 'modelo_previsao_desempenho.joblib'

    print("\nsalvando o pipeline em: {model_filename})")
    joblib.dump(pipeline_model, model_filename)

    print("\nprocesso concluido com exito")
    print(f"o arquivo '{model_filename}' esta pronto para ser utilizado")

else:
    print("o pipeline nao pode continuar pq vc cagou o pau e os dados tao uma bosta")