import streamlit as st
import pandas as pd
import joblib
import os


st.title("Hello mundovisk")

#bosta = st.text_input("qual o teu nome pvt", vakue="digite aqui")
#if st.button(label="clique aqui"):
#    st.success(f"seja bem vindo {bosta}")

#ETAPA 1 - definição de features
FEATURES_NAMES = [ 
    'nota_p1',
    'nota_p2',
    'media_trabalhos',
    'frequencia',
    'reprovacoes_anteriores',
    'acessos_plataforma_mes'
]

COLUNAS_HISTORICO = FEATURES_NAMES + ["previsao_resultado", 'prob_aprovado', 'prob_reprovado']

#criar uma sessão
if 'historico_previsoes' not in st.session_state:
    st.session_state.historico_previsoes = pd.DataFrame(columns=COLUNAS_HISTORICO)


#ETAPA 2 - carregamento do moledo para o nosso front end
#st.cache_resource para carregar o modelo apenas uma vez
#otimizando o desempenho do aplicativo

@st.cache_resource
def carregar_modelo(caminho_modelo = "modelo_previsao_desempenho.joblib"):
    try:
        if os.path.exists(caminho_modelo):
            modelo = joblib.load(caminho_modelo)
            return modelo
        else:
            st.error(f"erro epico do {caminho_modelo}")
            st.warning("por favor, execute o script 'modelo_treinamento.py' para gerar o modelo " )
    except Exception as e:
        st.error("erro exotico pprt")

pipeline_modelo = carregar_modelo()

st.set_page_config(layout='wide', page_title='previsão de notas')

st.title("previsor de desempenho academico")
st.markdown( """
    essa ferramenta usa inteligencia artifical para prever o status final (aprovado ou reprovado de um aluno com base em seu desempenho parcial
""")

#ETAPA 3 - formulario de entrada

if pipeline_modelo is not None:
    #utilizar um formulario para agrupar as entradas e o botao
    with st.form('formulario_previsao'):
        st.subheader('insira as notas e metricas do aluno')

        col1, col2 = st.columns(2)

        with col1:
            nota_p1 = st.slider("nota da prova 1 (0 a 10)", min_value=0.0, max_value=10.0, value=5.0, step=0.5)
            media_trabalho = st.slider("media dos trabalhos ( 0 a 10)", min_value=0.0, max_value=10.0, value=5.0, step=0.5)
            numero_reprovacoes = st.number_input('reprovações anteriores', min_value=0.0, max_value=10.0, value=5.0, step=1.0)
        with col2:
            nota_p2 = st.slider("nota da prova 2 (0 a 10", min_value=0.0, max_value=10.0, value=5.0, step=0.5)
            frequencia = st.slider("frequencia (%)", min_value=0.0, max_value=100.0, value=75.0, step=5.0)
            acesso_mes = st.number_input("media de acesso a plataforma (por mes)", min_value=0.0, max_value=100.0, value=1.0, step=1.0)

        submitted = st.form_submit_button("realizar previsão")
    
    if submitted:

        features_name = [
            'nota_p1',
            'nota_p2',
            'media_trabalhos',
            'frequencia',
            'reprovacoes_anteriores',
            'acesso_plataforma_mes'
        ]

        #criação de um dataframe a partir dos dados inseridos
        dados_alunos = pd.DataFrame(
            [(nota_p1, nota_p2, media_trabalho, frequencia, numero_reprovacoes, acesso_mes)]
        )

        st.info("processando dados e realizando a previsão...")

        try:
            #realizar a previsão ([0] ou [1])
            previsao = pipeline_modelo.predict(dados_alunos)

            #obter a probabilidade ()
            probabilidade = pipeline_modelo.predict_proba(dados_alunos)

            prob_reprovados = probabilidade[0][0]
            prob_aprovados = probabilidade[0][1]
            resultado_texto = "APROVADO" if previsao[0] == 1 else "REPROVADO"

            #EXIBIR OS RESULTADO
            st.subheader("resultado da previsão")
            if previsao[0] == 1:
                st.success("previsão: Aprovado")
                st.markdown(f"""
                            com base nos resultados fornecidos, o modelo prevê que o aluno tem: {prob_aprovados*100:.2f}% de chance de ser aprovado
                            
                            Chance de reprovação: {prob_reprovados*100:.2f}%
                            """)
            else:
                st.error("previsão: Reprovado (Area de Risco)")
                st.markdown(f"""
                            com base nos resultados fornecidos, o modelo prevê que o aluno tem: {prob_reprovados*100:.2f}% de chance de ser reprovado
                            
                            Chance de aprovação: {prob_aprovados*100:.2f}%
                            """)
            
            nova_linha_dict = {
                'nota_p1' : nota_p1,
                'nota_p2' : nota_p2,
                'media_trabalhos' : media_trabalho,
                'frequencia' : frequencia,
                'reprovacoes_anteriores' : numero_reprovacoes,
                'acessos_plataforma_mes' : acesso_mes,
                'previsao_resultado' : resultado_texto,
                'prob_aprovado' : prob_aprovados,
                'prob_reprovado' : prob_reprovados
            }

            nova_linha_df = pd.DataFrame([nova_linha_dict], columns=COLUNAS_HISTORICO)

            st.session_state.historico_previsoes = pd.concat(
                [st.session_state.historico_previsoes, nova_linha_df],
                ignore_index=True
            )

        except Exception as e:
            st.error(f"chapou prç: {e}")
            st.error("verifique se os dados estão inseridos corretamente")

    st.subheader("historico de previsões realizadas na sessão:")
    if st.session_state.historico_previsoes.empty:
        st.write("nenhuma previsao foi realizada ainda")
    else:
        st.dataframe(st.session_state.historico_previsoes, use_container_width=True)

        if st.button("Limpar Histórico"):
            st.session_state.historico_previsoes = pd.DataFrame(columns = COLUNAS_HISTORICO)

            st.rerun
else:
    st.warning("chapou cz")