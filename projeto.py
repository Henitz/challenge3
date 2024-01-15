import textwrap
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
# !pip install statsmodels
# from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.tsa.stattools import acf, pacf
import streamlit as st
from prophet.diagnostics import performance_metrics

# Ignorar os FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# selected_date = '2024-01-25'  # Replace this with the selected date
# selected_time = pd.Timestamp('00:00:00').time()

# from model import modelo
# from prevel_model import prevendo

import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objs as go
import plotly.express as px
import streamlit as st
import os
import streamlit as st
import pandas as pd
import requests

data_selecionada = None
hora_selecionada = None


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape


def prevendo(df2, data1, flag=True):
    from prophet import Prophet
    import pandas as pd

    # Criando o modelo Prophet
    feriados_sp = pd.DataFrame({
        'holiday': 'feriados_sp',
        'ds': pd.to_datetime(['2024-01-25', '2024-02-25', '2024-03-25', '2024-04-25', '2024-05-25',
                              '2024-06-25', '2024-07-25', '2024-08-25', '2024-09-25', '2024-10-25',
                              '2024-11-25', '2024-12-25']),
        'lower_window': 0,
        'upper_window': 0,
    })

    m = Prophet(holidays=feriados_sp)

    # Adicionando feriados semanais (sábados e domingos)
    df2['ds'] = pd.to_datetime(df2['ds'], format='%d.%m.%Y')
    df2['is_weekend'] = (df2['ds'].dt.weekday >= 5).astype(int)
    m.add_regressor('is_weekend')

    m.fit(df2)  # Ajustando o modelo com o DataFrame original

    # Estendendo o período de previsão para incluir janeiro de 2024
    future_dates = pd.date_range(start='2024-01-01', periods=31, freq='D')
    future = pd.DataFrame({'ds': future_dates})
    future['is_weekend'] = (future['ds'].dt.weekday >= 5).astype(int)

    # Fazendo a previsão com base nas datas futuras
    forecast = m.predict(future)

    # Plotando os resultados para o período estendido
    fig_extended = m.plot(forecast)
    if flag:
        st.write(fig_extended)
    flag = False
    # Filtrando os resultados para a data desejada, excluindo feriados e finais de semana
    prediction1 = forecast[(forecast['ds'] == pd.to_datetime(data1)) &
                           (~forecast['ds'].isin(feriados_sp['ds'])) &
                           (~(forecast['ds'].dt.weekday >= 5))]

    if not prediction1.empty:
        yhat_period = prediction1['yhat'].values[0]
        return yhat_period
    else:
        return None


def modelo(df1, data_selecionada1, hora_selecionada1):
    # Adicionando feriados nacionais brasileiros
    feriados_sp = pd.DataFrame({
        'holiday': 'feriados_sp',
        'ds': pd.to_datetime(['2024-01-25', '2024-02-25', '2024-03-25', '2024-04-25', '2024-05-25',
                              '2024-06-25', '2024-07-25', '2024-08-25', '2024-09-25', '2024-10-25',
                              '2024-11-25', '2024-12-25']),
        'lower_window': 0,
        'upper_window': 0,
    })

    # Criando o modelo Prophet
    m = Prophet(holidays=feriados_sp)

    # Converter 'ds' para o formato de data, se necessário
    df1['ds'] = pd.to_datetime(df1['ds'])

    # Adicionando feriados semanais (sábados e domingos)
    # Criar campo 'is_weekend' com 0 e 1 - significando fim de semana
    df1['is_weekend'] = (df1['ds'].dt.weekday >= 5).astype(int)
    m.add_regressor('is_weekend')

    m.fit(df1)

    # Criando o dataframe para previsão futura
    future = m.make_future_dataframe(periods=365)
    future['is_weekend'] = (future['ds'].dt.weekday >= 5).astype(int)
    forecast = m.predict(future)

    # Conversão para arrays para uso em plotagem
    fcst_t = np.array(forecast['ds'].dt.to_pydatetime())
    history_ds = np.array(m.history['ds'].dt.to_pydatetime())

    # Criando o gráfico com Plotly
    fig = go.Figure()

    # Adicionando os dados do histórico
    fig.add_trace(go.Scatter(x=history_ds, y=m.history['y'], mode='markers', name='Histórico'))

    # Adicionando os dados da previsão
    fig.add_trace(go.Scatter(x=fcst_t, y=forecast['yhat'], mode='lines', name='Previsão'))

    # Adicionando a faixa de intervalo
    fig.add_trace(go.Scatter(
        x=np.concatenate([fcst_t, fcst_t[::-1]]),
        y=np.concatenate([forecast['yhat_lower'], forecast['yhat_upper'][::-1]]),
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        name='Intervalo de Confiança'
    ))

    # Personalizando o layout do gráfico
    fig.update_layout(
        xaxis_title='Data',
        yaxis_title='Valores',
        title='Previsão com Prophet e Intervalo de Confiança'
    )

    # Exibindo o gráfico no Streamlit
    st.plotly_chart(fig)

    # Mostrando

    # Calculando métricas de desempenho
    # df_cv = performance_metrics(df)
    # st.write(df_cv)
    # este procedimento está dando erro

    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # Verificar se os DataFrames têm o mesmo número de amostras
    # df e forecast são diferentes pois forecast foi levado em conta os feridos e fins de semana
    # if df1.shape[0] == forecast.shape[0]:
    # Calcular métricas
    #   mae = mean_absolute_error(df1['y'], forecast['yhat'])
    #  mse = mean_squared_error(df1['y'], forecast['yhat'])
    #  rmse = np.sqrt(mse)

    # sf.write(f"MAE: {mae}")
    # sf.write(f"MSE: {mse}")
    # sf.write(f"RMSE: {rmse}")
    # else:
    #   df.write("Os DataFrames não têm o mesmo número de amostras. Verifique os dados.")
    # Calculando as previsões do modelo para os dados de teste
    # Calculando novamente forecast para todos os dados, vai ser prossivel calcular as métricas
    forecast = m.predict(df1)

    # Calculando o MAE entre as previsões e os valores reais
    mae = mean_absolute_error(df1['y'], forecast['yhat'])
    mae_rounded = round(mae, 2)

    st.write(f'MAE: {mae_rounded}')
    mse = mean_squared_error(df1['y'], forecast['yhat'])
    mse_rounded = round(mse, 2)
    rmse = mean_squared_error(df1['y'], forecast['yhat'], squared=False)
    rmse_rounded = round(rmse, 2)
    st.write(f'MSE: {mse_rounded}')
    st.write(f'RMSE: {rmse_rounded}')

    # Supondo que 'forecast' seja o resultado da previsão do Prophet
    # y_true é a série real do seu DataFrame, você pode usar df['y'] ou qualquer outra coluna correspondente
    y_true = df1['y']

    # Obtendo os valores previstos a partir do forecast
    y_pred = forecast['yhat']

    # Calculando o MAPE usando a função importada
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mape = round(mape, 2)
    st.write(f"MAPE: {mape}")

    if data_selecionada1 is not None and hora_selecionada1 is not None:
        # Convert the selected date to the DataFrame's format
        data_formatada_interna = pd.to_datetime(data_selecionada1).strftime('%Y-%m-%d')
        # Create a datetime combining the selected date with the default time
        data_hora_interna = pd.to_datetime(data_formatada_interna + ' ' + str(hora_selecionada1))
        # Filter the DataFrame based on the selected date and time
        df_filtrado_interno = forecast[forecast['ds'] == data_hora_interna]
        st.dataframe(df_filtrado_interno)
    else:
        st.warning("Não há dados para a data selecionada.")
    return forecast


st.title('📈 Projeção do Índice Bovespa')

# Importando bibliotecas
import streamlit as st

# Estilo para ajustar a largura da área de exibição e justificar o texto
st.markdown(
    """
    <style>
        .reportview-container .main .block-container {
            max-width: 50%;
            justify-content: center;
        }

        .custom-container {
            width: 80%;
            padding: 20px;
            border: 2px solid #333;
            border-radius: 10px;
            margin: 10px 0;
        }
        .custom-container li {
            text-align: justify; /* Alinha o texto à justificação */  
        }
        .custom-container p {
            text-align: justify; /* Alinha o texto à justificação */  
        }
        
    </style>
    """,
    unsafe_allow_html=True
)

# Definindo guias
tabs = ["Visão Geral", "Pontos-chave", "Utilização do Prophet", "Sobre o Autor"]
selected_tab = st.sidebar.radio("Escolha uma guia:", tabs)

# Conteúdo das guias
tab_contents = {
    "Visão Geral": """
    <div class="custom-container">
       <p> Este data app usa a Biblioteca open-source Prophet para automaticamente gerar valores futuros de previsão de um dataset importado. 
        Você poderá visualizar as projeções do índice Bovespa para o período de 01/01/2024 a 31/01/2024 😵.</p>
    </div>
    <style>
        .custom-container p {
            text-align: justify; /* Alinha o texto à justificação */ 
        }
    </style>
    """,
    "Pontos-chave": """
    <div class="custom-container">
      <p>O Prophet tem sido amplamente utilizado em diversas áreas, como previsão de vendas, demanda de produtos, análise financeira, previsão climática e muito mais, devido à sua capacidade de gerar previsões precisas e à sua facilidade de uso. 
        É importante notar que, embora seja uma ferramenta poderosa, a escolha entre modelos depende do contexto específico do problema e da natureza dos dados.</p>
    </div>
    <style>
        .custom-container p {
            text-align: justify; /* Alinha o texto à justificação */
        }
    </style>
    """,
    "Utilização do Prophet": """
    <div class="custom-container">
        <p>A biblioteca Prophet, desenvolvida pelo Facebook, é uma ferramenta popular e poderosa para previsão de séries temporais. Ela foi projetada para simplificar o processo de criação de modelos de previsão, oferecendo aos usuários uma maneira fácil de gerar previsões precisas e de alta qualidade, mesmo sem um profundo conhecimento em séries temporais ou estatística avançada.</p>
        <p>Aqui estão alguns pontos-chave sobre o Prophet:</p>
        <ol>
            <li><strong>Facilidade de Uso:</strong> O Prophet foi desenvolvido para ser acessível e fácil de usar, permitindo que usuários, mesmo sem experiência avançada em séries temporais, possam construir modelos de previsão.</li>
            <li><strong>Componentes Aditivos:</strong> O modelo do Prophet é baseado em componentes aditivos, onde são consideradas tendências anuais, sazonais e efeitos de feriados, além de componentes de regressão.</li>
            <li><strong>Tratamento de Dados Ausentes e Outliers:</strong> O Prophet lida bem com dados ausentes e outliers, reduzindo a necessidade de pré-processamento extensivo dos dados antes da modelagem.</li>
            <li><strong>Flexibilidade:</strong> Permite a inclusão de dados adicionais, como feriados e eventos especiais, para melhorar a precisão das previsões.</li>
            <li><strong>Estimativa Automática de Intervalos de Incerteza:</strong> O Prophet fornece intervalos de incerteza para as previsões, o que é essencial para compreender a confiabilidade dos resultados.</li>
            <li><strong>Implementação em Python e R:</strong> Está disponível tanto para Python quanto para R, ampliando sua acessibilidade para diferentes comunidades de usuários.</li>
            <li><strong>Comunidade Ativa e Documentação Detalhada:</strong> A biblioteca possui uma comunidade ativa de usuários e desenvolvedores, além de uma documentação detalhada e exemplos práticos que ajudam na aprendizagem e na solução de problemas.</li>
        </ol>
    </div>
    """,
    "Sobre o Autor": """
    <div class="custom-container">
        Criado por Henrique José Itzcovici.
        Código disponível em: <a href="https://github.com/Henitz/challenge2" target="_blank">GitHub</a>
    </div>
    """
}

# Renderizar conteúdo da guia selecionada
st.markdown(tab_contents[selected_tab], unsafe_allow_html=True)

"""
### Passo 1: Importar dados
"""
df = pd.DataFrame(columns=['Data'])  # Inicializa um DataFrame vazio

# Adiciona o diretório que contém app.py ao PATH para importações relativas
# diretorio_app = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(diretorio_app)
# from app import pasta_do_zip  # Importa a variável pasta_do_zip de app.py


# URL do arquivo CSV no GitHub
csv_url = "https://raw.githubusercontent.com/Henitz/challenge3/main/Dados Históricos - Ibovespa (4).csv"

# Baixa o arquivo CSV localmente
local_filename = "Dados Históricos - Ibovespa (4).csv"
response = requests.get(csv_url)
with open(local_filename, 'wb') as f:
    f.write(response.content)

# Obtém o nome do arquivo para usar como chave de cache
key_for_cache = str(os.path.basename(local_filename)) if os.path.exists(local_filename) else None

# File Uploader
upload_message = (
    "Importe os dados da série em formato CSV aqui. Posteriormente, as colunas serão nomeadas ds e y. "
    "A entrada de dados para o Prophet sempre deve ser com as colunas: ds e y. "
    "A coluna ds (datestamp) deve ter o formato esperado pelo Pandas, idealmente "
    "YYYY-MM-DD para data ou YYYY-MM-DD HH:MM:SS para timestamp. "
    "A coluna y deve ser numérica e representa a medida que queremos estimar."
)
uploaded_file = st.file_uploader(upload_message, type='csv', key=key_for_cache)

# Use o arquivo CSV local
# if uploaded_file is None:
# Use o arquivo CSV do GitHub se nenhum arquivo foi carregado
#   df = pd.read_csv(local_filename)
#  st.dataframe(df)
# else:
# Use o arquivo carregado
#  df = pd.read_csv(uploaded_file)
# st.dataframe(df)


import streamlit as st

upload_message = (
    "Importe os dados da série em formato CSV aqui. Posteriormente, as colunas serão nomeadas ds e y. "
    "A entrada de dados para o Prophet sempre deve ser com as colunas: ds e y. "
    "A coluna ds (datestamp) deve ter o formato esperado pelo Pandas, idealmente "
    "YYYY-MM-DD para data ou YYYY-MM-DD HH:MM:SS para timestamp. "
    "A coluna y deve ser numérica e representa a medida que queremos estimar."
)

# Use str() to convert the cache key to a string


# File Uploader
# uploaded_file = st.file_uploader(upload_message, type='csv', key=key_for_cache)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if not df.empty and 'Data' in df.columns and 'Último' in df.columns:
        # Renomear colunas para 'ds' e 'y'
        df = df.rename(columns={'Data': 'ds', 'Último': 'y'})
        df['ds'] = pd.to_datetime(df['ds'], format='%d.%m.%Y')

        # Outras operações no DataFrame, como remover colunas indesejadas
        colunas_para_remover = ['Abertura', 'Máxima', 'Mínima', 'Vol.', 'Var%']
        df = df.drop(columns=colunas_para_remover)

        st.dataframe(df.style.set_table_attributes('style="height: 50px; overflow: auto;"'))

        if 'ds' in df.columns:
            data_padrao = df['ds'].min()
            hora_padrao = pd.Timestamp('00:00:00').time()  # Hora padrão como 00:00:00
            data_minima = df['ds'].min()  # Data mínima do DataFrame
            data_maxima = df['ds'].max()  # Data máxima do DataFrame

            data_selecionada = st.sidebar.date_input("Selecione uma data", value=data_padrao, min_value=data_minima,
                                                     max_value=data_maxima)
            hora_selecionada = st.sidebar.time_input("Selecione um horário", value=hora_padrao)

            if data_selecionada:
                # Convertendo a data selecionada para o formato do DataFrame
                data_selecionada_formatada = pd.to_datetime(data_selecionada).strftime('%Y-%m-%d')

                # Criar um datetime combinando a data selecionada com a hora padrão
                data_hora_selecionada = pd.to_datetime(data_selecionada_formatada + ' ' + str(hora_selecionada))

                # Filtrar o DataFrame com base na data e hora selecionadas
                df_filtrado = df[df['ds'] == data_hora_selecionada]
                st.dataframe(df_filtrado)
            else:
                st.warning("Não há dados para a data selecionada.")
        else:
            st.warning("A coluna 'ds' não está presente no DataFrame.")

    else:
        st.warning("O arquivo não foi carregado corretamente ou não possui as colunas esperadas.")

if uploaded_file is not None and not df.empty and 'ds' in df.columns:
    """
    ### Passo 2: Modelo
    """
    modelo(df, data_selecionada, hora_selecionada)
    st.markdown(
        """
        <style>
            .reportview-container .main .block-container {
                max-width: 80%;
                justify-content: center;
            }

            .custom-container {
                border: 2px solid black;
                border-radius: 5px;
                padding: 20px;
                width: 80%;
                text-align: justify;
                margin: 20px 0;
            }

            .nested-container {
                border: 2px solid black;
                border-radius: 5px;
                padding: 10px;
                text-align: justify;
                margin: 10px 0;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    # Definindo guias
    tabs = ["Conceitos", "MAE", "MAPE", "RMSE", "Acurácia"]
    selected_tab = st.sidebar.radio("Escolha uma métrica:", tabs)

    # Conteúdo das guias
    tab_contents = {

        "Conceitos": """
            <div class="nested-container">
            <h6><strong>Conceitos</strong></h6>
            <p><strong>MAE (Mean Absolute Error)</strong>: Representa a média das diferenças absolutas entre as previsões e os valores reais. Indica o quão perto as previsões estão dos valores reais, sem considerar a direção do erro.</p>
            <p><strong>MSE (Mean Squared Error)</strong>: É a média das diferenças quadradas entre as previsões e os valores reais. Penaliza erros maiores mais significativamente que o MAE, devido ao termo quadrático, o que torna o MSE mais sensível a outliers.</p>
            <p><strong>RMSE (Root Mean Squared Error)</strong>: É a raiz quadrada do MSE. Apresenta o mesmo tipo de informação que o MSE, mas na mesma unidade que os dados originais, o que facilita a interpretação.</p>
            <p><strong>MAPE (Mean Absolute Percentage Error)</strong>: É uma métrica usada para avaliar a precisão de um modelo de previsão em relação ao tamanho dos erros em termos percentuais. Essa métrica calcula a média dos valores absolutos dos erros percentuais entre os valores reais e os valores previstos.</p>
            </div>
            """,

        "MAE": """
        <div class="nested-container">
            <h6>MAE</h6>
            <p>O valor do MAE, como qualquer métrica de erro, depende muito do contexto e da escala dos dados que você está considerando. No contexto do mercado de ações, um MAE de 12,06 pontos pode ser considerado alto ou baixo dependendo do valor médio dos índices ou ativos que você está analisando.</p>
            <p>Se o índice ou ativo em questão tem uma faixa de valores geralmente baixa (por exemplo, entre 100 e 200 pontos), um MAE de 12,06 pode ser considerado significativo, representando uma porcentagem considerável dessa faixa.</p>
            <p>Por outro lado, se o índice ou ativo tem valores muito mais altos (por exemplo, entre 1000 e 2000 pontos), um MAE de 12,06 pode ser relativamente pequeno.</p>
            <p>O importante é contextualizar esse valor em relação à escala dos dados que está sendo analisada e considerar como ele se compara a outros modelos ou análises similares. Em geral, um MAE mais baixo indica um melhor desempenho do modelo em prever os valores reais.</p>
        </div>
        """,
        "MAPE": """
        <div class="nested-container">
            <h6><strong>MAPE</strong></h6>
            <p>Para a bolsa de valores, 11,65% é um valor razoável?</p>
            <p>Para a bolsa de valores, um MAPE de 11,65% pode ser considerado relativamente alto em muitos contextos devido à sensibilidade e à volatilidade desse ambiente. No entanto, no mundo da previsão financeira e de ações, avaliar se um MAPE de 11,65% é considerado aceitável ou não depende de diversos fatores:</p>
            <ul>
                <li><strong>Horizonte de Tempo:</strong> O MAPE pode variar dependendo do horizonte de tempo das previsões. Em curtos períodos de tempo, como previsões intra-diárias,
                </li>
                <li><strong>Instrumento Financeiro:</strong> Diferentes tipos de ativos (ações, commodities, moedas) podem ter comportamentos diferentes. Algumas ações podem ser mais voláteis e imprevisíveis do que outras.</li>
                <li><strong>Estratégia de Negociação:</strong> O MAPE aceitável pode variar de acordo com a estratégia de negociação. Para um investidor de longo prazo, um MAPE mais alto pode ser tolerável, enquanto para traders de curto prazo, pode ser considerado menos aceitável.</li>
                <li><strong>Comparação com Referências:</strong> É útil comparar o MAPE obtido com o desempenho de outros modelos de previsão ou com benchmarks do mercado financeiro para avaliar sua eficácia relativa.</li>
                <li><strong>Consequências Financeiras:</strong> Avalie as consequências financeiras do MAPE. Mesmo que 11,65% pareça alto, se as previsões permitirem tomar decisões lucrativas ou reduzir perdas, pode ser aceitável.</
                </li>
            </ul>
            <p>Em geral, para muitos investidores e analistas da bolsa de valores, um MAPE de 11,65% poderia ser considerado relativamente alto, especialmente se a precisão das previsões for crucial para estratégias de negociação específicas. Contudo, é crucial contextualizar o MAPE dentro das especificidades do mercado financeiro e considerar outros indicadores e métricas ao avaliar a eficácia das previsões.</p>
        </div>
        """,
        "RMSE": """
        <div class="nested-container">
            <h6><strong>RMSE</strong></h6>
            <p>Em muitos casos envolvendo previsão na bolsa de valores, um RMSE de 11,65 pode ser considerado alto, especialmente se estiver lidando com a previsão de preços de ações individuais ou ativos específicos. No contexto financeiro, pequenas diferenças nas previsões podem ter um impacto significativo nos resultados e nas decisões de investimento.</p>
            <p>Um RMSE de 11,65 indicaria que, em média, as previsões estão a cerca de 11,65 unidades de distância dos valores reais. Para muitos investidores e analistas financeiros, essa margem de erro pode ser considerada grande, especialmente ao lidar com investimentos de curto prazo ou estratégias de trading onde a precisão é crucial.</p>
            <p>Portanto, para previsões na bolsa de valores, é comum buscar valores de erro menores, indicando uma maior precisão nas previsões. Um RMSE de 11,65 pode ser visto como relativamente alto, sugerindo a necessidade de melhorias no modelo para tornar as previsões mais precisas e confiáveis.</p>
        </div>
        """,
        "Acurácia": """
        <div class="nested-container">
            <h6><strong>Acurácia</strong></h6>
            <p>
                Em modelos de séries temporais, o conceito de "acurácia" não é tão direto quanto em modelos de classificação, onde se pode calcular a precisão de forma direta. A acurácia em modelos de séries temporais pode ser interpretada de maneira diferente, pois envolve a capacidade do modelo de fazer previsões precisas sobre pontos futuros desconhecidos.
            </p>
            <p>
                Em vez de usar termos como "acurácia", normalmente são utilizadas métricas específicas, como as mencionadas anteriormente (MAE, RMSE, MAPE, entre outras), para descrever o quão próximas as previsões do modelo estão dos valores reais.
            </p>
            <p>
                Então, dizer que um modelo de série temporal tem uma precisão de 70% pode não ser a maneira mais comum de descrever seu desempenho. Em vez disso, seria mais informativo dizer algo como "o modelo tem um RMSE de 10", o que indica uma certa magnitude média de erro entre as previsões e os valores reais, ou "o modelo tem um MAPE de 5%", o que mostra a média dos erros percentuais das previsões.
            </p>
            <p>
                Traduzir a performance de um modelo de séries temporais em uma única medida de "acurácia" pode não capturar completamente sua eficácia, já que esses modelos são geralmente avaliados por meio de várias métricas, cada uma fornecendo uma perspectiva diferente do desempenho do modelo.
            </p>
        </div>
        """
    }

    # Exibindo o conteúdo da guia selecionada
    st.markdown(tab_contents[selected_tab], unsafe_allow_html=True)

    """
    ### Passo 3: Previsão no Intervalo 01/01/2024 a 31/01/2024  
    """

    flag = False
    data = st.slider('Data', 1, 31, 1)
    if data <= 9:
        data2 = '2024-01-0' + str(data)
    else:
        data2 = '2024-01-' + str(data)

    btn = st.button("Previsão")

    if btn:
        x = prevendo(df, data2, flag)
        if x is None:
            st.write(f"A data {data2} não está disponível nas previsões ou é feriado/final de semana.")
        else:
            rounded_x = round(x, 3)
            st.write(f"Valor previsto para {data2}: {rounded_x}")
    flag = True
    prevendo(df, data)
