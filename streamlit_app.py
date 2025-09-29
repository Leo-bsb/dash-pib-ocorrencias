# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(layout="wide")
st.title("Análise de Dados de Criminalidade - RIDE/DF")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('_SELECT_CASE_WHEN_o_codigo_municipio_dv_5300108_THEN_5300108_ELS_202509282143.csv')
        return df
    except FileNotFoundError:
        st.error("Arquivo CSV não encontrado. Por favor, faça o upload do arquivo.")
        return None

df = load_data()
if df is None:
    st.stop()

# Abas principais
tab_tabelas, tab_graficos, tab_modelos = st.tabs(["📋 Tabelas", "📊 Gráficos", "🤖 Modelagem Preditiva"])

# =============================
# Aba de Tabelas
# =============================
with tab_tabelas:
    st.header("Visualização dos Dados")
    st.dataframe(df)

    st.header("Estatísticas Descritivas")
    st.write(df.describe())

    st.markdown("""
    **Contexto:**  
    - A base `Ocorrência` registra crimes nas regiões da RIDE/DF.  
    - A base `PIB_municipio` fornece indicadores econômicos municipais.  
    - A base `Censo_2022` foi usada para obter o total da população, permitindo calcular taxas normalizadas de criminalidade.  
    - O objetivo é analisar a relação entre desempenho econômico e criminalidade.
    """)

# =============================
# Aba de Gráficos
# =============================
with tab_graficos:
    # Cálculo de taxas por 100 mil habitantes
    cols_crimes = ['vitimas_feminicidio', 'vitimas_homicidio_doloso',
                   'vitimas_tentativa_homicidio', 'vitimas_totais',
                   'vitimas_lesao_corporal_seguida_de_morte',
                   'vitimas_transito_ou_decorrencia_dele', 'vitimas_sem_indicio_de_crime',
                   'vitimas_latrocinio', 'vitimas_suicidios']

    for col in cols_crimes:
        df[f'{col}_por100mil'] = df[col] / df['total_habitantes'] * 100000

    # -----------------------------
    # Coluna superior: heatmaps de correlação
    # -----------------------------
    st.subheader("📈 Correlação das Variáveis Numéricas")
    cols_selecionadas = [
        'vl_agropecuaria', 'vl_industria', 'vl_servicos', 'vl_administracao',
        'vl_bruto_total', 'vl_subsidios', 'vl_pib', 'vl_pib_per_capta',
        'total_habitantes'
    ] + cols_crimes

    numeric_df = df[cols_selecionadas]
    corr = numeric_df.corr().round(2)

    fig_corr = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        colorscale='Greens',
        zmin=-1, zmax=1,
        showscale=True
    )
    fig_corr.update_layout(title_text="Correlação das Variáveis Numéricas", width=1200, height=800)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("""
    **Explicação:**  
    - Este heatmap mostra a correlação entre variáveis econômicas e de criminalidade.  
    - Observa-se que o PIB total apresenta certa correlação com a criminalidade absoluta, provavelmente porque municípios maiores têm mais habitantes.  
    - Já o PIB per capita tem baixa correlação, sugerindo que o desempenho econômico individual médio não influencia diretamente a criminalidade.
    """)

    # -----------------------------
    # Correlação PIB per capita × taxas de crime
    # -----------------------------
    st.subheader("Correlação PIB per capita × Taxas de Crime")
    colunas_taxas = [f'{c}_por100mil' for c in cols_crimes] + ['vl_pib_per_capta']
    corr_taxas = df[colunas_taxas].corr()

    fig_taxas = go.Figure(data=go.Heatmap(
        z=corr_taxas.values,
        x=corr_taxas.columns,
        y=corr_taxas.columns,
        colorscale='Blues',
        zmin=-1,
        zmax=1,
        showscale=True,
        text=corr_taxas.values.round(2),
        texttemplate="%{text}"
    ))
    fig_taxas.update_layout(title='Correlação PIB per capita × taxas de crime', width=900, height=700)
    st.plotly_chart(fig_taxas, use_container_width=True)

    # -----------------------------
    # Linha inferior: duas colunas para os outros dois gráficos
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Total de Crimes por UF")
        df_uf = df.groupby('uf', as_index=False)['vitimas_totais'].sum().sort_values('vitimas_totais', ascending=False)
        fig1 = px.bar(df_uf, x='uf', y='vitimas_totais', title='Total de Crimes por UF')
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("**Observação:** A criminalidade absoluta é maior em municípios mais populosos.")

    with col2:
        st.subheader("Taxas de Vítimas por 100 mil habitantes")
        colunas_taxas_vitimas = [f'{c}_por100mil' for c in cols_crimes]
        df_media = df[colunas_taxas_vitimas].mean().sort_values()
        fig_vitimas = px.bar(df_media, x=df_media.values, y=df_media.index, orientation='h',
                             title='Taxas de Vítimas por 100 mil habitantes')
        fig_vitimas.update_layout(xaxis_title='Taxa por 100 mil habitantes', yaxis_title='Tipo de Crime')
        st.plotly_chart(fig_vitimas, use_container_width=True)

        st.markdown("**Interpretação:** Ao padronizar por população, podemos comparar municípios independentemente do tamanho.")

# =============================
# Aba de Modelagem
# =============================
with tab_modelos:
    st.header("Modelagem Preditiva")
    st.markdown("""
    **Objetivo:** Verificar se o total de habitantes e desempenho econômico podem prever a criminalidade total.  
    Foram testados dois modelos de regressão: Linear e Random Forest.
    """)
    
    X = df[['total_habitantes']]
    y = df['vitimas_totais']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.44, random_state=42)

    # Regressão Linear
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=41)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    st.subheader("Resultados dos Modelos")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Regressão Linear**")
        st.write(f"R²: {r2_score(y_test, y_pred_lr):.3f}")
        st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.3f}")
        st.markdown("Interpretação: O modelo linear explica bem a variação da criminalidade, indicando que a população é o principal fator.")
    with col2:
        st.write("**Random Forest**")
        st.write(f"R²: {r2_score(y_test, y_pred_rf):.3f}")
        st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.3f}")
        st.markdown("Interpretação: Modelo não conseguiu capturar melhor a relação, reforçando que a relação é linear simples com população.")

    fig_modelos = go.Figure()
    fig_modelos.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                     y=[y_test.min(), y_test.max()],
                                     mode='lines', line=dict(dash='dash'), name='Perfeita'))
    fig_modelos.add_trace(go.Scatter(x=y_test, y=y_pred_lr, mode='markers',
                                     name='Regressão Linear', opacity=0.5))
    fig_modelos.add_trace(go.Scatter(x=y_test, y=y_pred_rf, mode='markers',
                                     name='Random Forest', opacity=0.5))
    fig_modelos.update_layout(title="Comparação de Modelos",
                              xaxis_title="Valores Reais",
                              yaxis_title="Previsões")
    st.plotly_chart(fig_modelos, use_container_width=True)
