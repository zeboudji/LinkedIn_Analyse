# LinkedIn_Stats_Streamlit.py

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from io import BytesIO

# Configuration de la page Streamlit
st.set_page_config(page_title="Analyse des Performances LinkedIn", layout="wide")

# Titre de l'application
st.title("Analyse des Performances Réseaux Sociaux - LinkedIn")

# Fonction pour générer les graphiques de performance
def generate_performance_graphs(excel_data):
    try:
        # Charger le fichier Excel
        xls = pd.ExcelFile(excel_data)

        # Charger chaque feuille pertinente dans des dataframes
        engagement_df = pd.read_excel(xls, 'ENGAGEMENT')
        abonnés_df = pd.read_excel(xls, 'ABONNÉS', skiprows=2)
        meilleurs_posts_df = pd.read_excel(xls, 'MEILLEURS POSTS').iloc[2:, 1:3]

        # Nettoyer les données des posts
        meilleurs_posts_df.columns = ['Date de publication', 'Interactions']
        meilleurs_posts_df['Date de publication'] = pd.to_datetime(meilleurs_posts_df['Date de publication'], format='%d/%m/%Y')
        posts_per_day = meilleurs_posts_df['Date de publication'].value_counts().sort_index()

        # Nettoyer le dataframe des abonnés et calculer les abonnés cumulés
        abonnés_df_clean = abonnés_df.dropna()
        abonnés_df_clean['Cumulative Subscribers'] = abonnés_df_clean['Nouveaux abonnés'].cumsum()

        # Calculer le taux d'engagement
        engagement_df['Engagement Rate (%)'] = (engagement_df['Interactions'] / engagement_df['Impressions']) * 100

        # Combiner les données pour le traçage
        combined_df = pd.merge(engagement_df, abonnés_df_clean, left_on='Date', right_on='Date ', how='left')
        combined_df['Posts per Day'] = combined_df['Date'].map(posts_per_day).fillna(0)

        # Conversion des dates pour Plotly
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])

        # Graphique 1 : Nombre de posts par jour (Bar Chart)
        fig_posts = px.bar(combined_df, x='Date', y='Posts per Day',
                           title='Nombre de Posts par Jour',
                           labels={'Posts per Day': 'Posts'},
                           template='plotly_dark')

        # Graphique 2 : Impressions au fil du temps (Line Chart)
        fig_impressions = px.line(combined_df, x='Date', y='Impressions',
                                  title='Impressions au Fil du Temps',
                                  labels={'Impressions': 'Impressions'},
                                  markers=True,
                                  template='plotly_dark')

        # Graphique 3 : Interactions au fil du temps (Line Chart)
        fig_interactions = px.line(combined_df, x='Date', y='Interactions',
                                   title='Interactions au Fil du Temps',
                                   labels={'Interactions': 'Interactions'},
                                   markers=True,
                                   template='plotly_dark')

        # Graphique 4 : Taux d'engagement au fil du temps (Line Chart)
        fig_engagement = px.line(combined_df, x='Date', y='Engagement Rate (%)',
                                 title='Taux d\'Engagement au Fil du Temps',
                                 labels={'Engagement Rate (%)': 'Taux d\'Engagement (%)'},
                                 markers=True,
                                 template='plotly_dark')

        # Graphique 5 : Abonnés cumulés au fil du temps (Line Chart)
        fig_subscribers = px.line(combined_df, x='Date', y='Cumulative Subscribers',
                                  title='Abonnés Cumulés au Fil du Temps',
                                  labels={'Cumulative Subscribers': 'Abonnés Cumulés'},
                                  markers=True,
                                  template='plotly_dark')

        return fig_posts, fig_impressions, fig_interactions, fig_engagement, fig_subscribers

    except Exception as e:
        st.error(f"Une erreur est survenue lors de la génération des graphiques : {e}")
        return None, None, None, None, None

# Interface utilisateur
st.sidebar.header("Paramètres")

uploaded_file = st.sidebar.file_uploader("Sélectionnez un fichier Excel", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Appel de la fonction avec gestion des exceptions
    fig_posts, fig_impressions, fig_interactions, fig_engagement, fig_subscribers = generate_performance_graphs(uploaded_file)

    if fig_posts and fig_impressions and fig_interactions and fig_engagement and fig_subscribers:
        # Organisation des graphiques dans des onglets
        tab1, tab2 = st.tabs(["Performance des Posts", "Engagement et Abonnés"])

        with tab1:
            st.plotly_chart(fig_posts, use_container_width=True)
            st.plotly_chart(fig_impressions, use_container_width=True)
            st.plotly_chart(fig_interactions, use_container_width=True)

        with tab2:
            st.plotly_chart(fig_engagement, use_container_width=True)
            st.plotly_chart(fig_subscribers, use_container_width=True)
else:
    st.info("Veuillez télécharger un fichier Excel pour commencer l'analyse.")
