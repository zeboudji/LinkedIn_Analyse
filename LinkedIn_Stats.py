# LinkedIn_Stats_Streamlit.py

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.dates as mdates
from io import BytesIO

st.set_page_config(page_title="Analyse des Performances LinkedIn", layout="wide")

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

        # Conversion des dates pour matplotlib
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])

        # Création des graphiques
        fig1, axs1 = plt.subplots(3, 1, figsize=(15, 18))
        fig1.suptitle('Performance des Réseaux Sociaux', fontsize=20)

        # Graphique 1 : Nombre de posts par jour
        axs1[0].bar(combined_df['Date'], combined_df['Posts per Day'], color='purple')
        axs1[0].set_title('Nombre de Posts par Jour', fontsize=14)
        axs1[0].set_ylabel('Posts', fontsize=12)
        axs1[0].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
        axs1[0].tick_params(axis='x', rotation=45)

        # Graphique 2 : Impressions au fil du temps
        axs1[1].plot(combined_df['Date'], combined_df['Impressions'], marker='o', color='blue')
        axs1[1].set_title('Impressions au Fil du Temps', fontsize=14)
        axs1[1].set_ylabel('Impressions', fontsize=12)
        axs1[1].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
        axs1[1].tick_params(axis='x', rotation=45)

        # Graphique 3 : Interactions au fil du temps
        axs1[2].plot(combined_df['Date'], combined_df['Interactions'], marker='x', color='orange')
        axs1[2].set_title('Interactions au Fil du Temps', fontsize=14)
        axs1[2].set_ylabel('Interactions', fontsize=12)
        axs1[2].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
        axs1[2].tick_params(axis='x', rotation=45)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Graphiques supplémentaires
        fig2, axs2 = plt.subplots(2, 1, figsize=(15, 12))
        fig2.suptitle('Engagement et Abonnés', fontsize=20)

        # Taux d'engagement
        axs2[0].plot(combined_df['Date'], combined_df['Engagement Rate (%)'], marker='o', color='blue')
        axs2[0].set_title('Taux d\'Engagement au Fil du Temps', fontsize=14)
        axs2[0].set_ylabel('Taux d\'Engagement (%)', fontsize=12)
        axs2[0].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
        axs2[0].tick_params(axis='x', rotation=45)
        axs2[0].grid(True)

        # Abonnés cumulés
        axs2[1].plot(combined_df['Date'], combined_df['Cumulative Subscribers'], marker='o', color='green')
        axs2[1].set_title('Abonnés Cumulés au Fil du Temps', fontsize=14)
        axs2[1].set_xlabel('Date', fontsize=12)
        axs2[1].set_ylabel('Abonnés Cumulés', fontsize=12)
        axs2[1].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
        axs2[1].tick_params(axis='x', rotation=45)
        axs2[1].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        return fig1, fig2

    # Interface utilisateur
    st.sidebar.header("Paramètres")

    uploaded_file = st.sidebar.file_uploader("Sélectionnez un fichier Excel", type=["xlsx", "xls"])

    if uploaded_file is not None:
        try:
            fig1, fig2 = generate_performance_graphs(uploaded_file)

            # Affichage des graphiques
            st.pyplot(fig1)
            st.pyplot(fig2)

        except Exception as e:
            st.error(f"Une erreur est survenue lors du traitement du fichier : {e}")
    else:
        st.info("Veuillez télécharger un fichier Excel pour commencer l'analyse.")
