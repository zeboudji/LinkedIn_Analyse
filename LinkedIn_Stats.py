import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff

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
        demographics_df = pd.read_excel(xls, 'DONNÉES DÉMOGRAPHIQUES')

        # Nettoyer les données des posts
        meilleurs_posts_df.columns = ['Date de publication', 'Interactions']
        meilleurs_posts_df['Date de publication'] = pd.to_datetime(meilleurs_posts_df['Date de publication'], format='%d/%m/%Y')
        meilleurs_posts_df['Interactions'] = pd.to_numeric(meilleurs_posts_df['Interactions'], errors='coerce')
        posts_per_day = meilleurs_posts_df['Date de publication'].value_counts().sort_index()

        # Nettoyer le dataframe des abonnés et calculer les abonnés cumulés
        abonnés_df_clean = abonnés_df.dropna()
        abonnés_df_clean['Date'] = pd.to_datetime(abonnés_df_clean['Date'], format='%d/%m/%Y', errors='coerce')
        abonnés_df_clean['Cumulative Subscribers'] = abonnés_df_clean['Nouveaux abonnés'].cumsum()

        # Calculer le taux d'engagement
        engagement_df['Engagement Rate (%)'] = (engagement_df['Interactions'] / engagement_df['Impressions']) * 100

        # Combiner les données pour le traçage
        combined_df = pd.merge(engagement_df, abonnés_df_clean, on='Date', how='left')
        combined_df['Posts per Day'] = combined_df['Date'].map(posts_per_day).fillna(0)

        # Conversion des dates pour Plotly
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])

        # Graphiques de performance (comme précédemment)
        # ... [Vos graphiques précédents ici] ...

        # Nettoyer les données démographiques
        demographics_df['Pourcentage'] = demographics_df['Pourcentage'].str.rstrip('%').astype(float)
        demographics_categories = demographics_df['Principales données démographiques'].unique()

        # Créer un dictionnaire pour stocker les figures démographiques
        demographics_figures = {}

        for category in demographics_categories:
            df_category = demographics_df[demographics_df['Principales données démographiques'] == category]

            # Créer un graphique en barres pour chaque catégorie
            fig = px.bar(df_category, x='Valeur', y='Pourcentage',
                         title=f'Distribution de {category}',
                         labels={'Valeur': category, 'Pourcentage': 'Pourcentage (%)'},
                         template='plotly_dark')
            fig.update_layout(xaxis_tickangle=-45)
            demographics_figures[category] = fig

        # Retourner toutes les figures
        return (fig_posts, fig_impressions, fig_interactions, fig_engagement, fig_subscribers,
                fig_corr_abonnes_engagement, fig_growth_peaks, fig_interaction_distribution,
                fig_engagement_rolling, fig_corr_matrix, demographics_figures)

    except Exception as e:
        st.error(f"Une erreur est survenue lors de la génération des graphiques : {e}")
        return [None] * 11

# Interface utilisateur
st.sidebar.header("Paramètres")

uploaded_file = st.sidebar.file_uploader("Sélectionnez un fichier Excel", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Appel de la fonction avec gestion des exceptions
    (fig_posts, fig_impressions, fig_interactions, fig_engagement, fig_subscribers,
     fig_corr_abonnes_engagement, fig_growth_peaks, fig_interaction_distribution,
     fig_engagement_rolling, fig_corr_matrix, demographics_figures) = generate_performance_graphs(uploaded_file)

    if all([fig_posts, fig_impressions, fig_interactions, fig_engagement, fig_subscribers]):
        # Organisation des graphiques dans des onglets
        tab1, tab2, tab3, tab4 = st.tabs(["Performance des Posts", "Engagement et Abonnés", "Analyses Supplémentaires", "Données Démographiques"])

        with tab1:
            st.plotly_chart(fig_posts, use_container_width=True)
            st.plotly_chart(fig_impressions, use_container_width=True)
            st.plotly_chart(fig_interactions, use_container_width=True)

        with tab2:
            st.plotly_chart(fig_engagement, use_container_width=True)
            st.plotly_chart(fig_subscribers, use_container_width=True)
            st.plotly_chart(fig_engagement_rolling, use_container_width=True)

        with tab3:
            st.plotly_chart(fig_corr_abonnes_engagement, use_container_width=True)
            st.plotly_chart(fig_growth_peaks, use_container_width=True)
            st.plotly_chart(fig_interaction_distribution, use_container_width=True)
            st.plotly_chart(fig_corr_matrix, use_container_width=True)

        with tab4:
            for category, fig in demographics_figures.items():
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Erreur dans la génération des graphiques.")
else:
    st.info("Veuillez télécharger un fichier Excel pour commencer l'analyse.")
