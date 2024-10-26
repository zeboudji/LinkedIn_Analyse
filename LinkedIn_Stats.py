import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np

from io import BytesIO

# Configuration de la page Streamlit
st.set_page_config(page_title="Analyse des Performances LinkedIn", layout="wide")

# Titre de l'application
st.title("📊 Analyse des Performances Réseaux Sociaux - LinkedIn")

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

        # Nettoyer les noms de colonnes pour enlever les espaces
        engagement_df.columns = engagement_df.columns.str.strip()
        abonnés_df.columns = abonnés_df.columns.str.strip()
        meilleurs_posts_df.columns = meilleurs_posts_df.columns.str.strip()
        demographics_df.columns = demographics_df.columns.str.strip()

        # Nettoyer les données des posts
        meilleurs_posts_df.columns = ['Date de publication', 'Interactions']
        meilleurs_posts_df['Date de publication'] = pd.to_datetime(meilleurs_posts_df['Date de publication'], format='%d/%m/%Y', errors='coerce')
        meilleurs_posts_df['Interactions'] = pd.to_numeric(meilleurs_posts_df['Interactions'], errors='coerce')
        posts_per_day = meilleurs_posts_df['Date de publication'].value_counts().sort_index()

        # Nettoyer le dataframe des abonnés et calculer les abonnés cumulés
        abonnés_df_clean = abonnés_df.dropna(subset=['Nouveaux abonnés'])
        # Vérifier le nom exact de la colonne Date dans abonnés_df_clean
        date_column_abonnes = [col for col in abonnés_df_clean.columns if 'Date' in col][0]
        abonnés_df_clean.rename(columns={date_column_abonnes: 'Date'}, inplace=True)
        abonnés_df_clean['Date'] = pd.to_datetime(abonnés_df_clean['Date'], format='%d/%m/%Y', errors='coerce')
        abonnés_df_clean['Nouveaux abonnés'] = pd.to_numeric(abonnés_df_clean['Nouveaux abonnés'], errors='coerce')
        abonnés_df_clean['Cumulative Subscribers'] = abonnés_df_clean['Nouveaux abonnés'].cumsum()

        # Calculer le taux d'engagement
        engagement_df['Interactions'] = pd.to_numeric(engagement_df['Interactions'], errors='coerce')
        engagement_df['Impressions'] = pd.to_numeric(engagement_df['Impressions'], errors='coerce')
        engagement_df['Date'] = pd.to_datetime(engagement_df['Date'], format='%d/%m/%Y', errors='coerce')
        engagement_df['Engagement Rate (%)'] = (engagement_df['Interactions'] / engagement_df['Impressions']) * 100

        # Calculer les chiffres clés
        total_subscribers = abonnés_df_clean['Cumulative Subscribers'].iloc[-1]
        average_engagement_rate = engagement_df['Engagement Rate (%)'].mean()
        total_impressions = engagement_df['Impressions'].sum()
        total_interactions = engagement_df['Interactions'].sum()
        average_subscriber_growth = abonnés_df_clean['Nouveaux abonnés'].mean()

        # Combiner les données pour le traçage
        combined_df = pd.merge(engagement_df, abonnés_df_clean, on='Date', how='left')
        combined_df['Posts per Day'] = combined_df['Date'].map(posts_per_day).fillna(0)

        # Ajouter les colonnes 'Mois' et 'Année' pour les analyses mensuelles
        combined_df['Mois'] = combined_df['Date'].dt.to_period('M').astype(str)
        abonnés_df_clean['Mois'] = abonnés_df_clean['Date'].dt.to_period('M').astype(str)

        # Calculer le taux d'engagement moyen par mois
        monthly_engagement = combined_df.groupby('Mois')['Engagement Rate (%)'].mean().reset_index()
        monthly_subscribers = abonnés_df_clean.groupby('Mois')['Nouveaux abonnés'].sum().reset_index()

        # Graphique : Taux d'engagement moyen par mois
        fig_monthly_engagement = px.line(monthly_engagement, x='Mois', y='Engagement Rate (%)',
                                         title='Taux d\'Engagement Moyen par Mois',
                                         labels={'Engagement Rate (%)': 'Taux d\'Engagement Moyen (%)'},
                                         markers=True, template='plotly_dark')

        # Graphique : Croissance des abonnés par mois
        fig_monthly_subscribers = px.bar(monthly_subscribers, x='Mois', y='Nouveaux abonnés',
                                         title='Croissance des Abonnés par Mois',
                                         labels={'Nouveaux abonnés': 'Nouveaux Abonnés'},
                                         template='plotly_dark')

        # Conversion des dates pour Plotly
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])

        # Graphique 1 : Nombre de posts par jour (Bar Chart)
        fig_posts = px.bar(combined_df, x='Date', y='Posts per Day',
                           title='Nombre de Posts par Jour',
                           labels={'Posts per Day': 'Nombre de posts'},
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
        fig_subscribers = px.line(abonnés_df_clean, x='Date', y='Cumulative Subscribers',
                                  title='Abonnés Cumulés au Fil du Temps',
                                  labels={'Cumulative Subscribers': 'Abonnés Cumulés'},
                                  markers=True,
                                  template='plotly_dark')

        # Graphique 6 : Corrélation entre abonnés cumulés et taux d'engagement (Scatter Plot)
        fig_corr_abonnes_engagement = px.scatter(combined_df, x='Cumulative Subscribers', y='Engagement Rate (%)',
                                                 title="Corrélation entre Abonnés Cumulés et Taux d'Engagement",
                                                 labels={'Cumulative Subscribers': 'Abonnés Cumulés', 'Engagement Rate (%)': 'Taux d\'Engagement (%)'},
                                                 trendline="ols", template='plotly_dark')

        # Graphique 7 : Analyse des pics de croissance des abonnés (Line Chart)
        abonnés_df_clean['Growth Rate'] = abonnés_df_clean['Nouveaux abonnés'].pct_change().fillna(0) * 100  # En pourcentage
        fig_growth_peaks = px.line(abonnés_df_clean, x='Date', y='Growth Rate',
                                   title="Analyse des Pics de Croissance des Abonnés",
                                   labels={'Date': 'Date', 'Growth Rate': 'Taux de Croissance (%)'},
                                   markers=True, template='plotly_dark')

        # Ajout des graphiques démographiques
        # Nettoyer les données démographiques
        demographics_df['Pourcentage'] = demographics_df['Pourcentage'].astype(str)
        demographics_df['Pourcentage'].replace('nan', pd.NA, inplace=True)
        demographics_df['Pourcentage'] = demographics_df['Pourcentage'].str.rstrip('%')
        demographics_df['Pourcentage'] = pd.to_numeric(demographics_df['Pourcentage'], errors='coerce')

        demographics_categories = demographics_df['Principales données démographiques'].unique()

        # Créer un dictionnaire pour stocker les figures démographiques
        demographics_figures = {}

        for category in demographics_categories:
            df_category = demographics_df[demographics_df['Principales données démographiques'] == category]

            # Trier les valeurs par pourcentage décroissant
            df_category = df_category.sort_values(by='Pourcentage', ascending=False)

            # Créer un graphique en barres pour chaque catégorie
            fig = px.bar(df_category, x='Valeur', y='Pourcentage',
                         title=f'Distribution de {category}',
                         labels={'Valeur': category, 'Pourcentage': 'Pourcentage (%)'},
                         template='plotly_dark')
            fig.update_layout(xaxis_tickangle=-45)
            demographics_figures[category] = fig

        # Retourner les chiffres clés et toutes les figures
        return (total_subscribers, average_engagement_rate, total_impressions, total_interactions, average_subscriber_growth,
                fig_posts, fig_impressions, fig_interactions, fig_engagement, fig_subscribers,
                fig_corr_abonnes_engagement, fig_growth_peaks, fig_monthly_engagement, fig_monthly_subscribers, demographics_figures)

    except Exception as e:
        st.error(f"Une erreur est survenue lors de la génération des graphiques : {e}")
        st.exception(e)
        return [None] * 15

# Interface utilisateur
st.sidebar.header("Paramètres")

uploaded_file = st.sidebar.file_uploader("Sélectionnez un fichier Excel", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Appel de la fonction avec gestion des exceptions
    (total_subscribers, average_engagement_rate, total_impressions, total_interactions, average_subscriber_growth,
     fig_posts, fig_impressions, fig_interactions, fig_engagement, fig_subscribers,
     fig_corr_abonnes_engagement, fig_growth_peaks, fig_monthly_engagement, fig_monthly_subscribers, demographics_figures) = generate_performance_graphs(uploaded_file)

    if all([fig_posts, fig_impressions, fig_interactions, fig_engagement, fig_subscribers]):
        # Affichage des chiffres clés
        st.markdown("## 🗝️ Chiffres Clés")

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("Total Abonnés", f"{int(total_subscribers):,}".replace(",", " "))
        col2.metric("Taux d'Engagement Moyen", f"{average_engagement_rate:.2f}%")
        col3.metric("Total Impressions", f"{int(total_impressions):,}".replace(",", " "))
        col4.metric("Total Interactions", f"{int(total_interactions):,}".replace(",", " "))
        col5.metric("Croissance Moyenne des Abonnés", f"{average_subscriber_growth:.2f}")

        st.markdown("---")

        # Organisation des graphiques en sections
        st.markdown("## 📈 Tendances Générales")
        st.plotly_chart(fig_impressions, use_container_width=True)
        st.plotly_chart(fig_interactions, use_container_width=True)
        st.plotly_chart(fig_engagement, use_container_width=True)
        st.plotly_chart(fig_subscribers, use_container_width=True)

        st.markdown("---")

        st.markdown("## 📝 Performance des Posts")
        st.plotly_chart(fig_posts, use_container_width=True)
        st.plotly_chart(fig_monthly_engagement, use_container_width=True)
        st.plotly_chart(fig_monthly_subscribers, use_container_width=True)

        st.markdown("---")

        st.markdown("## 🌍 Données Démographiques")
        for category, fig in demographics_figures.items():
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        st.markdown("## 🔍 Analyses Supplémentaires")
        st.plotly_chart(fig_corr_abonnes_engagement, use_container_width=True)
        st.plotly_chart(fig_growth_peaks, use_container_width=True)

    else:
        st.error("Erreur dans la génération des graphiques.")
else:
    st.info("Veuillez télécharger un fichier Excel pour commencer l'analyse.")
