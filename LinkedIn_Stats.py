import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
import re

from io import BytesIO

# Configuration de la page Streamlit
st.set_page_config(page_title="Analyse des Performances LinkedIn", layout="wide")

# Titre de l'application
st.title("📊 Analyse des Performances Réseaux Sociaux - LinkedIn")

# Fonction pour agréger les données selon la granularité choisie
def aggregate_data(df, date_col, agg_dict, granularity):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    if granularity == "Jour":
        df['Période'] = df[date_col].dt.date
    elif granularity == "Semaine":
        df['Période'] = df[date_col].dt.to_period('W').apply(lambda r: r.start_time)
    elif granularity == "Mois":
        df['Période'] = df[date_col].dt.to_period('M').apply(lambda r: r.start_time)
    aggregated_df = df.groupby('Période').agg(agg_dict).reset_index()
    return aggregated_df

# Fonction pour générer les graphiques de performance
def generate_performance_graphs(excel_data, time_granularity):
    try:
        # Charger le fichier Excel
        xls = pd.ExcelFile(excel_data)

        # Lire la feuille 'ABONNÉS' complète pour extraire le nombre total d'abonnés
        abonnés_df_full = pd.read_excel(xls, 'ABONNÉS')

        # Extraire le texte de la cellule A1
        total_subscribers_text = abonnés_df_full.iloc[0, 0]

        # Afficher le contenu pour le débogage
        st.write("Contenu de la cellule A1 :", total_subscribers_text)

        # Utiliser une expression régulière pour extraire le nombre total d'abonnés
        match = re.search(r'([\d\s]+)$', str(total_subscribers_text))
        if match:
            total_subscribers_str = match.group(1).replace(" ", "")
            total_subscribers = int(total_subscribers_str)
        else:
            total_subscribers = None  # Gérer l'erreur si nécessaire

        # Lire les données à partir de la troisième ligne
        abonnés_df = pd.read_excel(xls, 'ABONNÉS', skiprows=2)
        engagement_df = pd.read_excel(xls, 'ENGAGEMENT')
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

        # Calculer les chiffres clés (le total_subscribers est déjà extrait)
        average_engagement_rate = engagement_df['Engagement Rate (%)'].mean()
        total_impressions = engagement_df['Impressions'].sum()
        total_interactions = engagement_df['Interactions'].sum()
        average_subscriber_growth = abonnés_df_clean['Nouveaux abonnés'].mean()

        # Agréger les données selon la granularité choisie
        agg_dict_engagement = {
            'Impressions': 'sum',
            'Interactions': 'sum',
            'Engagement Rate (%)': 'mean'
        }
        engagement_agg = aggregate_data(engagement_df, 'Date', agg_dict_engagement, time_granularity)

        agg_dict_abonnes = {
            'Nouveaux abonnés': 'sum',
            'Cumulative Subscribers': 'last'
        }
        abonnés_agg = aggregate_data(abonnés_df_clean, 'Date', agg_dict_abonnes, time_granularity)

        # Combiner les données pour le traçage
        combined_df = pd.merge(engagement_agg, abonnés_agg, on='Période', how='left')

        # Posts par période
        posts_df = meilleurs_posts_df.copy()
        posts_df['Date'] = posts_df['Date de publication']
        posts_agg = aggregate_data(posts_df, 'Date', {'Interactions': 'count'}, time_granularity)
        posts_agg.rename(columns={'Interactions': 'Posts per Period'}, inplace=True)

        combined_df = pd.merge(combined_df, posts_agg, on='Période', how='left')

        # Graphiques
        fig_impressions = px.line(combined_df, x='Période', y='Impressions',
                                  title=f'Impressions au Fil du Temps ({time_granularity})',
                                  labels={'Impressions': 'Impressions'},
                                  markers=True,
                                  template='plotly_dark')

        fig_interactions = px.line(combined_df, x='Période', y='Interactions',
                                   title=f'Interactions au Fil du Temps ({time_granularity})',
                                   labels={'Interactions': 'Interactions'},
                                   markers=True,
                                   template='plotly_dark')

        fig_engagement = px.line(combined_df, x='Période', y='Engagement Rate (%)',
                                 title=f'Taux d\'Engagement au Fil du Temps ({time_granularity})',
                                 labels={'Engagement Rate (%)': 'Taux d\'Engagement (%)'},
                                 markers=True,
                                 template='plotly_dark')

        fig_subscribers = px.line(combined_df, x='Période', y='Cumulative Subscribers',
                                  title=f'Abonnés Cumulés au Fil du Temps ({time_granularity})',
                                  labels={'Cumulative Subscribers': 'Abonnés Cumulés'},
                                  markers=True,
                                  template='plotly_dark')

        fig_posts = px.bar(combined_df, x='Période', y='Posts per Period',
                           title=f'Nombre de Posts par {time_granularity}',
                           labels={'Posts per Period': 'Nombre de posts'},
                           template='plotly_dark')

        # Retourner les chiffres clés et les figures
        return (total_subscribers, average_engagement_rate, total_impressions, total_interactions, average_subscriber_growth,
                fig_posts, fig_impressions, fig_interactions, fig_engagement, fig_subscribers)

    except Exception as e:
        st.error(f"Une erreur est survenue lors de la génération des graphiques : {e}")
        st.exception(e)
        return [None] * 10

# Interface utilisateur
st.sidebar.header("Paramètres")

uploaded_file = st.sidebar.file_uploader("Sélectionnez un fichier Excel", type=["xlsx", "xls"])

time_granularity = st.sidebar.selectbox(
    "Sélectionnez la granularité temporelle",
    options=["Jour", "Semaine", "Mois"],
    index=0  # Par défaut, "Jour"
)

if uploaded_file is not None:
    # Appel de la fonction avec gestion des exceptions
    (total_subscribers, average_engagement_rate, total_impressions, total_interactions, average_subscriber_growth,
     fig_posts, fig_impressions, fig_interactions, fig_engagement, fig_subscribers) = generate_performance_graphs(uploaded_file, time_granularity)

    if all([fig_posts, fig_impressions, fig_interactions, fig_engagement, fig_subscribers]):
        # Affichage des chiffres clés
        st.markdown("## 🗝️ Chiffres Clés")

        col1, col2, col3, col4, col5 = st.columns(5)

        if total_subscribers is not None:
            col1.metric("Total Abonnés", f"{total_subscribers:,}".replace(",", " "))
        else:
            col1.metric("Total Abonnés", "Données non disponibles")

        col2.metric("Taux d'Engagement Moyen", f"{average_engagement_rate:.2f}%")
        col3.metric("Total Impressions", f"{int(total_impressions):,}".replace(",", " "))
        col4.metric("Total Interactions", f"{int(total_interactions):,}".replace(",", " "))
        col5.metric("Croissance Moyenne des Abonnés", f"{average_subscriber_growth:.2f}")

        # Organisation des graphiques dans des onglets
        tab1, tab2, tab3 = st.tabs(["Tendances Générales", "Performance des Posts", "Données Démographiques"])

        with tab1:
            st.plotly_chart(fig_impressions, use_container_width=True)
            st.plotly_chart(fig_interactions, use_container_width=True)
            st.plotly_chart(fig_engagement, use_container_width=True)
            st.plotly_chart(fig_subscribers, use_container_width=True)

        with tab2:
            st.plotly_chart(fig_posts, use_container_width=True)
            # Ajouter d'autres graphiques liés aux posts si nécessaire

        with tab3:
            # Afficher les graphiques démographiques si disponibles
            st.write("Graphiques démographiques non inclus dans cet exemple.")
            # for category, fig in demographics_figures.items():
            #     st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Erreur dans la génération des graphiques.")
else:
    st.info("Veuillez télécharger un fichier Excel pour commencer l'analyse.")
