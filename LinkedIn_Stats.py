# LinkedIn_Stats_Streamlit.py

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from io import BytesIO

# Configuration de la page Streamlit
st.set_page_config(page_title="Analyse des Performances LinkedIn", layout="wide")

# Titre de l'application
st.title("Analyse des Performances Réseaux Sociaux - LinkedIn")

# Fonctions de génération des graphiques supplémentaires
def plot_top_10_posts(meilleurs_posts_df):
    # Trier les posts par interactions décroissantes et sélectionner les 10 meilleurs
    top_posts = meilleurs_posts_df.sort_values(by='Interactions', ascending=False).head(10)

    fig_top_posts = px.bar(top_posts, x='Date de publication', y='Interactions',
                           title='Top 10 Meilleurs Posts',
                           labels={'Interactions': 'Nombre d\'Interactions', 'Date de publication': 'Date de Publication'},
                           template='plotly_dark',
                           text='Interactions')
    fig_top_posts.update_traces(texttemplate='%{text}', textposition='outside')
    fig_top_posts.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    return fig_top_posts

def plot_daily_subscribers(abonnés_df_clean):
    fig_daily_subscribers = px.line(abonnés_df_clean, x='Date', y='Nouveaux abonnés',
                                    title='Croissance Quotidienne des Abonnés',
                                    labels={'Nouveaux abonnés': 'Nouveaux Abonnés', 'Date': 'Date'},
                                    markers=True,
                                    template='plotly_dark')
    return fig_daily_subscribers

def plot_correlation_impressions_interactions(engagement_df):
    fig_corr = px.scatter(engagement_df, x='Impressions', y='Interactions',
                          title='Corrélation entre Impressions et Interactions',
                          labels={'Impressions': 'Impressions', 'Interactions': 'Interactions'},
                          trendline='ols',
                          template='plotly_dark')
    return fig_corr

def plot_monthly_growth(abonnés_df_clean):
    # Créer une colonne 'Month' pour l'agrégation
    abonnés_df_clean['Month'] = abonnés_df_clean['Date'].dt.to_period('M')
    monthly_growth = abonnés_df_clean.groupby('Month')['Nouveaux abonnés'].sum().reset_index()
    monthly_growth['Month'] = monthly_growth['Month'].dt.to_timestamp()

    fig_monthly_growth = px.bar(monthly_growth, x='Month', y='Nouveaux abonnés',
                                 title='Taux de Croissance Mensuel des Abonnés',
                                 labels={'Nouveaux abonnés': 'Nouveaux Abonnés', 'Month': 'Mois'},
                                 template='plotly_dark',
                                 text='Nouveaux abonnés')
    fig_monthly_growth.update_traces(texttemplate='%{text}', textposition='outside')
    fig_monthly_growth.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    return fig_monthly_growth

def plot_moving_average_engagement(engagement_df):
    # Calculer la moyenne mobile sur 7 jours
    engagement_df['Engagement Rate MA'] = engagement_df['Engagement Rate (%)'].rolling(window=7).mean()

    fig_engagement_ma = px.line(engagement_df, x='Date', y='Engagement Rate MA',
                                title='Moyenne Mobile du Taux d\'Engagement (7 jours)',
                                labels={'Engagement Rate MA': 'Taux d\'Engagement (%)', 'Date': 'Date'},
                                markers=True,
                                template='plotly_dark')
    return fig_engagement_ma

def plot_distribution_impressions_interactions(engagement_df):
    fig_distribution = make_subplots(rows=1, cols=2, subplot_titles=("Distribution des Impressions", "Distribution des Interactions"))

    fig_distribution.add_trace(
        go.Histogram(x=engagement_df['Impressions'], nbinsx=20, marker_color='blue', name='Impressions'),
        row=1, col=1
    )

    fig_distribution.add_trace(
        go.Histogram(x=engagement_df['Interactions'], nbinsx=20, marker_color='orange', name='Interactions'),
        row=1, col=2
    )

    fig_distribution.update_layout(title_text='Distribution des Impressions et Interactions', template='plotly_dark')
    return fig_distribution

def plot_weekly_activity(meilleurs_posts_df):
    # Extraire le jour de la semaine
    meilleurs_posts_df['Jour'] = meilleurs_posts_df['Date de publication'].dt.day_name()

    # Compter le nombre de posts et interactions par jour
    weekly_activity = meilleurs_posts_df.groupby('Jour').agg({'Interactions': 'sum', 'Date de publication': 'count'}).reset_index()
    weekly_activity.rename(columns={'Date de publication': 'Nombre de Posts'}, inplace=True)

    # Ordre des jours de la semaine
    jours_semaine = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_activity['Jour'] = pd.Categorical(weekly_activity['Jour'], categories=jours_semaine, ordered=True)
    weekly_activity = weekly_activity.sort_values('Jour')

    # Création du heatmap
    fig_weekly_activity = make_subplots(rows=1, cols=2, subplot_titles=("Interactions par Jour", "Nombre de Posts par Jour"))

    fig_weekly_activity.add_trace(
        go.Heatmap(
            z=weekly_activity['Interactions'],
            x=['Interactions'],
            y=weekly_activity['Jour'],
            colorscale='Viridis',
            colorbar=dict(title='Interactions')
        ),
        row=1, col=1
    )

    fig_weekly_activity.add_trace(
        go.Heatmap(
            z=weekly_activity['Nombre de Posts'],
            x=['Nombre de Posts'],
            y=weekly_activity['Jour'],
            colorscale='Blues',
            colorbar=dict(title='Nombre de Posts')
        ),
        row=1, col=2
    )

    fig_weekly_activity.update_layout(
        title_text='Analyse Hebdomadaire des Publications',
        template='plotly_dark'
    )

    return fig_weekly_activity

# Fonction pour générer les graphiques de performance
def generate_performance_graphs(excel_data):
    try:
        # Charger le fichier Excel
        xls = pd.ExcelFile(excel_data)

        # Liste des feuilles disponibles
        sheets = xls.sheet_names
        st.write("Feuilles disponibles dans le fichier Excel :", sheets)

        # Vérifier la présence des feuilles requises
        required_sheets = ['ENGAGEMENT', 'ABONNÉS', 'MEILLEURS POSTS']
        missing_sheets = [sheet for sheet in required_sheets if sheet not in sheets]

        if missing_sheets:
            st.error(f"Feuilles manquantes dans le fichier Excel : {', '.join(missing_sheets)}")
            return (None,) * 12

        # Charger chaque feuille pertinente dans des dataframes
        engagement_df = pd.read_excel(xls, 'ENGAGEMENT')
        abonnés_df = pd.read_excel(xls, 'ABONNÉS', skiprows=2)
        meilleurs_posts_df = pd.read_excel(xls, 'MEILLEURS POSTS').iloc[2:, 1:3]

        # Nettoyer les noms des colonnes en supprimant les espaces
        engagement_df.columns = engagement_df.columns.str.strip()
        abonnés_df.columns = abonnés_df.columns.str.strip()
        meilleurs_posts_df.columns = meilleurs_posts_df.columns.str.strip()

        # Vérifier la présence des colonnes requises
        required_columns_engagement = ['Date', 'Interactions', 'Impressions']
        required_columns_abonnes = ['Date', 'Nouveaux abonnés']
        required_columns_meilleurs_posts = ['Date de publication', 'Interactions']

        missing_columns = []

        for col in required_columns_engagement:
            if col not in engagement_df.columns:
                missing_columns.append(f"'ENGAGEMENT' - {col}")

        for col in required_columns_abonnes:
            if col not in abonnés_df.columns:
                missing_columns.append(f"'ABONNÉS' - {col}")

        for col in required_columns_meilleurs_posts:
            if col not in meilleurs_posts_df.columns:
                missing_columns.append(f"'MEILLEURS POSTS' - {col}")

        if missing_columns:
            st.error(f"Colonnes manquantes dans le fichier Excel : {', '.join(missing_columns)}")
            return (None,) * 12

        # Convertir les colonnes de dates en datetime avec gestion des erreurs
        engagement_df['Date'] = pd.to_datetime(engagement_df['Date'], format='%d/%m/%Y', errors='coerce')
        abonnés_df['Date'] = pd.to_datetime(abonnés_df['Date'], format='%d/%m/%Y', errors='coerce')
        meilleurs_posts_df['Date de publication'] = pd.to_datetime(meilleurs_posts_df['Date de publication'], format='%d/%m/%Y', errors='coerce')

        # Vérifier les conversions
        if engagement_df['Date'].isnull().any():
            st.warning("Certaines dates dans la feuille 'ENGAGEMENT' n'ont pas pu être converties et seront ignorées.")
        if abonnés_df['Date'].isnull().any():
            st.warning("Certaines dates dans la feuille 'ABONNÉS' n'ont pas pu être converties et seront ignorées.")
        if meilleurs_posts_df['Date de publication'].isnull().any():
            st.warning("Certaines dates dans la feuille 'MEILLEURS POSTS' n'ont pas pu être converties et seront ignorées.")

        # Supprimer les lignes avec des dates non valides
        engagement_df = engagement_df.dropna(subset=['Date'])
        abonnés_df_clean = abonnés_df.dropna(subset=['Date'])
        meilleurs_posts_df = meilleurs_posts_df.dropna(subset=['Date de publication'])

        # Calculer les posts par jour
        posts_per_day = meilleurs_posts_df['Date de publication'].value_counts().sort_index()

        # Calculer les abonnés cumulés
        abonnés_df_clean['Cumulative Subscribers'] = abonnés_df_clean['Nouveaux abonnés'].cumsum()

        # Calculer le taux d'engagement
        engagement_df['Engagement Rate (%)'] = (engagement_df['Interactions'] / engagement_df['Impressions']) * 100

        # Combiner les données pour le traçage
        combined_df = pd.merge(engagement_df, abonnés_df_clean, on='Date', how='left')
        combined_df['Posts per Day'] = combined_df['Date'].map(posts_per_day).fillna(0)

        # Assurez-vous que 'Date' est bien datetime
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])

        # Génération des graphiques principaux
        fig_posts = px.bar(combined_df, x='Date', y='Posts per Day',
                           title='Nombre de Posts par Jour',
                           labels={'Posts per Day': 'Posts', 'Date': 'Date'},
                           template='plotly_dark')

        fig_impressions = px.line(combined_df, x='Date', y='Impressions',
                                  title='Impressions au Fil du Temps',
                                  labels={'Impressions': 'Impressions'},
                                  markers=True,
                                  template='plotly_dark')

        fig_interactions = px.line(combined_df, x='Date', y='Interactions',
                                   title='Interactions au Fil du Temps',
                                   labels={'Interactions': 'Interactions'},
                                   markers=True,
                                   template='plotly_dark')

        fig_engagement = px.line(combined_df, x='Date', y='Engagement Rate (%)',
                                 title='Taux d\'Engagement au Fil du Temps',
                                 labels={'Engagement Rate (%)': 'Taux d\'Engagement (%)'},
                                 markers=True,
                                 template='plotly_dark')

        fig_subscribers = px.line(combined_df, x='Date', y='Cumulative Subscribers',
                                  title='Abonnés Cumulés au Fil du Temps',
                                  labels={'Cumulative Subscribers': 'Abonnés Cumulés'},
                                  markers=True,
                                  template='plotly_dark')

        # Génération des graphiques supplémentaires
        fig_top_posts = plot_top_10_posts(meilleurs_posts_df)
        fig_daily_subscribers = plot_daily_subscribers(abonnés_df_clean)
        fig_corr = plot_correlation_impressions_interactions(engagement_df)
        fig_monthly_growth = plot_monthly_growth(abonnés_df_clean)
        fig_engagement_ma = plot_moving_average_engagement(engagement_df)
        fig_distribution = plot_distribution_impressions_interactions(engagement_df)
        fig_weekly_activity = plot_weekly_activity(meilleurs_posts_df)

        return (fig_posts, fig_impressions, fig_interactions, fig_engagement, fig_subscribers,
                fig_top_posts, fig_daily_subscribers, fig_corr, fig_monthly_growth,
                fig_engagement_ma, fig_distribution, fig_weekly_activity)

    except FileNotFoundError:
        st.error("Le fichier Excel sélectionné est introuvable. Veuillez vérifier le chemin et réessayer.")
        return (None,) * 12
    except pd.errors.EmptyDataError:
        st.error("Le fichier Excel est vide. Veuillez sélectionner un fichier contenant des données.")
        return (None,) * 12
    except KeyError as e:
        st.error(f"Feuille manquante dans le fichier Excel : {e}")
        return (None,) * 12
    except Exception as e:
        st.error(f"Une erreur inattendue est survenue : {e}")
        return (None,) * 12

# Interface utilisateur
st.sidebar.header("Paramètres")

uploaded_file = st.sidebar.file_uploader("Sélectionnez un fichier Excel", type=["xlsx", "xls"])

# Création des onglets principaux
tab_main, tab_help = st.tabs(["📈 Analyse des Données", "❓ Aide et Documentation"])

with tab_main:
    if uploaded_file is not None:
        # Appel de la fonction avec gestion des exceptions
        (fig_posts, fig_impressions, fig_interactions, fig_engagement, fig_subscribers,
         fig_top_posts, fig_daily_subscribers, fig_corr, fig_monthly_growth,
         fig_engagement_ma, fig_distribution, fig_weekly_activity) = generate_performance_graphs(uploaded_file)

        if all([fig_posts, fig_impressions, fig_interactions, fig_engagement, fig_subscribers,
                fig_top_posts, fig_daily_subscribers, fig_corr, fig_monthly_growth,
                fig_engagement_ma, fig_distribution, fig_weekly_activity]):
            # Organisation des graphiques dans des sous-onglets
            subtab1, subtab2, subtab3 = st.tabs(["Performance des Posts", "Engagement et Abonnés", "Analyses Supplémentaires"])

            with subtab1:
                st.plotly_chart(fig_posts, use_container_width=True)
                st.plotly_chart(fig_top_posts, use_container_width=True)

            with subtab2:
                st.plotly_chart(fig_impressions, use_container_width=True)
                st.plotly_chart(fig_interactions, use_container_width=True)
                st.plotly_chart(fig_engagement, use_container_width=True)
                st.plotly_chart(fig_subscribers, use_container_width=True)
                st.plotly_chart(fig_daily_subscribers, use_container_width=True)
                st.plotly_chart(fig_monthly_growth, use_container_width=True)
                st.plotly_chart(fig_engagement_ma, use_container_width=True)
            
            with subtab3:
                st.plotly_chart(fig_corr, use_container_width=True)
                st.plotly_chart(fig_distribution, use_container_width=True)
                st.plotly_chart(fig_weekly_activity, use_container_width=True)
    else:
        st.info("Veuillez télécharger un fichier Excel pour commencer l'analyse.")

with tab_help:
    st.header("Comment Utiliser l'Outil d'Analyse des Performances LinkedIn")
    st.markdown("""
    ### **Bienvenue !**

    Ce didacticiel vous guidera à travers les étapes pour utiliser l'outil d'analyse des performances LinkedIn.

    ### **Étape 1 : Préparer votre Fichier Excel**

    - **ENGAGEMENT**
      - Colonnes requises : `Date`, `Interactions`, `Impressions`
    - **ABONNÉS**
      - Colonnes requises : `Date`, `Nouveaux abonnés`
      - **Remarque :** Les deux premières lignes sont ignorées.
    - **MEILLEURS POSTS**
      - Colonnes requises : `Date de publication`, `Interactions`
      - **Remarque :** Les deux premières lignes et la première colonne sont ignorées.

    ### **Étape 2 : Télécharger votre Fichier Excel**

    1. Dans la **barre latérale**, cliquez sur **"Sélectionnez un fichier Excel"**.
    2. Une fenêtre de dialogue s'ouvrira. Sélectionnez votre fichier Excel préparé.

    ### **Étape 3 : Analyser les Graphiques**

    - **Performance des Posts :**
      - **Nombre de Posts par Jour :** Visualisez la fréquence de vos publications.
      - **Top 10 Meilleurs Posts :** Identifiez vos publications les plus performantes.

    - **Engagement et Abonnés :**
      - **Impressions au Fil du Temps :** Suivez la portée de vos posts.
      - **Interactions au Fil du Temps :** Mesurez l'engagement généré par vos publications.
      - **Taux d'Engagement au Fil du Temps :** Évaluez l'efficacité de vos interactions.
      - **Abonnés Cumulés au Fil du Temps :** Observez la croissance de votre audience.
      - **Croissance Quotidienne des Abonnés :** Suivez les nouveaux abonnés chaque jour.
      - **Taux de Croissance Mensuel des Abonnés :** Analysez la croissance de votre audience par mois.
      - **Moyenne Mobile du Taux d'Engagement :** Visualisez les tendances sous-jacentes du taux d'engagement.

    - **Analyses Supplémentaires :**
      - **Corrélation entre Impressions et Interactions :** Comprenez la relation entre la portée et l'engagement.
      - **Distribution des Impressions et Interactions :** Analysez la variabilité de vos données.
      - **Analyse Hebdomadaire des Publications :** Identifiez les jours les plus performants pour publier du contenu.

    ### **Étape 4 : Interagir avec les Graphiques**

    - **Zoomer et Panorer :** Utilisez votre souris pour explorer les détails des graphiques.
    - **Télécharger les Graphiques :** Cliquez sur l'icône de téléchargement sur chaque graphique pour les sauvegarder.
    - **Filtres Dynamiques :** Utilisez les options de filtrage pour analyser des périodes spécifiques ou d'autres critères pertinents.

    ### **Conseils pour une Analyse Optimale**

    - **Filtrer par Date :** Utilisez les options de filtrage pour analyser des périodes spécifiques.
    - **Comparer les Performances :** Comparez différentes périodes pour identifier les tendances.
    - **Exporter les Résultats :** Intégrez les graphiques dans vos rapports pour une présentation professionnelle.

    ### **Besoin d'Aide ?**

    Si vous rencontrez des problèmes ou avez des questions, n'hésitez pas à contacter le support technique ou à consulter la [documentation officielle de Streamlit](https://docs.streamlit.io/).

    ### **Bonnes Analyses !**
    """)
