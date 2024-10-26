import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from io import BytesIO

# Configuration de la page Streamlit
st.set_page_config(page_title="Analyse des Performances LinkedIn", layout="wide")

# Fonction pour convertir un DataFrame en CSV pour téléchargement
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# Fonction pour renommer les colonnes de manière dynamique
def rename_columns(df, mapping):
    df = df.copy()
    df.columns = [mapping.get(col.strip(), col.strip()) for col in df.columns]
    return df

# Fonction principale pour générer les graphiques de performance
def generate_performance_graphs(dataframes):
    try:
        # Extraction des DataFrames
        engagement_df = dataframes['ENGAGEMENT'].copy()
        abonnes_df = dataframes['ABONNES'].copy()
        meilleurs_posts_df = dataframes['MEILLEURS POSTS'].copy()
        demographics_df = dataframes['DONNÉES DEMOGRAPHIQUES'].copy()

        # Nettoyer les noms de colonnes pour enlever les espaces
        engagement_df.columns = engagement_df.columns.str.strip()
        abonnes_df.columns = abonnes_df.columns.str.strip()
        meilleurs_posts_df.columns = meilleurs_posts_df.columns.str.strip()
        demographics_df.columns = demographics_df.columns.str.strip()

        # Définir les mappings de renommage des colonnes
        mapping_engagement = {
            'Date Originale': 'Date',  # Ajustez selon vos données réelles
            'Interactions Original': 'Interactions',
            'Impressions Original': 'Impressions'
        }
        mapping_abonnes = {
            'Date Originale': 'Date',
            'Nouveaux abonnés Original': 'Nouveaux abonnés'
        }
        mapping_meilleurs_posts = {
            'Date de publication Originale': 'Date de publication',
            'Interactions Originales': 'Interactions'
        }
        mapping_demographics = {
            'Principales données démographiques Originale': 'Principales données démographiques',
            'Valeur Originale': 'Valeur',
            'Pourcentage Original': 'Pourcentage'
        }

        # Renommer les colonnes de manière dynamique
        engagement_df = rename_columns(engagement_df, mapping_engagement)
        abonnes_df = rename_columns(abonnes_df, mapping_abonnes)
        meilleurs_posts_df = rename_columns(meilleurs_posts_df, mapping_meilleurs_posts)
        demographics_df = rename_columns(demographics_df, mapping_demographics)

        # Vérification des colonnes essentielles après renommage
        essential_columns = {
            'ENGAGEMENT': ['Date', 'Interactions', 'Impressions'],
            'ABONNES': ['Date', 'Nouveaux abonnés'],
            'MEILLEURS POSTS': ['Date de publication', 'Interactions'],
            'DONNÉES DEMOGRAPHIQUES': ['Principales données démographiques', 'Valeur', 'Pourcentage']
        }

        for sheet, cols in essential_columns.items():
            for col in cols:
                if col not in dataframes[sheet].columns:
                    st.warning(f"La colonne '{col}' est manquante dans la feuille '{sheet}'. Certaines analyses pourraient être affectées.")

        # Nettoyer les données des posts
        meilleurs_posts_df['Date de publication'] = pd.to_datetime(meilleurs_posts_df['Date de publication'], format='%d/%m/%Y', errors='coerce')
        meilleurs_posts_df['Interactions'] = pd.to_numeric(meilleurs_posts_df['Interactions'], errors='coerce')

        # Gérer les valeurs NaN après conversion
        meilleurs_posts_df = meilleurs_posts_df.dropna(subset=['Date de publication', 'Interactions'])

        if meilleurs_posts_df.empty:
            st.warning("Le DataFrame 'MEILLEURS POSTS' est vide après nettoyage. Aucune donnée à afficher pour les meilleurs posts.")

        posts_per_day = meilleurs_posts_df['Date de publication'].value_counts().sort_index()

        # Nettoyer le dataframe des abonnés et calculer les abonnés cumulés
        abonnes_df_clean = abonnes_df.dropna(subset=['Date', 'Nouveaux abonnés'])
        abonnes_df_clean['Date'] = pd.to_datetime(abonnes_df_clean['Date'], format='%d/%m/%Y', errors='coerce')
        abonnes_df_clean['Nouveaux abonnés'] = pd.to_numeric(abonnes_df_clean['Nouveaux abonnés'], errors='coerce')
        abonnes_df_clean = abonnes_df_clean.dropna(subset=['Date', 'Nouveaux abonnés'])
        abonnes_df_clean['Cumulative Subscribers'] = abonnes_df_clean['Nouveaux abonnés'].cumsum()

        if abonnes_df_clean.empty:
            st.warning("Le DataFrame 'ABONNES' est vide après nettoyage. Aucune donnée à afficher pour les abonnés.")

        # Calculer le taux d'engagement
        engagement_df['Interactions'] = pd.to_numeric(engagement_df['Interactions'], errors='coerce')
        engagement_df['Impressions'] = pd.to_numeric(engagement_df['Impressions'], errors='coerce')
        engagement_df['Date'] = pd.to_datetime(engagement_df['Date'], format='%d/%m/%Y', errors='coerce')
        engagement_df['Engagement Rate (%)'] = (engagement_df['Interactions'] / engagement_df['Impressions']) * 100

        # Gérer les NaN dans les taux d'engagement
        engagement_df['Engagement Rate (%)'] = engagement_df['Engagement Rate (%)'].fillna(0)

        # Combiner les données pour le traçage
        combined_df = pd.merge(engagement_df, abonnes_df_clean, on='Date', how='left')
        combined_df['Posts per Day'] = combined_df['Date'].map(posts_per_day).fillna(0)

        # Conversion des dates pour Plotly
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])

        # Calculer le taux de croissance des abonnés
        abonnes_df_clean['Growth Rate'] = abonnes_df_clean['Nouveaux abonnés'].pct_change().fillna(0) * 100  # En pourcentage
        combined_df = pd.merge(combined_df, abonnes_df_clean[['Date', 'Growth Rate']], on='Date', how='left')
        combined_df['Growth Rate'] = combined_df['Growth Rate'].fillna(0)

        # Extraire les dates de début et de fin
        if not combined_df['Date'].isnull().all():
            start_date = combined_df['Date'].min().strftime('%d/%m/%Y')
            end_date = combined_df['Date'].max().strftime('%d/%m/%Y')
        else:
            start_date = 'N/A'
            end_date = 'N/A'

        # Mettre à jour le titre de l'application avec la période d'analyse
        st.title(f"Analyse des Performances Réseaux Sociaux - LinkedIn ({start_date} - {end_date})")

        # Vérifier si combined_df est vide
        if combined_df.empty:
            st.warning("Le DataFrame combiné est vide. Aucune donnée à afficher.")
            return None

        # Graphique 1 : Nombre de posts par jour (Bar Chart)
        fig_posts = px.bar(combined_df, x='Date', y='Posts per Day',
                           title='Nombre de Posts par Jour',
                           labels={'Posts per Day': 'Nombre de posts'},
                           template='plotly_dark')

        # Explication pour le graphique 1
        explanation_posts = """
        **Interprétation :** Ce graphique montre le nombre de posts que vous avez publiés chaque jour. 
        Une fréquence de publication régulière peut aider à maintenir l'engagement de votre audience. 
        Identifiez les jours où vous publiez le plus ou le moins et ajustez votre stratégie en conséquence.
        """

        # Graphique 2 : Impressions au fil du temps (Line Chart)
        fig_impressions = px.line(combined_df, x='Date', y='Impressions',
                                  title='Impressions au Fil du Temps',
                                  labels={'Impressions': 'Impressions'},
                                  markers=True,
                                  template='plotly_dark')

        # Explication pour le graphique 2
        explanation_impressions = """
        **Interprétation :** Les impressions représentent le nombre de fois où vos posts ont été affichés. 
        Une augmentation des impressions indique une portée plus large. Analysez les périodes de hausse pour comprendre ce qui a bien fonctionné.
        """

        # Graphique 3 : Interactions au fil du temps (Line Chart)
        fig_interactions = px.line(combined_df, x='Date', y='Interactions',
                                   title='Interactions au Fil du Temps',
                                   labels={'Interactions': 'Interactions'},
                                   markers=True,
                                   template='plotly_dark')

        # Explication pour le graphique 3
        explanation_interactions = """
        **Interprétation :** Les interactions incluent les likes, commentaires et partages de vos posts. 
        Un nombre élevé d'interactions indique un bon engagement de votre audience. Identifiez les types de contenus qui génèrent le plus d'interactions.
        """

        # Graphique 4 : Taux d'engagement au fil du temps (Line Chart)
        fig_engagement = px.line(combined_df, x='Date', y='Engagement Rate (%)',
                                 title='Taux d\'Engagement au Fil du Temps',
                                 labels={'Engagement Rate (%)': 'Taux d\'Engagement (%)'},
                                 markers=True,
                                 template='plotly_dark')

        # Explication pour le graphique 4
        explanation_engagement = """
        **Interprétation :** Le taux d'engagement est calculé en divisant les interactions par les impressions. 
        Un taux d'engagement élevé signifie que votre contenu résonne bien avec votre audience. Suivez ce taux pour évaluer l'efficacité de vos posts.
        """

        # Graphique 5 : Abonnés cumulés au fil du temps (Line Chart)
        fig_subscribers = px.line(combined_df, x='Date', y='Cumulative Subscribers',
                                  title='Abonnés Cumulés au Fil du Temps',
                                  labels={'Cumulative Subscribers': 'Abonnés Cumulés'},
                                  markers=True,
                                  template='plotly_dark')

        # Explication pour le graphique 5
        explanation_subscribers = """
        **Interprétation :** Ce graphique montre l'évolution du nombre total de vos abonnés. 
        Une croissance constante des abonnés est un indicateur positif de votre visibilité et de votre influence sur LinkedIn.
        """

        # Graphique 6 : Corrélation entre abonnés cumulés et taux d'engagement (Scatter Plot)
        fig_corr_abonnes_engagement = px.scatter(combined_df, x='Cumulative Subscribers', y='Engagement Rate (%)',
                                                 title="Corrélation entre Abonnés Cumulés et Taux d'Engagement",
                                                 labels={'Cumulative Subscribers': 'Abonnés Cumulés', 'Engagement Rate (%)': 'Taux d\'Engagement (%)'},
                                                 trendline="ols", template='plotly_dark')

        # Explication pour le graphique 6
        explanation_corr_abonnes_engagement = """
        **Interprétation :** Ce graphique de dispersion montre la relation entre le nombre d'abonnés cumulés et le taux d'engagement. 
        Une tendance positive indique que l'augmentation du nombre d'abonnés est associée à un meilleur engagement. Cela peut aider à identifier si la croissance des abonnés influence directement l'engagement.
        """

        # Graphique 7 : Analyse des pics de croissance des abonnés (Line Chart)
        fig_growth_peaks = px.line(abonnes_df_clean, x='Date', y='Growth Rate',
                                   title="Analyse des Pics de Croissance des Abonnés",
                                   labels={'Date': 'Date', 'Growth Rate': 'Taux de Croissance (%)'},
                                   markers=True, template='plotly_dark')

        # Explication pour le graphique 7
        explanation_growth_peaks = """
        **Interprétation :** Ce graphique montre les variations du taux de croissance des abonnés au fil du temps. 
        Identifiez les périodes de forte croissance pour comprendre quels événements ou contenus ont contribué à l'augmentation rapide de vos abonnés.
        """

        # Graphique 8 : Matrice de Corrélation (Heatmap)
        numeric_cols = ['Interactions', 'Impressions', 'Engagement Rate (%)', 'Cumulative Subscribers', 'Growth Rate']
        corr_matrix = combined_df[numeric_cols].corr()

        fig_corr_matrix = ff.create_annotated_heatmap(
            z=corr_matrix.values,
            x=list(corr_matrix.columns),
            y=list(corr_matrix.index),
            colorscale='Viridis',
            showscale=True,
            hoverinfo='text',
            annotation_text=corr_matrix.round(2).values,
            xgap=2, ygap=2
        )
        fig_corr_matrix.update_layout(title='Matrice de Corrélation',
                                      template='plotly_dark',
                                      width=700, height=700)

        # Explication pour la Matrice de Corrélation
        explanation_corr_matrix = """
        **Interprétation :** La matrice de corrélation montre les relations linéaires entre différentes variables clés. 
        Des coefficients proches de 1 ou -1 indiquent une forte corrélation positive ou négative, respectivement. 
        Cela aide à identifier quelles métriques sont étroitement liées et peuvent influencer votre stratégie de contenu.
        """

        # Graphique 9a : Corrélation entre Impressions et Interactions
        fig_corr_inter_impr = px.scatter(combined_df, x='Impressions', y='Interactions',
                                         title="Corrélation entre Impressions et Interactions",
                                         labels={'Impressions': 'Impressions', 'Interactions': 'Interactions'},
                                         trendline="ols", template='plotly_dark')

        # Explication pour le graphique 9a
        explanation_corr_inter_impr = """
        **Interprétation :** Ce graphique examine la relation entre les impressions et les interactions. 
        Une corrélation positive suggère que plus vos posts sont vus, plus ils génèrent d'interactions. 
        Cela peut indiquer que l'augmentation des impressions pourrait directement améliorer l'engagement.
        """

        # Graphique 9b : Corrélation entre Posts par Jour et Taux d'Engagement
        fig_corr_posts_engagement = px.scatter(combined_df, x='Posts per Day', y='Engagement Rate (%)',
                                               title="Corrélation entre Nombre de Posts et Taux d'Engagement",
                                               labels={'Posts per Day': 'Nombre de Posts par Jour', 'Engagement Rate (%)': 'Taux d\'Engagement (%)'},
                                               trendline="ols", template='plotly_dark')

        # Explication pour le graphique 9b
        explanation_corr_posts_engagement = """
        **Interprétation :** Ce graphique explore la relation entre la fréquence de publication et le taux d'engagement. 
        Une corrélation positive pourrait indiquer que publier plus fréquemment améliore l'engagement, tandis qu'une corrélation négative pourrait suggérer que trop de posts peuvent diluer l'intérêt de votre audience.
        """

        # Graphique 10 : Régression Linéaire pour Prédire le Taux d'Engagement
        def regression_engagement(combined_df):
            # Sélectionner les colonnes pertinentes et gérer les NaN
            X = combined_df[['Impressions', 'Posts per Day', 'Cumulative Subscribers']].dropna()
            y = combined_df.loc[X.index, 'Engagement Rate (%)']

            if X.empty or y.empty:
                st.warning("Pas assez de données pour effectuer la régression linéaire.")
                return None, None

            model = LinearRegression()
            model.fit(X, y)
            predictions = model.predict(X)

            r2 = r2_score(y, predictions)

            fig = px.scatter(x=y, y=predictions, labels={'x': 'Taux d\'Engagement Réel (%)', 'y': 'Taux d\'Engagement Prévu (%)'},
                             title=f'Prédiction du Taux d\'Engagement (R² = {r2:.2f})',
                             template='plotly_dark')
            fig.add_traces(px.line(x=y, y=y).data)
            fig.update_layout(showlegend=False)
            return fig, r2

        fig_regression, r2_value = regression_engagement(combined_df)

        # Explication pour le graphique 10
        if fig_regression is not None:
            explanation_regression = f"""
            **Interprétation :** Ce graphique montre la prédiction du taux d'engagement basé sur les impressions, le nombre de posts par jour et le nombre d'abonnés cumulés. 
            Le coefficient de détermination (R² = {r2_value:.2f}) indique la proportion de la variance du taux d'engagement expliquée par le modèle. 
            Un R² proche de 1 suggère que le modèle prédit bien le taux d'engagement.
            """
        else:
            explanation_regression = "**Interprétation :** Pas assez de données pour effectuer la régression linéaire."

        # Calcul des KPI
        mean_engagement_rate = combined_df['Engagement Rate (%)'].mean()
        mean_growth_rate = combined_df['Growth Rate'].mean()
        total_interactions = combined_df['Interactions'].sum()

        # Recommandations Basées sur les Analyses
        recommendations = """
        **Recommandations :**
        - **Augmentez la fréquence de publication** les jours où le taux d'engagement est élevé.
        - **Ciblez les segments démographiques** qui montrent un engagement supérieur.
        - **Optimisez les heures de publication** en fonction des pics d'impressions et d'interactions.
        - **Analysez les contenus performants** pour identifier les thèmes qui résonnent le plus avec votre audience.
        """

        # Ajout des graphiques démographiques
        # Nettoyer les données démographiques
        demographics_df['Pourcentage'] = demographics_df['Pourcentage'].astype(str)
        demographics_df['Pourcentage'].replace('nan', pd.NA, inplace=True)
        demographics_df['Pourcentage'] = demographics_df['Pourcentage'].str.rstrip('%')
        demographics_df['Pourcentage'] = pd.to_numeric(demographics_df['Pourcentage'], errors='coerce')

        demographics_categories = demographics_df['Principales données démographiques'].dropna().unique()

        # Créer un dictionnaire pour stocker les figures démographiques
        demographics_figures = {}

        for category in demographics_categories:
            df_category = demographics_df[demographics_df['Principales données démographiques'] == category].dropna(subset=['Valeur', 'Pourcentage'])

            if df_category.empty:
                st.warning(f"Aucune donnée valide pour la catégorie démographique '{category}'.")
                continue

            # Trier les valeurs par pourcentage décroissant
            df_category = df_category.sort_values(by='Pourcentage', ascending=False)

            # Créer un graphique en barres pour chaque catégorie
            fig = px.bar(df_category, x='Valeur', y='Pourcentage',
                         title=f'Distribution de {category}',
                         labels={'Valeur': category, 'Pourcentage': 'Pourcentage (%)'},
                         template='plotly_dark')
            fig.update_layout(xaxis_tickangle=-45)
            demographics_figures[category] = fig

        # Préparer le dictionnaire de résultats
        return {
            "fig_posts": fig_posts,
            "explanation_posts": explanation_posts,
            "fig_impressions": fig_impressions,
            "explanation_impressions": explanation_impressions,
            "fig_interactions": fig_interactions,
            "explanation_interactions": explanation_interactions,
            "fig_engagement": fig_engagement,
            "explanation_engagement": explanation_engagement,
            "fig_subscribers": fig_subscribers,
            "explanation_subscribers": explanation_subscribers,
            "fig_corr_abonnes_engagement": fig_corr_abonnes_engagement,
            "explanation_corr_abonnes_engagement": explanation_corr_abonnes_engagement,
            "fig_growth_peaks": fig_growth_peaks,
            "explanation_growth_peaks": explanation_growth_peaks,
            "fig_corr_matrix": fig_corr_matrix,
            "explanation_corr_matrix": explanation_corr_matrix,
            "fig_corr_inter_impr": fig_corr_inter_impr,
            "explanation_corr_inter_impr": explanation_corr_inter_impr,
            "fig_corr_posts_engagement": fig_corr_posts_engagement,
            "explanation_corr_posts_engagement": explanation_corr_posts_engagement,
            "fig_regression": fig_regression,
            "explanation_regression": explanation_regression,
            "demographics_figures": demographics_figures,
            "kpi_mean_engagement_rate": mean_engagement_rate,
            "kpi_mean_growth_rate": mean_growth_rate,
            "kpi_total_interactions": total_interactions,
            "recommendations": recommendations,
            "combined_df": combined_df  # Pour le téléchargement
        }

    # Interface utilisateur
    st.sidebar.header("Paramètres")

    uploaded_file = st.sidebar.file_uploader("Sélectionnez un fichier Excel", type=["xlsx", "xls"])

    if uploaded_file is not None:
        try:
            # Charger le fichier Excel
            xls = pd.ExcelFile(uploaded_file)
            available_sheets = xls.sheet_names

            # Afficher les feuilles disponibles
            st.sidebar.subheader("Feuilles disponibles dans le fichier Excel")
            st.sidebar.write(available_sheets)

            # Vérifier si les feuilles attendues sont présentes
            # Ici, les noms avec accents doivent être exacts
            required_sheets_default = ['ENGAGEMENT', 'ABONNÉS', 'MEILLEURS POSTS', 'DONNÉES DEMOGRAPHIQUES']

            missing_sheets = [sheet for sheet in required_sheets_default if sheet not in available_sheets]
            if missing_sheets:
                st.error(f"Feuilles manquantes dans le fichier Excel : {', '.join(missing_sheets)}")
            else:
                # Permettre à l'utilisateur de sélectionner les feuilles correspondantes
                st.sidebar.subheader("Associez les feuilles Excel aux catégories")

                sheet_engagement = st.sidebar.selectbox("Sélectionnez la feuille 'ENGAGEMENT'", available_sheets, index=available_sheets.index('ENGAGEMENT'))
                sheet_abonnes = st.sidebar.selectbox("Sélectionnez la feuille 'ABONNÉS'", available_sheets, index=available_sheets.index('ABONNÉS'))
                sheet_meilleurs_posts = st.sidebar.selectbox("Sélectionnez la feuille 'MEILLEURS POSTS'", available_sheets, index=available_sheets.index('MEILLEURS POSTS'))
                sheet_demographics = st.sidebar.selectbox("Sélectionnez la feuille 'DONNÉES DEMOGRAPHIQUES'", available_sheets, index=available_sheets.index('DONNÉES DEMOGRAPHIQUES'))

                # Charger les feuilles sélectionnées dans un dictionnaire
                dataframes = {}
                dataframes['ENGAGEMENT'] = pd.read_excel(xls, sheet_engagement)
                dataframes['ABONNES'] = pd.read_excel(xls, sheet_abonnes, skiprows=2)
                dataframes['MEILLEURS POSTS'] = pd.read_excel(xls, sheet_meilleurs_posts).iloc[2:, 1:3]
                dataframes['DONNÉES DEMOGRAPHIQUES'] = pd.read_excel(xls, sheet_demographics)

                # Appel de la fonction principale
                results = generate_performance_graphs(dataframes)

                if results:
                    # Organisation des graphiques dans des onglets
                    tab1, tab2, tab3 = st.tabs(["Performance des Posts", "Engagement et Abonnés", "Analyses Avancées"])

                    with tab1:
                        st.plotly_chart(results["fig_posts"], use_container_width=True)
                        st.markdown(results["explanation_posts"])

                        st.plotly_chart(results["fig_impressions"], use_container_width=True)
                        st.markdown(results["explanation_impressions"])

                        st.plotly_chart(results["fig_interactions"], use_container_width=True)
                        st.markdown(results["explanation_interactions"])

                    with tab2:
                        st.plotly_chart(results["fig_engagement"], use_container_width=True)
                        st.markdown(results["explanation_engagement"])

                        st.plotly_chart(results["fig_subscribers"], use_container_width=True)
                        st.markdown(results["explanation_subscribers"])

                        st.plotly_chart(results["fig_corr_abonnes_engagement"], use_container_width=True)
                        st.markdown(results["explanation_corr_abonnes_engagement"])

                        st.plotly_chart(results["fig_growth_peaks"], use_container_width=True)
                        st.markdown(results["explanation_growth_peaks"])

                    with tab3:
                        st.header("Matrice de Corrélation")
                        st.plotly_chart(results["fig_corr_matrix"], use_container_width=True)
                        st.markdown(results["explanation_corr_matrix"])

                        st.header("Corrélations Supplémentaires")
                        st.plotly_chart(results["fig_corr_inter_impr"], use_container_width=True)
                        st.markdown(results["explanation_corr_inter_impr"])

                        st.plotly_chart(results["fig_corr_posts_engagement"], use_container_width=True)
                        st.markdown(results["explanation_corr_posts_engagement"])

                        st.header("Régression Linéaire")
                        st.plotly_chart(results["fig_regression"], use_container_width=True)
                        st.markdown(results["explanation_regression"])

                        # Ajouter des graphiques démographiques
                        st.header("Données Démographiques")
                        for category, fig in results["demographics_figures"].items():
                            st.plotly_chart(fig, use_container_width=True)

                        # Afficher les KPI
                        st.header("KPI Clés")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Taux d'Engagement Moyen (%)", f"{results['kpi_mean_engagement_rate']:.2f}")
                        col2.metric("Taux de Croissance Moyen (%)", f"{results['kpi_mean_growth_rate']:.2f}")
                        col3.metric("Total des Interactions", f"{int(results['kpi_total_interactions'])}")

                        # Afficher les recommandations
                        st.header("Recommandations")
                        st.markdown(results["recommendations"])

                        # Option de téléchargement
                        csv = convert_df(results["combined_df"])
                        st.download_button(
                            label="Télécharger les Données CSV",
                            data=csv,
                            file_name='combined_data.csv',
                            mime='text/csv',
                        )
        except Exception as e:
            st.error(f"Une erreur est survenue lors de la génération des graphiques : {e}")
