import pandas as pd 
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from io import BytesIO
import plotly.io as pio

# Configuration de la page Streamlit
st.set_page_config(page_title="Analyse des Performances LinkedIn", layout="wide")

# Fonction pour convertir un DataFrame en CSV pour téléchargement
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# Fonction pour convertir une figure Plotly en image PNG
def fig_to_png(fig):
    return pio.to_image(fig, format='png')

# Fonction pour générer les graphiques de performance
def generate_performance_graphs(excel_data):
    try:
        # Charger le fichier Excel
        xls = pd.ExcelFile(excel_data)

        # Charger chaque feuille pertinente dans des dataframes
        engagement_df = pd.read_excel(xls, 'ENGAGEMENT')
        abonnes_df = pd.read_excel(xls, 'ABONNÉS', skiprows=2)
        meilleurs_posts_df = pd.read_excel(xls, 'MEILLEURS POSTS').iloc[2:, 1:3]
        demographics_df = pd.read_excel(xls, 'DONNÉES DÉMOGRAPHIQUES')

        # Nettoyer les noms de colonnes pour enlever les espaces
        engagement_df.columns = engagement_df.columns.str.strip()
        abonnes_df.columns = abonnes_df.columns.str.strip()
        meilleurs_posts_df.columns = meilleurs_posts_df.columns.str.strip()
        demographics_df.columns = demographics_df.columns.str.strip()

        # Nettoyer les données des posts
        meilleurs_posts_df.columns = ['Date de publication', 'Interactions']
        meilleurs_posts_df['Date de publication'] = pd.to_datetime(meilleurs_posts_df['Date de publication'], format='%d/%m/%Y', errors='coerce')
        meilleurs_posts_df['Interactions'] = pd.to_numeric(meilleurs_posts_df['Interactions'], errors='coerce')
        posts_per_day = meilleurs_posts_df['Date de publication'].value_counts().sort_index()

        # Nettoyer le dataframe des abonnés et calculer les abonnés cumulés
        abonnes_df_clean = abonnes_df.dropna()
        date_column_abonnes = [col for col in abonnes_df_clean.columns if 'Date' in col][0]
        abonnes_df_clean.rename(columns={date_column_abonnes: 'Date'}, inplace=True)
        abonnes_df_clean['Date'] = pd.to_datetime(abonnes_df_clean['Date'], format='%d/%m/%Y', errors='coerce')
        abonnes_df_clean['Nouveaux abonnés'] = pd.to_numeric(abonnes_df_clean['Nouveaux abonnés'], errors='coerce')
        abonnes_df_clean['Cumulative Subscribers'] = abonnes_df_clean['Nouveaux abonnés'].cumsum()

        # Calculer le taux d'engagement
        engagement_df['Interactions'] = pd.to_numeric(engagement_df['Interactions'], errors='coerce')
        engagement_df['Impressions'] = pd.to_numeric(engagement_df['Impressions'], errors='coerce')
        engagement_df['Date'] = pd.to_datetime(engagement_df['Date'], format='%d/%m/%Y', errors='coerce')
        engagement_df['Engagement Rate (%)'] = (engagement_df['Interactions'] / engagement_df['Impressions']) * 100

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
        start_date = combined_df['Date'].min().strftime('%d/%m/%Y')
        end_date = combined_df['Date'].max().strftime('%d/%m/%Y')

        # Mettre à jour le titre de l'application avec la période d'analyse
        st.title(f"Analyse des Performances Réseaux Sociaux - LinkedIn ({start_date} - {end_date})")

        # Définir une palette de couleurs cohérente et attrayante
        color_palette = {
            'Posts per Day': 'rgba(54, 162, 235, 0.8)',        # Bleu
            'Impressions': 'rgba(255, 99, 132, 0.8)',         # Rouge
            'Interactions': 'rgba(75, 192, 192, 0.8)',        # Vert
            'Engagement Rate (%)': 'rgba(255, 206, 86, 0.8)', # Jaune
            'Cumulative Subscribers': 'rgba(153, 102, 255, 0.8)', # Violet
            'Growth Rate': 'rgba(255, 159, 64, 0.8)'          # Orange
        }

        # Graphiques principaux (ordre d'importance)
        # 1. Taux d'Engagement
        fig_engagement = px.line(combined_df, x='Date', y='Engagement Rate (%)',
                                 title='Taux d\'Engagement au Fil du Temps',
                                 labels={'Engagement Rate (%)': 'Taux d\'Engagement (%)'},
                                 markers=True,
                                 template='plotly_dark',
                                 line_shape='linear',
                                 color_discrete_sequence=[color_palette['Engagement Rate (%)']])

        # 2. Impressions au Fil du Temps
        fig_impressions = px.line(combined_df, x='Date', y='Impressions',
                                  title='Impressions au Fil du Temps',
                                  labels={'Impressions': 'Impressions'},
                                  markers=True,
                                  template='plotly_dark',
                                  line_shape='linear',
                                  color_discrete_sequence=[color_palette['Impressions']])

        # 3. Interactions au Fil du Temps
        fig_interactions = px.line(combined_df, x='Date', y='Interactions',
                                   title='Interactions au Fil du Temps',
                                   labels={'Interactions': 'Interactions'},
                                   markers=True,
                                   template='plotly_dark',
                                   line_shape='linear',
                                   color_discrete_sequence=[color_palette['Interactions']])

        # 4. Abonnés Cumulés au Fil du Temps
        fig_subscribers = px.line(combined_df, x='Date', y='Cumulative Subscribers',
                                  title='Abonnés Cumulés au Fil du Temps',
                                  labels={'Cumulative Subscribers': 'Abonnés Cumulés'},
                                  markers=True,
                                  template='plotly_dark',
                                  line_shape='linear',
                                  color_discrete_sequence=[color_palette['Cumulative Subscribers']])

        # 5. Nombre de Posts par Jour (Histogramme)
        fig_posts_bar = px.bar(combined_df, x='Date', y='Posts per Day',
                               title='Nombre de Posts par Jour',
                               labels={'Posts per Day': 'Nombre de posts', 'Date': 'Date'},
                               template='plotly_dark',
                               color_discrete_sequence=[color_palette['Posts per Day']])

        # 6. Corrélation entre Abonnés Cumulés et Taux d'Engagement
        fig_corr_abonnes_engagement = px.scatter(combined_df, x='Cumulative Subscribers', y='Engagement Rate (%)',
                                                 title="Corrélation entre Abonnés Cumulés et Taux d'Engagement",
                                                 labels={'Cumulative Subscribers': 'Abonnés Cumulés', 'Engagement Rate (%)': 'Taux d\'Engagement (%)'},
                                                 trendline="ols",
                                                 template='plotly_dark',
                                                 color_discrete_sequence=[color_palette['Cumulative Subscribers']])

        # 7. Analyse des Pics de Croissance des Abonnés
        fig_growth_peaks = px.line(abonnes_df_clean, x='Date', y='Growth Rate',
                                   title="Analyse des Pics de Croissance des Abonnés",
                                   labels={'Date': 'Date', 'Growth Rate': 'Taux de Croissance (%)'},
                                   markers=True, 
                                   template='plotly_dark',
                                   line_shape='linear',
                                   color_discrete_sequence=[color_palette['Growth Rate']])

        # 8. Matrice de Corrélation (Affichage en grand)
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
                                      width=900, height=900)

        # 9a. Corrélation entre Impressions et Interactions
        fig_corr_inter_impr = px.scatter(combined_df, x='Impressions', y='Interactions',
                                         title="Corrélation entre Impressions et Interactions",
                                         labels={'Impressions': 'Impressions', 'Interactions': 'Interactions'},
                                         trendline="ols",
                                         template='plotly_dark',
                                         color_discrete_sequence=[color_palette['Impressions']])

        # 9b. Corrélation entre Posts par Jour et Taux d'Engagement
        fig_corr_posts_engagement = px.scatter(combined_df, x='Posts per Day', y='Engagement Rate (%)',
                                               title="Corrélation entre Nombre de Posts et Taux d'Engagement",
                                               labels={'Posts per Day': 'Nombre de Posts par Jour', 'Engagement Rate (%)': 'Taux d\'Engagement (%)'},
                                               trendline="ols",
                                               template='plotly_dark',
                                               color_discrete_sequence=[color_palette['Posts per Day']])

        # 10. Régression Linéaire pour Prédire le Taux d'Engagement
        def regression_engagement(combined_df):
            X = combined_df[['Impressions', 'Posts per Day', 'Cumulative Subscribers']]
            y = combined_df['Engagement Rate (%)']

            model = LinearRegression()
            model.fit(X, y)
            predictions = model.predict(X)

            r2 = r2_score(y, predictions)

            fig = px.scatter(x=y, y=predictions, labels={'x': 'Taux d\'Engagement Réel (%)', 'y': 'Taux d\'Engagement Prévu (%)'},
                             title=f'Prédiction du Taux d\'Engagement (R² = {r2:.2f})',
                             template='plotly_dark',
                             color_discrete_sequence=[color_palette['Engagement Rate (%)']])
            fig.add_traces(px.line(x=y, y=y).data)
            fig.update_layout(showlegend=False)
            return fig, r2

        fig_regression, r2_value = regression_engagement(combined_df)

        # Calcul des KPI
        mean_engagement_rate = combined_df['Engagement Rate (%)'].mean()
        mean_growth_rate = combined_df['Growth Rate'].mean()
        total_interactions = combined_df['Interactions'].sum()
        total_impressions = combined_df['Impressions'].sum()  # Nouveau KPI pour les impressions totales

        # Recommandations Basées sur les Analyses
        recommendations = """
        **Recommandations :**
        - **Augmentez la fréquence de publication** les jours où le taux d'engagement est élevé.
        - **Ciblez les segments démographiques** qui montrent un engagement supérieur.
        - **Optimisez les heures de publication** en fonction des pics d'impressions et d'interactions.
        - **Analysez les contenus performants** pour identifier les thèmes qui résonnent le plus avec votre audience.
        """

        # Ajout des graphiques démographiques (Conversion en Camemberts)
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

            # Regrouper les petites catégories en "Autres" si nécessaire
            threshold = 1  # Réduire le seuil pour éviter que tout soit "Autres"
            df_category.loc[df_category['Pourcentage'] < threshold, 'Valeur'] = 'Autres'
            df_category = df_category.groupby('Valeur', as_index=False).sum()

            # Vérifier si après regroupement, il y a au moins deux catégories
            if len(df_category) < 2:
                # Si toutes les valeurs sont "Autres", afficher les catégories sans regroupement
                df_category = demographics_df[demographics_df['Principales données démographiques'] == category].copy()
                df_category = df_category.groupby('Valeur', as_index=False).sum()

            # Créer un graphique en camembert pour chaque catégorie avec améliorations
            fig = px.pie(df_category, values='Pourcentage', names='Valeur',
                         title=f'Distribution de {category}',
                         template='plotly_dark',
                         hole=0.3,
                         color_discrete_sequence=px.colors.qualitative.Bold)

            # Ajouter des étiquettes de pourcentage et placer la légende en dessous
            fig.update_traces(textinfo='percent+label', textposition='inside')
            fig.update_layout(
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.2,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10)  # Ajustez la taille selon vos besoins
                ),
                height=600  # Augmenter la hauteur pour accommoder la légende
            )

            demographics_figures[category] = fig

        # Création du Graphique Combiné : Posts, Impressions, Interactions par Jour avec 2 axes Y
        fig_combined = go.Figure()

        # Ajouter les Posts par Jour en tant que Barres (y1)
        fig_combined.add_trace(go.Bar(
            x=combined_df['Date'],
            y=combined_df['Posts per Day'],
            name='Nombre de Posts',
            marker_color=color_palette['Posts per Day'],
            opacity=0.6
        ))

        # Ajouter les Impressions par Jour en tant que Ligne avec Axe Y2
        fig_combined.add_trace(go.Scatter(
            x=combined_df['Date'],
            y=combined_df['Impressions'],
            name='Impressions',
            mode='lines+markers',
            marker=dict(color=color_palette['Impressions'], size=8),
            line=dict(width=2),
            yaxis='y2'
        ))

        # Ajouter les Interactions par Jour en tant que Ligne avec Axe Y2
        fig_combined.add_trace(go.Scatter(
            x=combined_df['Date'],
            y=combined_df['Interactions'],
            name='Interactions',
            mode='lines+markers',
            marker=dict(color=color_palette['Interactions'], size=10, symbol='diamond'),
            line=dict(width=3, dash='dot'),
            yaxis='y2'
        ))

        # Mise en page du graphique combiné avec 2 axes Y
        fig_combined.update_layout(
            title='Nombre de Posts, Impressions et Interactions par Jour',
            xaxis=dict(title='Date'),
            yaxis=dict(
                title='Nombre de Posts',
                titlefont=dict(color=color_palette['Posts per Day']),
                tickfont=dict(color=color_palette['Posts per Day'])
            ),
            yaxis2=dict(
                title='Impressions / Interactions',
                titlefont=dict(color=color_palette['Impressions']),
                tickfont=dict(color=color_palette['Impressions']),
                overlaying='y',
                side='right',
                showgrid=False
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            template='plotly_dark',
            barmode='group',
            height=700
        )

        # Création du Nouveau Graphique Combiné : Impressions et Interactions par Jour avec 2 axes Y
        fig_impr_inter = go.Figure()

        # Ajouter les Impressions par Jour en tant que Ligne avec Axe Y1
        fig_impr_inter.add_trace(go.Scatter(
            x=combined_df['Date'],
            y=combined_df['Impressions'],
            name='Impressions',
            mode='lines+markers',
            marker=dict(color=color_palette['Impressions'], size=8),
            line=dict(width=2),
            yaxis='y1'
        ))

        # Ajouter les Interactions par Jour en tant que Ligne avec Axe Y2
        fig_impr_inter.add_trace(go.Scatter(
            x=combined_df['Date'],
            y=combined_df['Interactions'],
            name='Interactions',
            mode='lines+markers',
            marker=dict(color=color_palette['Interactions'], size=10, symbol='diamond'),
            line=dict(width=3, dash='dot'),
            yaxis='y2'
        ))

        # Mise en page du nouveau graphique combiné avec 2 axes Y
        fig_impr_inter.update_layout(
            title='Impressions et Interactions par Jour',
            xaxis=dict(title='Date'),
            yaxis=dict(
                title='Impressions',
                titlefont=dict(color=color_palette['Impressions']),
                tickfont=dict(color=color_palette['Impressions'])
            ),
            yaxis2=dict(
                title='Interactions',
                titlefont=dict(color=color_palette['Interactions']),
                tickfont=dict(color=color_palette['Interactions']),
                overlaying='y',
                side='right',
                showgrid=False
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            template='plotly_dark',
            height=700
        )

        # Préparer le dictionnaire de retour
        return {
            "fig_posts_bar": fig_posts_bar,
            "explanation_posts_bar": """
            **Interprétation :** Cet histogramme montre le nombre de posts publiés chaque jour. 
            Une fréquence de publication régulière peut aider à maintenir l'engagement de votre audience. 
            Identifiez les jours où vous publiez le plus ou le moins et ajustez votre stratégie en conséquence.
            """,
            "fig_impressions": fig_impressions,
            "explanation_impressions": """
            **Interprétation :** Les impressions représentent le nombre de fois où vos posts ont été affichés. 
            Une augmentation des impressions indique une portée plus large. Analysez les périodes de hausse pour comprendre ce qui a bien fonctionné.
            """,
            "fig_interactions": fig_interactions,
            "explanation_interactions": """
            **Interprétation :** Les interactions incluent les likes, commentaires et partages de vos posts. 
            Un nombre élevé d'interactions indique un bon engagement de votre audience. Identifiez les types de contenus qui génèrent le plus d'interactions.
            """,
            "fig_engagement": fig_engagement,
            "explanation_engagement": """
            **Interprétation :** Le taux d'engagement est calculé en divisant les interactions par les impressions. 
            Un taux d'engagement élevé signifie que votre contenu résonne bien avec votre audience. Suivez ce taux pour évaluer l'efficacité de vos posts.
            """,
            "fig_subscribers": fig_subscribers,
            "explanation_subscribers": """
            **Interprétation :** Ce graphique montre l'évolution du nombre total de vos abonnés. 
            Une croissance constante des abonnés est un indicateur positif de votre visibilité et de votre influence sur LinkedIn.
            """,
            "fig_corr_abonnes_engagement": fig_corr_abonnes_engagement,
            "explanation_corr_abonnes_engagement": """
            **Interprétation :** Ce graphique de dispersion montre la relation entre le nombre d'abonnés cumulés et le taux d'engagement. 
            Une tendance positive indique que l'augmentation du nombre d'abonnés est associée à un meilleur engagement. Cela peut aider à identifier si la croissance des abonnés influence directement l'engagement.
            """,
            "fig_growth_peaks": fig_growth_peaks,
            "explanation_growth_peaks": """
            **Interprétation :** Ce graphique montre les variations du taux de croissance des abonnés au fil du temps. 
            Identifiez les périodes de forte croissance pour comprendre quels événements ou contenus ont contribué à l'augmentation rapide de vos abonnés.
            """,
            "fig_corr_matrix": fig_corr_matrix,
            "explanation_corr_matrix": """
            **Interprétation :** La matrice de corrélation montre les relations linéaires entre différentes variables clés. 
            Des coefficients proches de 1 ou -1 indiquent une forte corrélation positive ou négative, respectivement. 
            Cela aide à identifier quelles métriques sont étroitement liées et peuvent influencer votre stratégie de contenu.
            """,
            "fig_corr_inter_impr": fig_corr_inter_impr,
            "explanation_corr_inter_impr": """
            **Interprétation :** Ce graphique examine la relation entre les impressions et les interactions. 
            Une corrélation positive suggère que plus vos posts sont vus, plus ils génèrent d'interactions. 
            Cela peut indiquer que l'augmentation des impressions pourrait directement améliorer l'engagement.
            """,
            "fig_corr_posts_engagement": fig_corr_posts_engagement,
            "explanation_corr_posts_engagement": """
            **Interprétation :** Ce graphique explore la relation entre la fréquence de publication et le taux d'engagement. 
            Une corrélation positive pourrait indiquer que publier plus fréquemment améliore l'engagement, tandis qu'une corrélation négative pourrait suggérer que trop de posts peuvent diluer l'intérêt de votre audience.
            """,
            "fig_regression": fig_regression,
            "explanation_regression": f"""
            **Interprétation :** Ce graphique montre la prédiction du taux d'engagement basé sur les impressions, le nombre de posts par jour et le nombre d'abonnés cumulés. 
            Le coefficient de détermination (R² = {r2_value:.2f}) indique la proportion de la variance du taux d'engagement expliquée par le modèle. 
            Un R² proche de 1 suggère que le modèle prédit bien le taux d'engagement.
            """,
            "fig_combined": fig_combined,
            "explanation_combined": """
            **Interprétation :** Ce graphique combiné affiche le nombre de posts par jour sous forme de barres, ainsi que les impressions et les interactions par jour sous forme de lignes avec un axe Y secondaire. 
            Cela permet de visualiser simultanément la fréquence des publications et leur impact en termes de portée et d'engagement.
            """,
            "fig_impr_inter": fig_impr_inter,
            "explanation_impr_inter": """
            **Interprétation :** Ce graphique combiné affiche les impressions et les interactions par jour sous forme de lignes avec des axes Y séparés. 
            Cela permet de comparer directement la portée de vos posts avec le niveau d'engagement généré.
            """,
            "demographics_figures": demographics_figures,
            "kpi_mean_engagement_rate": mean_engagement_rate,
            "kpi_mean_growth_rate": mean_growth_rate,
            "kpi_total_interactions": total_interactions,
            "kpi_total_impressions": total_impressions,  # Nouveau KPI pour les impressions totales
            "recommendations": recommendations,
            "combined_df": combined_df  # Pour le téléchargement
        }

    except Exception as e:
        st.error(f"Une erreur est survenue lors de la génération des graphiques : {e}")
        return None

# Interface utilisateur
st.sidebar.header("Paramètres")

uploaded_file = st.sidebar.file_uploader("Sélectionnez un fichier Excel", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Appel de la fonction avec gestion des exceptions
    results = generate_performance_graphs(uploaded_file)

    if results:
        # Organisation des graphiques dans des onglets avec "Engagement et Abonnés" en premier
        tab_engagement, tab_posts, tab_advanced = st.tabs(["Engagement et Abonnés", "Performance des Posts", "Analyses Avancées"])

        with tab_engagement:
            st.header("Engagement et Abonnés")

            # Indicateurs Clés de Performance (KPI)
            st.subheader("Indicateurs Clés de Performance (KPI)")
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Taux d'Engagement Moyen", f"{results['kpi_mean_engagement_rate']:.2f}%")
            kpi2.metric("Croissance Moyenne des Abonnés", f"{results['kpi_mean_growth_rate']:.2f}%")
            kpi3.metric("Total des Interactions", f"{results['kpi_total_interactions']}")
            kpi4.metric("Total des Impressions", f"{results['kpi_total_impressions']}")

            st.subheader("Recommandations Basées sur les Analyses")
            st.markdown(results["recommendations"])

            # Disposition en deux colonnes : Abonnés Cumulés et Corrélation Abonnés-Engagement
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(results["fig_subscribers"], use_container_width=True)
                st.markdown(results["explanation_subscribers"])
                # Bouton de téléchargement individuel
                buf = fig_to_png(results["fig_subscribers"])
                st.download_button(
                    label="Télécharger ce graphique en PNG",
                    data=buf,
                    file_name="abonnes_cumules.png",
                    mime="image/png",
                )

            with col2:
                st.plotly_chart(results["fig_corr_abonnes_engagement"], use_container_width=True)
                st.markdown(results["explanation_corr_abonnes_engagement"])
                # Bouton de téléchargement individuel
                buf = fig_to_png(results["fig_corr_abonnes_engagement"])
                st.download_button(
                    label="Télécharger ce graphique en PNG",
                    data=buf,
                    file_name="corr_abonnes_engagement.png",
                    mime="image/png",
                )

            # Disposition en deux colonnes : Croissance des Abonnés et Régression Linéaire
            col3, col4 = st.columns(2)
            with col3:
                st.plotly_chart(results["fig_growth_peaks"], use_container_width=True)
                st.markdown(results["explanation_growth_peaks"])
                # Bouton de téléchargement individuel
                buf = fig_to_png(results["fig_growth_peaks"])
                st.download_button(
                    label="Télécharger ce graphique en PNG",
                    data=buf,
                    file_name="growth_peaks.png",
                    mime="image/png",
                )

            with col4:
                st.plotly_chart(results["fig_regression"], use_container_width=True)
                st.markdown(results["explanation_regression"])
                # Bouton de téléchargement individuel
                buf = fig_to_png(results["fig_regression"])
                st.download_button(
                    label="Télécharger ce graphique en PNG",
                    data=buf,
                    file_name="regression_engagement.png",
                    mime="image/png",
                )

        with tab_posts:
            st.header("Performance des Posts")

            # Graphique Combiné : Posts, Impressions, Interactions par Jour avec 2 axes Y
            st.subheader("Nombre de Posts, Impressions et Interactions par Jour")
            st.plotly_chart(results["fig_combined"], use_container_width=True)
            st.markdown(results["explanation_combined"])
            st.download_button(
                label="Télécharger ce graphique en PNG",
                data=fig_to_png(results["fig_combined"]),
                file_name="combined_posts_impressions_interactions.png",
                mime="image/png",
            )

            # Nouveau Graphique Combiné : Impressions et Interactions par Jour avec 2 axes Y
            st.subheader("Impressions et Interactions par Jour")
            st.plotly_chart(results["fig_impr_inter"], use_container_width=True)
            st.markdown(results["explanation_impr_inter"])
            st.download_button(
                label="Télécharger ce graphique en PNG",
                data=fig_to_png(results["fig_impr_inter"]),
                file_name="impressions_interactions.png",
                mime="image/png",
            )

            # Disposition en deux colonnes : Nombre de Posts (Histogramme) et Impressions
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(results["fig_posts_bar"], use_container_width=True)
                st.markdown(results["explanation_posts_bar"])
                # Bouton de téléchargement individuel
                buf = fig_to_png(results["fig_posts_bar"])
                st.download_button(
                    label="Télécharger ce graphique en PNG",
                    data=buf,
                    file_name="posts_bar.png",
                    mime="image/png",
                )

            with col2:
                st.plotly_chart(results["fig_impressions"], use_container_width=True)
                st.markdown(results["explanation_impressions"])
                # Bouton de téléchargement individuel
                buf = fig_to_png(results["fig_impressions"])
                st.download_button(
                    label="Télécharger ce graphique en PNG",
                    data=buf,
                    file_name="impressions.png",
                    mime="image/png",
                )

            # Disposition en deux colonnes : Interactions et Taux d'Engagement
            col3, col4 = st.columns(2)
            with col3:
                st.plotly_chart(results["fig_interactions"], use_container_width=True)
                st.markdown(results["explanation_interactions"])
                # Bouton de téléchargement individuel
                buf = fig_to_png(results["fig_interactions"])
                st.download_button(
                    label="Télécharger ce graphique en PNG",
                    data=buf,
                    file_name="interactions.png",
                    mime="image/png",
                )

            with col4:
                st.plotly_chart(results["fig_engagement"], use_container_width=True)
                st.markdown(results["explanation_engagement"])
                # Bouton de téléchargement individuel
                buf = fig_to_png(results["fig_engagement"])
                st.download_button(
                    label="Télécharger ce graphique en PNG",
                    data=buf,
                    file_name="engagement.png",
                    mime="image/png",
                )

        with tab_advanced:
            st.header("Analyses Avancées")

            # Affichage de la Matrice de Corrélation en grand
            st.subheader("Matrice de Corrélation")
            st.plotly_chart(results["fig_corr_matrix"], use_container_width=True)
            st.markdown(results["explanation_corr_matrix"])
            # Bouton de téléchargement individuel
            buf = fig_to_png(results["fig_corr_matrix"])
            st.download_button(
                label="Télécharger ce graphique en PNG",
                data=buf,
                file_name="corr_matrix.png",
                mime="image/png",
            )

            # Disposition en deux colonnes pour les corrélations supplémentaires
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(results["fig_corr_inter_impr"], use_container_width=True)
                st.markdown(results["explanation_corr_inter_impr"])
                # Bouton de téléchargement individuel
                buf = fig_to_png(results["fig_corr_inter_impr"])
                st.download_button(
                    label="Télécharger ce graphique en PNG",
                    data=buf,
                    file_name="corr_inter_impr.png",
                    mime="image/png",
                )

            with col2:
                st.plotly_chart(results["fig_corr_posts_engagement"], use_container_width=True)
                st.markdown(results["explanation_corr_posts_engagement"])
                # Bouton de téléchargement individuel
                buf = fig_to_png(results["fig_corr_posts_engagement"])
                st.download_button(
                    label="Télécharger ce graphique en PNG",
                    data=buf,
                    file_name="corr_posts_engagement.png",
                    mime="image/png",
                )

            # Section : Données Démographiques
            st.header("Données Démographiques")
            demographics_figures = results["demographics_figures"]
            categories = list(demographics_figures.keys())
            num_cols = 2  # Nombre de colonnes par ligne

            for i in range(0, len(categories), num_cols):
                cols = st.columns(num_cols)
                for j in range(num_cols):
                    if i + j < len(categories):
                        category = categories[i + j]
                        fig = demographics_figures[category]
                        with cols[j]:
                            st.plotly_chart(fig, use_container_width=True)
                            # Bouton de téléchargement individuel
                            buf = fig_to_png(fig)
                            st.download_button(
                                label=f"Télécharger {category} en PNG",
                                data=buf,
                                file_name=f"distribution_{category}.png",
                                mime="image/png",
                            )

            # Section : Téléchargement des Données
            st.header("Télécharger les Données Analytiques")
            csv = convert_df(results["combined_df"])
            st.download_button(
                label="Télécharger les Données Analytiques",
                data=csv,
                file_name='analyse_linkedin.csv',
                mime='text/csv',
            )

    else:
        st.error("Erreur dans la génération des graphiques. Veuillez vérifier vos données.")
else:
    st.info("Veuillez télécharger un fichier Excel pour commencer l'analyse.")
