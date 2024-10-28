import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from io import BytesIO
from fpdf import FPDF
import base64

# Configuration de la page Streamlit
st.set_page_config(page_title="Analyse des Performances LinkedIn", layout="wide")

# Fonction pour convertir un DataFrame en CSV pour téléchargement
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# Fonction pour rendre les URL cliquables en HTML
def make_clickable(url):
    return f'<a href="{url}" target="_blank">{url}</a>'

# Fonction pour afficher les Top 5 posts performants
def top_performing_posts(meilleurs_posts_df_clean):
    # Convertir les colonnes en types appropriés si ce n'est pas déjà fait
    meilleurs_posts_df_clean['Interactions'] = pd.to_numeric(meilleurs_posts_df_clean['Interactions'], errors='coerce')
    meilleurs_posts_df_clean['Impressions'] = pd.to_numeric(meilleurs_posts_df_clean['Impressions'], errors='coerce')
    meilleurs_posts_df_clean['Date_de_publication_post'] = pd.to_datetime(meilleurs_posts_df_clean['Date_de_publication_post'], format='%Y-%m-%d', errors='coerce')
    
    # Trier les posts par le nombre d'interactions pour obtenir le top 5
    top_5_posts = meilleurs_posts_df_clean.sort_values(by='Interactions', ascending=False).head(5)
    
    # Rendre les URL cliquables
    top_5_posts['URL_du_post'] = top_5_posts['URL_du_post'].apply(lambda url: make_clickable(url))
    
    # Sélectionner les colonnes pertinentes pour l'affichage
    top_5_posts_display = top_5_posts[['URL_du_post', 'Date_de_publication_post', 'Interactions', 'Impressions']].copy()
    
    # Retourner le tableau en HTML pour affichage dans Streamlit
    return top_5_posts_display.to_html(escape=False, index=False)

# Fonctions pour exporter les graphiques en PDF
def save_fig_as_image(fig):
    """
    Sauvegarde une figure Plotly en tant qu'image PNG en mémoire.
    """
    img_bytes = fig.to_image(format="png")
    return img_bytes

def create_pdf(figures, titles):
    """
    Crée un PDF à partir d'une liste d'images en mémoire.
    """
    pdf = FPDF()
    for img, title in zip(figures, titles):
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, title)
        # Enregistrer l'image temporairement
        with BytesIO(img) as img_buffer:
            encoded = base64.b64encode(img_buffer.read()).decode('latin1')
            pdf.image(f"data:image/png;base64,{encoded}", x=10, y=20, w=190)
    return pdf.output(dest="S").encode("latin1")

# Fonction pour générer les graphiques de performance
def generate_performance_graphs(excel_data):
    try:
        # Charger le fichier Excel
        xls = pd.ExcelFile(excel_data)

        # Charger chaque feuille pertinente dans des dataframes
        engagement_df = pd.read_excel(xls, 'ENGAGEMENT')
        abonnes_df = pd.read_excel(xls, 'ABONNÉS', skiprows=2)
        meilleurs_posts_df = pd.read_excel(xls, 'MEILLEURS POSTS', skiprows=2, usecols="B:E")  # Colonnes B à E
        demographics_df = pd.read_excel(xls, 'DONNÉES DÉMOGRAPHIQUES')

        # Nettoyer les noms de colonnes pour enlever les espaces
        engagement_df.columns = engagement_df.columns.str.strip()
        abonnes_df.columns = abonnes_df.columns.str.strip()
        meilleurs_posts_df.columns = meilleurs_posts_df.columns.str.strip()
        demographics_df.columns = demographics_df.columns.str.strip()

        # Vérifier les colonnes lues dans 'MEILLEURS POSTS'
        st.write("Colonnes dans 'MEILLEURS POSTS':", meilleurs_posts_df.columns.tolist())

        # Nettoyer les données des posts
        meilleurs_posts_df.columns = ['URL_du_post', 'Date_de_publication_post', 'Interactions', 'Impressions']
        meilleurs_posts_df['Date_de_publication_post'] = pd.to_datetime(meilleurs_posts_df['Date_de_publication_post'], format='%Y-%m-%d', errors='coerce')
        meilleurs_posts_df['Interactions'] = pd.to_numeric(meilleurs_posts_df['Interactions'], errors='coerce')
        meilleurs_posts_df['Impressions'] = pd.to_numeric(meilleurs_posts_df['Impressions'], errors='coerce')
        meilleurs_posts_df_clean = meilleurs_posts_df.copy()

        # Remplir les NaN dans 'Impressions' si nécessaire, par exemple avec 0 ou une autre stratégie
        meilleurs_posts_df_clean['Impressions'] = meilleurs_posts_df_clean['Impressions'].fillna(0)

        posts_per_day = meilleurs_posts_df_clean['Date_de_publication_post'].value_counts().sort_index()

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

        # Graphiques principaux (ordre d'importance)
        # 1. Taux d'Engagement
        fig_engagement = px.line(combined_df, x='Date', y='Engagement Rate (%)',
                                 title='Taux d\'Engagement au Fil du Temps',
                                 labels={'Engagement Rate (%)': 'Taux d\'Engagement (%)'},
                                 markers=True,
                                 template='plotly_dark')

        # 2. Impressions au Fil du Temps
        fig_impressions = px.line(combined_df, x='Date', y='Impressions',
                                  title='Impressions au Fil du Temps',
                                  labels={'Impressions': 'Impressions'},
                                  markers=True,
                                  template='plotly_dark')

        # 3. Interactions au Fil du Temps
        fig_interactions = px.line(combined_df, x='Date', y='Interactions',
                                   title='Interactions au Fil du Temps',
                                   labels={'Interactions': 'Interactions'},
                                   markers=True,
                                   template='plotly_dark')

        # 4. Abonnés Cumulés au Fil du Temps
        fig_subscribers = px.line(combined_df, x='Date', y='Cumulative Subscribers',
                                  title='Abonnés Cumulés au Fil du Temps',
                                  labels={'Cumulative Subscribers': 'Abonnés Cumulés'},
                                  markers=True,
                                  template='plotly_dark')

        # 5. Nombre de Posts par Jour (Histogramme)
        fig_posts_bar = px.bar(combined_df, x='Date', y='Posts per Day',
                               title='Nombre de Posts par Jour',
                               labels={'Posts per Day': 'Nombre de posts', 'Date': 'Date'},
                               template='plotly_dark')

        # 6. Corrélation entre Abonnés Cumulés et Taux d'Engagement
        fig_corr_abonnes_engagement = px.scatter(combined_df, x='Cumulative Subscribers', y='Engagement Rate (%)',
                                                 title="Corrélation entre Abonnés Cumulés et Taux d'Engagement",
                                                 labels={'Cumulative Subscribers': 'Abonnés Cumulés', 'Engagement Rate (%)': 'Taux d\'Engagement (%)'},
                                                 trendline="ols", template='plotly_dark')

        # 7. Analyse des Pics de Croissance des Abonnés
        fig_growth_peaks = px.line(abonnes_df_clean, x='Date', y='Growth Rate',
                                   title="Analyse des Pics de Croissance des Abonnés",
                                   labels={'Date': 'Date', 'Growth Rate': 'Taux de Croissance (%)'},
                                   markers=True, template='plotly_dark')

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
                                         trendline="ols", template='plotly_dark')

        # 9b. Corrélation entre Posts par Jour et Taux d'Engagement
        fig_corr_posts_engagement = px.scatter(combined_df, x='Posts per Day', y='Engagement Rate (%)',
                                               title="Corrélation entre Nombre de Posts et Taux d'Engagement",
                                               labels={'Posts per Day': 'Nombre de Posts par Jour', 'Engagement Rate (%)': 'Taux d\'Engagement (%)'},
                                               trendline="ols", template='plotly_dark')

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
                             template='plotly_dark')
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

            # Créer un graphique en camembert pour chaque catégorie avec des améliorations
            fig = px.pie(df_category, values='Pourcentage', names='Valeur',
                         title=f'Distribution de {category}',
                         template='plotly_dark',
                         hole=0.3,
                         labels={'Valeur': 'Valeur'},
                         hover_data=['Pourcentage'],
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            demographics_figures[category] = fig

        # Préparer le dictionnaire de retour
        return {
            "fig_engagement": fig_engagement,
            "explanation_engagement": """
            **Interprétation :** Le taux d'engagement est calculé en divisant les interactions par les impressions. 
            Un taux d'engagement élevé signifie que votre contenu résonne bien avec votre audience. Suivez ce taux pour évaluer l'efficacité de vos posts.
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
            "fig_posts_bar": fig_posts_bar,
            "explanation_posts_bar": """
            **Interprétation :** Cet histogramme montre le nombre de posts publiés chaque jour. 
            Une fréquence de publication régulière peut aider à maintenir l'engagement de votre audience. 
            Identifiez les jours où vous publiez le plus ou le moins et ajustez votre stratégie en conséquence.
            """,
            "demographics_figures": demographics_figures,
            "kpi_mean_engagement_rate": mean_engagement_rate,
            "kpi_mean_growth_rate": mean_growth_rate,
            "kpi_total_interactions": total_interactions,
            "kpi_total_impressions": total_impressions,  # Nouveau KPI pour les impressions totales
            "recommendations": recommendations,
            "combined_df": combined_df,  # Pour le téléchargement
            "meilleurs_posts_df_clean": meilleurs_posts_df_clean  # Ajout pour le Top 5 des posts
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

            with col2:
                st.plotly_chart(results["fig_corr_abonnes_engagement"], use_container_width=True)
                st.markdown(results["explanation_corr_abonnes_engagement"])

            # Disposition en deux colonnes : Croissance des Abonnés et Régression Linéaire
            col3, col4 = st.columns(2)
            with col3:
                st.plotly_chart(results["fig_growth_peaks"], use_container_width=True)
                st.markdown(results["explanation_growth_peaks"])

            with col4:
                st.plotly_chart(results["fig_regression"], use_container_width=True)
                st.markdown(results["explanation_regression"])

        with tab_posts:
            st.header("Performance des Posts")

            # Affichage des Top 5 posts performants
            st.subheader("Top 5 Posts Performants")
            top_5_html = top_performing_posts(results["meilleurs_posts_df_clean"])
            st.write(top_5_html, unsafe_allow_html=True)

            # Disposition en deux colonnes : Nombre de Posts (Histogramme) et Impressions
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(results["fig_posts_bar"], use_container_width=True)
                st.markdown(results["explanation_posts_bar"])

            with col2:
                st.plotly_chart(results["fig_impressions"], use_container_width=True)
                st.markdown(results["explanation_impressions"])

            # Disposition en deux colonnes : Interactions et Taux d'Engagement
            col3, col4 = st.columns(2)
            with col3:
                st.plotly_chart(results["fig_interactions"], use_container_width=True)
                st.markdown(results["explanation_interactions"])

            with col4:
                st.plotly_chart(results["fig_engagement"], use_container_width=True)
                st.markdown(results["explanation_engagement"])

        with tab_advanced:
            st.header("Analyses Avancées")

            # Affichage de la Matrice de Corrélation en grand
            st.subheader("Matrice de Corrélation")
            st.plotly_chart(results["fig_corr_matrix"], use_container_width=True)
            st.markdown(results["explanation_corr_matrix"])

            # Disposition en deux colonnes pour les corrélations supplémentaires
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(results["fig_corr_inter_impr"], use_container_width=True)
                st.markdown(results["explanation_corr_inter_impr"])

            with col2:
                st.plotly_chart(results["fig_corr_posts_engagement"], use_container_width=True)
                st.markdown(results["explanation_corr_posts_engagement"])

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

            # Section : Téléchargement des Données et Exportation PDF
            st.header("Téléchargement des Données et Exportation PDF")

            # Téléchargement des Données
            st.subheader("Télécharger les Données Analytiques")
            csv = convert_df(results["combined_df"])
            st.download_button(
                label="Télécharger les Données Analytiques",
                data=csv,
                file_name='analyse_linkedin.csv',
                mime='text/csv',
            )

            # Exportation des Graphiques en PDF
            st.subheader("Exporter les Graphiques en PDF")

            # Collecter toutes les figures et leurs titres
            all_figures = [
                (results["fig_engagement"], "Taux d'Engagement au Fil du Temps"),
                (results["fig_impressions"], "Impressions au Fil du Temps"),
                (results["fig_interactions"], "Interactions au Fil du Temps"),
                (results["fig_subscribers"], "Abonnés Cumulés au Fil du Temps"),
                (results["fig_posts_bar"], "Nombre de Posts par Jour"),
                (results["fig_corr_abonnes_engagement"], "Corrélation entre Abonnés Cumulés et Taux d'Engagement"),
                (results["fig_growth_peaks"], "Analyse des Pics de Croissance des Abonnés"),
                (results["fig_corr_matrix"], "Matrice de Corrélation"),
                (results["fig_corr_inter_impr"], "Corrélation entre Impressions et Interactions"),
                (results["fig_corr_posts_engagement"], "Corrélation entre Nombre de Posts et Taux d'Engagement"),
                (results["fig_regression"], "Régression Linéaire pour Prédire le Taux d'Engagement"),
            ]

            # Ajouter les graphiques démographiques
            for category, fig in demographics_figures.items():
                all_figures.append((fig, f"Distribution de {category}"))

            # Fonction pour générer le PDF
            def generate_pdf(all_figures):
                figures_images = []
                titles = []
                for fig, title in all_figures:
                    img_bytes = save_fig_as_image(fig)
                    figures_images.append(img_bytes)
                    titles.append(title)
                pdf_bytes = create_pdf(figures_images, titles)
                return pdf_bytes

            # Générer le PDF
            pdf_bytes = generate_pdf(all_figures)

            # Bouton de téléchargement pour le PDF groupé
            st.download_button(
                label="Télécharger Tous les Graphiques en PDF",
                data=pdf_bytes,
                file_name='analyse_linkedin.pdf',
                mime='application/pdf',
            )

            # Bouton de téléchargement individuel
            st.markdown("### Télécharger les Graphiques Individuellement")
            for fig, title in all_figures:
                img_bytes = save_fig_as_image(fig)
                # Encode l'image en base64 pour le téléchargement
                b64 = base64.b64encode(img_bytes).decode()
                href = f'<a href="data:image/png;base64,{b64}" download="{title}.png">Télécharger {title}.png</a>'
                st.markdown(href, unsafe_allow_html=True)

    else:
        st.error("Erreur dans la génération des graphiques. Veuillez vérifier vos données.")
else:
    st.info("Veuillez télécharger un fichier Excel pour commencer l'analyse.")
