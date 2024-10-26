import plotly.express as px
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Function to convert a DataFrame to a CSV for download
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# Function to clean the dataframes
def clean_dataframe(df):
    df.fillna(0, inplace=True)
    if 'Interactions' in df.columns:
        df['Interactions'] = pd.to_numeric(df['Interactions'], errors='coerce').fillna(0)
    if 'Impressions' in df.columns:
        df['Impressions'] = pd.to_numeric(df['Impressions'], errors='coerce').fillna(0)
    if 'Engagement Rate (%)' in df.columns:
        df['Engagement Rate (%)'] = pd.to_numeric(df['Engagement Rate (%)'], errors='coerce').fillna(0)
    return df

# Function to generate the performance graphs
def generate_performance_graphs(excel_data):
    try:
        # Load the Excel file
        xls = pd.ExcelFile(excel_data)

        # Load relevant sheets
        engagement_df = pd.read_excel(xls, 'ENGAGEMENT')
        abonnes_df = pd.read_excel(xls, 'ABONNÉS', skiprows=2)
        meilleurs_posts_df = pd.read_excel(xls, 'MEILLEURS POSTS').iloc[2:, 1:3]

        # Clean the data
        engagement_df = clean_dataframe(engagement_df)
        abonnes_df = clean_dataframe(abonnes_df)

        # Rename and handle missing values
        engagement_df.columns = engagement_df.columns.str.strip()
        abonnes_df.columns = abonnes_df.columns.str.strip()
        meilleurs_posts_df.columns = ['Date de publication', 'Interactions']
        meilleurs_posts_df['Date de publication'] = pd.to_datetime(meilleurs_posts_df['Date de publication'], format='%d/%m/%Y', errors='coerce')
        meilleurs_posts_df['Interactions'] = pd.to_numeric(meilleurs_posts_df['Interactions'], errors='coerce')

        # Combine and prepare data
        engagement_df['Date'] = pd.to_datetime(engagement_df['Date'], format='%d/%m/%Y', errors='coerce')
        abonnes_df['Date'] = pd.to_datetime(abonnes_df['Date'], format='%d/%m/%Y', errors='coerce')
        abonnes_df['Nouveaux abonnés'] = pd.to_numeric(abonnes_df['Nouveaux abonnés'], errors='coerce').fillna(0)
        abonnes_df['Cumulative Subscribers'] = abonnes_df['Nouveaux abonnés'].cumsum()

        # Merge the dataframes for analysis
        combined_df = pd.merge(engagement_df, abonnes_df[['Date', 'Cumulative Subscribers']], on='Date', how='left')

        # Calculate engagement rate and ensure no NaN
        combined_df['Engagement Rate (%)'] = (combined_df['Interactions'] / combined_df['Impressions']).fillna(0) * 100

        # Clean remaining NaN values
        combined_df.fillna(0, inplace=True)

        # Create plots
        fig_engagement = px.line(combined_df, x='Date', y='Engagement Rate (%)', title="Taux d'Engagement", markers=True)
        fig_impressions = px.line(combined_df, x='Date', y='Impressions', title='Impressions au Fil du Temps', markers=True)
        fig_interactions = px.line(combined_df, x='Date', y='Interactions', title='Interactions au Fil du Temps', markers=True)

        # Posts per day bar chart
        posts_per_day = meilleurs_posts_df['Date de publication'].value_counts().sort_index()
        combined_df['Posts per Day'] = combined_df['Date'].map(posts_per_day).fillna(0)
        fig_posts_per_day = px.bar(combined_df, x='Date', y='Posts per Day', title='Nombre de Posts par Jour')

        # Subscribers growth rate
        abonnes_df['Growth Rate'] = abonnes_df['Nouveaux abonnés'].pct_change().fillna(0) * 100
        fig_growth_rate = px.line(abonnes_df, x='Date', y='Growth Rate', title="Taux de Croissance des Abonnés", markers=True)

        # Regression prediction
        def regression_engagement(combined_df):
            # Ensure no NaN values in the columns used for regression
            X = combined_df[['Impressions', 'Posts per Day', 'Cumulative Subscribers']].fillna(0)
            y = combined_df['Engagement Rate (%)'].fillna(0)

            model = LinearRegression()
            model.fit(X, y)
            predictions = model.predict(X)

            r2 = r2_score(y, predictions)
            fig = px.scatter(x=y, y=predictions, title=f'Prédiction du Taux d\'Engagement (R² = {r2:.2f})')
            fig.add_traces(px.line(x=y, y=y).data)
            return fig, r2

        fig_regression, r2_value = regression_engagement(combined_df)

        # Return the generated plots and cleaned dataframe
        return {
            "fig_engagement": fig_engagement,
            "fig_impressions": fig_impressions,
            "fig_interactions": fig_interactions,
            "fig_posts_per_day": fig_posts_per_day,
            "fig_growth_rate": fig_growth_rate,
            "fig_regression": fig_regression,
            "combined_df": combined_df
        }

    except Exception as e:
        st.error(f"Une erreur est survenue lors de la génération des graphiques : {e}")
        return None

# Streamlit app layout
st.set_page_config(page_title="Analyse des Performances LinkedIn", layout="wide")

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Sélectionnez un fichier Excel", type=["xlsx", "xls"])

if uploaded_file is not None:
    results = generate_performance_graphs(uploaded_file)

    if results:
        st.plotly_chart(results["fig_engagement"], use_container_width=True)
        st.plotly_chart(results["fig_impressions"], use_container_width=True)
        st.plotly_chart(results["fig_interactions"], use_container_width=True)
        st.plotly_chart(results["fig_posts_per_day"], use_container_width=True)
        st.plotly_chart(results["fig_growth_rate"], use_container_width=True)
        st.plotly_chart(results["fig_regression"], use_container_width=True)

        # CSV download option
        csv = convert_df(results["combined_df"])
        st.download_button(
            label="Télécharger les Données Analytiques",
            data=csv,
            file_name='analyse_linkedin.csv',
            mime='text/csv',
        )
else:
    st.info("Veuillez télécharger un fichier Excel pour commencer l'analyse.")
