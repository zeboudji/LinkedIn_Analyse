import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, Frame, Button, Label, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates

# Fonction pour générer les graphiques de performance
def generate_performance_graphs(excel_file_path, parent_frame):
    try:
        # Charger le fichier Excel
        xls = pd.ExcelFile(excel_file_path)

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

        # Créer une figure avec plusieurs sous-graphiques
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        fig.suptitle('Performance des Réseaux Sociaux', fontsize=16)

        # Graphique 1 : Nombre de posts par jour
        axs[0].bar(combined_df['Date'], combined_df['Posts per Day'], color='purple')
        axs[0].set_title('Nombre de Posts par Jour')
        axs[0].set_ylabel('Posts')
        axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
        axs[0].tick_params(axis='x', rotation=45)

        # Graphique 2 : Impressions au fil du temps
        axs[1].plot(combined_df['Date'], combined_df['Impressions'], marker='o', color='blue')
        axs[1].set_title('Impressions au Fil du Temps')
        axs[1].set_ylabel('Impressions')
        axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
        axs[1].tick_params(axis='x', rotation=45)

        # Graphique 3 : Interactions au fil du temps
        axs[2].plot(combined_df['Date'], combined_df['Interactions'], marker='x', color='orange')
        axs[2].set_title('Interactions au Fil du Temps')
        axs[2].set_ylabel('Interactions')
        axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
        axs[2].tick_params(axis='x', rotation=45)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Intégrer le graphique dans Tkinter
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

        # Deuxième figure pour le taux d'engagement et abonnés cumulés
        fig2, axs2 = plt.subplots(2, 1, figsize=(10, 8))
        fig2.suptitle('Engagement et Abonnés', fontsize=16)

        # Taux d'engagement
        axs2[0].plot(combined_df['Date'], combined_df['Engagement Rate (%)'], marker='o', color='blue')
        axs2[0].set_title('Taux d\'Engagement au Fil du Temps')
        axs2[0].set_ylabel('Taux d\'Engagement (%)')
        axs2[0].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
        axs2[0].tick_params(axis='x', rotation=45)
        axs2[0].grid(True)

        # Abonnés cumulés
        axs2[1].plot(combined_df['Date'], combined_df['Cumulative Subscribers'], marker='o', color='green')
        axs2[1].set_title('Abonnés Cumulés au Fil du Temps')
        axs2[1].set_xlabel('Date')
        axs2[1].set_ylabel('Abonnés Cumulés')
        axs2[1].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
        axs2[1].tick_params(axis='x', rotation=45)
        axs2[1].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Intégrer le deuxième graphique dans Tkinter
        canvas2 = FigureCanvasTkAgg(fig2, master=parent_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(side='top', fill='both', expand=True)

    except Exception as e:
        messagebox.showerror("Erreur", f"Une erreur est survenue : {e}")

# Fonction pour ouvrir la boîte de dialogue de sélection de fichier
def select_file(root, display_frame):
    file_path = filedialog.askopenfilename(
        title="Sélectionner un fichier Excel",
        filetypes=[("Fichiers Excel", "*.xlsx *.xls")]
    )
    if file_path:
        # Effacer les graphiques précédents
        for widget in display_frame.winfo_children():
            widget.destroy()
        # Générer les graphiques
        generate_performance_graphs(file_path, display_frame)
        # Mettre à jour le label avec le chemin du fichier
        file_label.config(text=f"Fichier sélectionné : {file_path}")

# Création de la fenêtre principale
root = Tk()
root.title("Analyse des Performances Réseaux Sociaux")
root.geometry("1200x800")
root.configure(bg="#f0f0f0")

# Cadre en haut pour les contrôles
top_frame = Frame(root, bg="#f0f0f0", pady=10)
top_frame.pack(side='top', fill='x')

# Bouton pour sélectionner le fichier
select_button = Button(top_frame, text="Sélectionner un fichier Excel", command=lambda: select_file(root, display_frame),
                       bg="#4CAF50", fg="white", font=("Arial", 12), padx=10, pady=5, borderwidth=0, cursor="hand2")
select_button.pack(side='left', padx=20)

# Label pour afficher le chemin du fichier sélectionné
file_label = Label(top_frame, text="Aucun fichier sélectionné", bg="#f0f0f0", fg="#333", font=("Arial", 10))
file_label.pack(side='left', padx=10)

# Cadre principal pour afficher les graphiques avec un scroll
from tkinter import Canvas, Scrollbar

canvas_main = Canvas(root, bg="#f0f0f0")
scrollbar = Scrollbar(root, orient="vertical", command=canvas_main.yview)
scrollable_frame = Frame(canvas_main, bg="#f0f0f0")

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas_main.configure(
        scrollregion=canvas_main.bbox("all")
    )
)

canvas_main.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas_main.configure(yscrollcommand=scrollbar.set)

canvas_main.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# Cadre pour les graphiques
display_frame = Frame(scrollable_frame, bg="#f0f0f0")
display_frame.pack(pady=20, padx=20)

# Lancer l'application
root.mainloop()
