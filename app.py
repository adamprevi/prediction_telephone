import gradio as gr
import joblib as jb
import pandas as pd
import numpy as np

# =========================
# Chargement des fichiers
# =========================

encoders = jb.load("encoders_telephone.joblib")
uniques = jb.load("uniques_telephone.joblib")
scaler = jb.load("scaler_telephone.joblib")
xgb = jb.load("xgb_model_telephone.joblib")

# Noms des classes
clasnames = uniques[2]


# =========================
# Prédiction simple
# =========================

def Pred_func(prix, adresse, marque, dim_ecr, ram, stockage):

    # Encodage des variables catégorielles
    adresse = encoders[0].transform([adresse])[0]
    marque = encoders[1].transform([marque])[0]

    # Création du vecteur
    x_new = np.array([
        prix,
        adresse,
        marque,
        dim_ecr,
        ram,
        stockage
    ])

    # Mise en forme
    x_new = x_new.reshape(1, -1)

    # Normalisation
    x_new = scaler.transform(x_new)

    # Prédiction
    y_pred = xgb.predict(x_new)

    return clasnames[y_pred[0]]


# =========================
# Prédiction CSV
# =========================

def Pred_func_csv(file):

    df = pd.read_csv(file)

    predictions = []

    for row in df.iloc[:, :].values:

        y_pred = Pred_func(
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[5]
        )

        predictions.append(y_pred)

    df["etat"] = predictions

    output_file = "predictions.csv"
    df.to_csv(output_file, index=False)

    return output_file


# =========================
# Interface 1
# =========================

interface1 = gr.Interface(
    fn=Pred_func,
    inputs=[
        gr.Number(label="Prix"),
        gr.Dropdown(
            choices=uniques[0],
            label="Adresse"
        ),
        gr.Dropdown(
            choices=uniques[1],
            label="Marque"
        ),
        gr.Number(label="Dimension écran"),
        gr.Number(label="RAM"),
        gr.Number(label="Stockage")
    ],
    outputs=gr.Textbox(label="État du portable"),
    title="Prédire l'état d'un portable avec une entrée",
    description=(
        "Ce modèle permet de prédire l'état d'un portable "
        "à partir du prix, de l'adresse, de la marque, "
        "de la dimension de l'écran, de la RAM et du stockage."
    )
)


# =========================
# Interface 2
# =========================

interface2 = gr.Interface(
    fn=Pred_func_csv,
    inputs=gr.File(
        label="Importer un fichier CSV",
        type="filepath"
    ),
    outputs=gr.File(
        label="Télécharger le fichier CSV"
    ),
    title="Prédire l'état d'un portable avec plusieurs entrées",
    description=(
        "Importez un fichier CSV contenant le prix, l'adresse, "
        "la marque, la dimension de l'écran, la RAM et le stockage."
    )
)


# =========================
# Interface avec onglets
# =========================

demo = gr.TabbedInterface(
    [interface1, interface2],
    ["Prédiction simple", "Prédiction multiple"]
)


# =========================
# Lancement
# =========================

demo.launch()
