# ============================================================
# APPLICATION DE PREDICTION - TELEPHONE PORTABLE
# ============================================================

import gradio as gr
import joblib as jb
import pandas as pd
import numpy as np


# ============================================================
# 1. CHARGEMENT DES FICHIERS
# ============================================================

# Encodeurs
encoders = jb.load("encoders_doc3.joblib")

# Valeurs uniques
uniques = jb.load("uniques_doc3.joblib")

# Normaliseur
scaler = jb.load("scaler_doc3.joblib")

# Modèle XGBoost
xgb = jb.load("xgb_model_doc3.joblib")

# Noms des classes de la variable cible
classnames = uniques[2]


# ============================================================
# 2. FONCTION DE PREDICTION SIMPLE
# ============================================================

def Pred_func(prix, adresse, marque, dim_ecr, ram, stockage):

    # Vérification des valeurs
    if prix is None:
        return "Veuillez saisir le prix."

    if adresse is None:
        return "Veuillez sélectionner une adresse."

    if marque is None:
        return "Veuillez sélectionner une marque."

    if dim_ecr is None:
        return "Veuillez saisir la dimension de l'écran."

    if ram is None:
        return "Veuillez saisir la RAM."

    if stockage is None:
        return "Veuillez saisir le stockage."

    # --------------------------------------------------------
    # Encodage des variables catégorielles
    # --------------------------------------------------------

    adresse_encoded = encoders[0].transform([adresse])[0]

    marque_encoded = encoders[1].transform([marque])[0]

    # --------------------------------------------------------
    # Création du vecteur
    # Ordre :
    # prix
    # adresse
    # marque
    # dim_ecr
    # ram
    # stockage
    # --------------------------------------------------------

    x_new = np.array([
        prix,
        adresse_encoded,
        marque_encoded,
        dim_ecr,
        ram,
        stockage
    ], dtype=float)

    # Transformation en tableau 2D
    x_new = x_new.reshape(1, -1)

    # --------------------------------------------------------
    # Normalisation
    # --------------------------------------------------------

    x_new = scaler.transform(x_new)

    # --------------------------------------------------------
    # Prédiction
    # --------------------------------------------------------

    y_pred = xgb.predict(x_new)

    # Classe prédite
    prediction = classnames[y_pred[0]]

    return prediction


# ============================================================
# 3. FONCTION DE PREDICTION MULTIPLE
# ============================================================

def Pred_func_csv(file):

    if file is None:
        return None

    # --------------------------------------------------------
    # Lire le fichier CSV
    # --------------------------------------------------------

    try:
        df = pd.read_csv(file.name)
    except Exception as e:
        raise gr.Error(f"Impossible de lire le fichier CSV : {e}")

    # --------------------------------------------------------
    # Vérifier le nombre de colonnes
    # --------------------------------------------------------

    if df.shape[1] < 6:
        raise gr.Error(
            "Le fichier CSV doit contenir au minimum 6 colonnes : "
            "prix, adresse, marque, dim_ecr, ram, stockage."
        )

    predictions = []

    # --------------------------------------------------------
    # Parcourir les lignes
    # --------------------------------------------------------

    for i, row in df.iterrows():

        try:

            prediction = Pred_func(
                row.iloc[0],  # prix
                row.iloc[1],  # adresse
                row.iloc[2],  # marque
                row.iloc[3],  # dim_ecr
                row.iloc[4],  # ram
                row.iloc[5]   # stockage
            )

            predictions.append(prediction)

        except Exception as e:

            raise gr.Error(
                f"Erreur à la ligne {i + 1} du fichier CSV : {e}"
            )

    # --------------------------------------------------------
    # Ajouter la colonne de prédiction
    # --------------------------------------------------------

    df["etat"] = predictions

    # --------------------------------------------------------
    # Sauvegarder le résultat
    # --------------------------------------------------------

    output_file = "predictions.csv"

    df.to_csv(
        output_file,
        index=False
    )

    return output_file


# ============================================================
# 4. INPUTS
# ============================================================

inputs = [

    gr.Number(
        label="Prix"
    ),

    gr.Dropdown(
        choices=uniques[0],
        label="Adresse"
    ),

    gr.Dropdown(
        choices=uniques[1],
        label="Marque"
    ),

    gr.Number(
        label="Dimension écran"
    ),

    gr.Number(
        label="RAM"
    ),

    gr.Number(
        label="Stockage"
    )
]


# ============================================================
# 5. OUTPUT
# ============================================================

outputs = gr.Textbox(
    label="Client va emprunter"
)


# ============================================================
# 6. INTERFACE 1 : PREDICTION SIMPLE
# ============================================================

interface1 = gr.Interface(

    fn=Pred_func,

    inputs=inputs,

    outputs=outputs,

    title="Prédire l'état d'un portable avec une entrée",

    description="""
    Ce modèle permet de prédire l'état d'un portable
    à partir du prix, de l'adresse, de la marque,
    de la dimension de l'écran, de la RAM et du stockage.
    """
)


# ============================================================
# 7. INTERFACE 2 : PREDICTION MULTIPLE
# ============================================================

interface2 = gr.Interface(

    fn=Pred_func_csv,

    inputs=gr.File(
        label="Importer un fichier CSV"
    ),

    outputs=gr.File(
        label="Télécharger le fichier CSV"
    ),

    title="Prédire l'état d'un portable avec plusieurs entrées",

    description="""
    Importez un fichier CSV contenant les caractéristiques
    des portables afin d'obtenir automatiquement les prédictions.
    """
)


# ============================================================
# 8. CREATION DES ONGLETS
# ============================================================

demo = gr.TabbedInterface(

    [interface1, interface2],

    [
        "Simple Prediction",
        "Prédiction multiple"
    ]
)


# ============================================================
# 9. LANCEMENT
# ============================================================

demo.launch()
