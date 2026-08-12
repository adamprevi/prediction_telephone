# Fonction de prédiction
import gradio as gr
import joblib as jb
import pandas as pd
import numpy as np
# importer les encodeurs prix	adresse	marque	dim_ecr	ram	stockage	etat
encoders = jb.load('encoders_doc3.joblib')
# importer les valeurs uniques
uniques = jb.load('uniques_doc3.joblib')
# importer le normaliseur
scaler = jb.load('scaler_doc3.joblib')
# importer le modèle
xgb = jb.load('xgb_model_doc3.joblib')

# noms des classes
clasnames = uniques[2]
# fonction de prédiction simple
def Pred_func(prix, adresse, marque, dim_ecr, ram, stockage):
  # Encoder les valeurs des Fuel_Type, Seller_Type et Transmission

  # Encoder les valeurs des Fuel_Type, Seller_Type et Transmission
  adresse= encoders[0].transform([adresse])[0]
  marque = encoders[1].transform([marque])[0]


  # vecteur des valeurs numériques
  x_new = np.array([prix, adresse, marque, dim_ecr, ram, stockage])
  x_new = x_new.reshape(1,-1) # convert en un 2D array
  # Normaliser les données
  x_new = scaler.transform(x_new)
  # Prédire
  y_pred = xgb.predict(x_new)
  return clasnames[y_pred[0]] 

# Fonction de prédiction multiple
def Pred_func_csv(file):
  # Lire le fichier csv
  df = pd.read_csv(file)
  predictions = []
  # Boucle sur les lignes du dataframe
  for row in df.iloc[:, :].values:
    # prédiction simple
    y_pred = Pred_func(row[0], row[1], row[2], row[3], row[4], row[5])
    predictions.append(y_pred)

  df['etat'] = predictions
  df.to_csv('predictions.csv', index = False)
  return 'predictions.csv'

# définir les blocks
demo = gr.Blocks(theme='shivi/calm_seafoam')
# Créer les inputs
inputs = [gr.Number(label='prix'),
          gr.Dropdown(choices=uniques[0], label='adresse'),
          gr.Dropdown(choices=uniques[1], label='marque'),
          gr.Number(label='dim_ecr'),
          gr.Number(label='ram'),
          gr.Number(label='stockage')]
# Créer les outputs
outputs = gr.Textbox(label='Client_va_emprunter')
# Créer l'interface 1
interface1 = gr.Interface(fn = Pred_func,
                         inputs = inputs,
                         outputs = outputs,
                         title="Prédire l'état d'un portable avec une entrée",
                         description = """Ce modèle de machine permet de prédire l'état d'un portable en partant
                        du prix, l'adresse, la marque, la dimension de l'écran, le nombre de ram et le stockage.
                         """)
# Créer l'interface 2
interface2 = gr.Interface(fn = Pred_func_csv,
                         inputs = gr.File(label='Importer un fichier csv'),
                          outputs = gr.File(label='Télécharger un fichier csv'),
                         title="Prédire l'état d'un portable avec plusieurs entrées",
                         description = """Ce modèle de machine permet de prédire l'état d'un portable en partant
                        du prix, l'adresse, la marque, la dimension de l'écran, le nombre de ram et le stockage.
                         """)

# faire un tabbing des interfaces
with demo:
  gr.TabbedInterface([interface1, interface2], ['Simple Prediction', 'Prédiction multiple'])

# lancer l'interface
demo.launch(share= True)

