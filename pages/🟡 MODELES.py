import streamlit as st
import pickle
from PIL import Image
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# shapash pour l'interprétabilité des résultats


st.set_page_config(layout='wide')

# Chargement des informations de chaque modèle
with open("./modeles/modeles_info", "rb") as fp:  # Unpickling
    infos_modeles = pickle.load(fp)

model_names =["KNN","Régression Logistique","SVG","Arbre de décision","Random Forest"]

def couleur_f1score(val):
    '''
    Permet de modifier la couleur de la colonne "f1-score" d'une table streamlit en appliquant cette fonction à la colonne "f1-score"
    '''
    color = '#e63946' if val<0.5 else '#ffb703' if val<0.75 else '#57cc99'
    return f'background-color: {color}'

def couleur_prediction(s):
    '''
    Permet de modifier la couleur d'une ligne si la prédiction est bonne ou mauvaise
    '''
    return ['background-color: #57cc99']*len(s) if s.target == s.Prédiction else ['background-color: #e63946']*len(s)

# data = pd.read_csv(r'./../depart_employes.csv', sep=";")

# ----------------------------------------------------------------------- ANALYSE DES MODELES
st.title('Quel modèle choisir ?')

filtre_model = st.selectbox("Sélectionner un modèle à analyser.", model_names)

# Sélectionner l'index correspondant au nom du modèle pour afficher les infos de infos_modeles[index]
model_info = infos_modeles[model_names.index(filtre_model)]

matrice_confusion = model_info[3]
rapport = pd.DataFrame(model_info[4]).transpose()
accuracy = rapport.loc["accuracy", "support"]

model_importance = pd.DataFrame(model_info[5])
model_importance.rename(columns={0: 'Importance'}, inplace=True)


col1,col2,col3,col4 = st.columns([1,2,2,2])

# ------------------------------------------------------ Première colonne : accuracy du modèle
col1.metric("Accuracy", f"{round(accuracy,2)} %")

# ------------------------------------------------------ Deuxième colonne : rapport de classification
col2.markdown("##### Rapport de classification")
col2.table(rapport.style.applymap(couleur_f1score, subset=['f1-score']))

# ------------------------------------------------------ Troisième colonne : matrice de confusion
fig_confusion = plt.figure()
sns.heatmap(matrice_confusion, annot=True, cmap="Blues", fmt='d')
plt.legend()

col3.markdown("##### Matrice de confusion")
col3.pyplot(fig_confusion)

# ------------------------------------------------------ Quatrième colonne : importance des variables
col4.markdown("##### Importance des variables")
col4.table(model_importance)


# ------------------------------------------------------ Exemple de prédictions
predictions = pd.DataFrame(model_info[1]).reset_index().drop('index', axis=1)
predictions.rename(columns={0: 'Prédiction'}, inplace=True)
val_reelle = pd.DataFrame(model_info[2]).reset_index().drop('index', axis=1)

df_predictions = pd.concat([val_reelle,predictions], axis=1)

st.markdown("##### Exemple de prédictions")
st.table(df_predictions.sample(20).style.apply(couleur_prediction, axis=1))