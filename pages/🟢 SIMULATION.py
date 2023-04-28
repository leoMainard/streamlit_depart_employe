import streamlit as st
from PIL import Image
import pickle
import numpy as np
import sklearn
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
# shapash pour l'interprétabilité des résultats


st.set_page_config(layout='wide')
model_names =["KNN","Régression Logistique","SVG","Arbre de décision","Random Forest"]

# Chargement de chaque modèle
modeles = pickle.load(open("./modeles/modele.sav", 'rb'))

# Chargement du pipeline
pipeline = joblib.load(f'./modeles/pipeline.pkl')


def prediction(model, pipeline, satisfaction,derniere_eval,nb_projets,nb_heures_mensu,temps_entreprise,accident_travail,promotion,service,niveau_salaire):
    '''
    Permet de tester si l'employé part ou non en fonction des valeurs des variables
    '''
    accident_travail = 1 if accident_travail else 0
    promotion = 1 if promotion else 0

    num_cols = ['Satisfaction','derniere_evaluation', 'Nombre_de_projets', 'Nombre_heures_mensuelles_moyenne',
                'Temps_passe_dans_entreprise','Accident_du travail', 'promotion_5_dernieres_annees']

    cat_cols = ['Service', 'niveau_salaire']


    # Préparation de la données
    data = pd.DataFrame([[satisfaction, derniere_eval, nb_projets, nb_heures_mensu, temps_entreprise, accident_travail,
                       promotion, service, niveau_salaire]],
                     columns=['Satisfaction','derniere_evaluation','Nombre_de_projets','Nombre_heures_mensuelles_moyenne','Temps_passe_dans_entreprise'
                         ,'Accident_du travail','promotion_5_dernieres_annees','Service','niveau_salaire'])

    df = pipeline['MinMaxScaler'].transform(data[num_cols])
    df = pd.DataFrame(df, columns=list(data[num_cols].columns))
    df[cat_cols] = data[cat_cols]

    df[num_cols] = pipeline['StandardScaler'].transform(df[num_cols])
    df['Service'] = pipeline['encoder_Service'].transform(df['Service'])
    df['niveau_salaire'] = pipeline['encoder_niveau_salaire'].transform(df['niveau_salaire'])

    # Classification
    return model.predict(df)

# ----------------------------------------------------------------------- ANALYSE DES DONNEES
st.title('Simulation de départ')

filtre_model = st.selectbox("Sélectionner un modèle à analyser.", model_names)

with st.expander("Est-ce que mon employé part si ..."):
    satisfaction = st.slider("Satisfaction", 0, 100, 60)

    derniere_eval = st.slider("Dernière évaluation", 0, 100, 70)

    nb_projets = st.slider("Nombre de projets", 0, 10, 4)

    nb_heures_mensu = st.slider("Nombre d'heures mensuelles moyenne", 0, 400, 200)

    temps_entreprise = st.slider("Temps passé dans l'entreprise", 0, 30, 3)

    accident_travail = st.checkbox('Accident du travail')

    promotion = st.checkbox('Promotion ces 5 dernières années')

    service = st.selectbox(
        'Service',['sales','accounting','hr','technical','support','management','IT','product_mng','marketing','RandD'])

    niveau_salaire = st.selectbox(
        'Niveau de salaire',
        ('low', 'medium', 'high'))

model_choisi = modeles[model_names.index(filtre_model)]

prediction = prediction(model_choisi, pipeline, satisfaction,derniere_eval,nb_projets,nb_heures_mensu,temps_entreprise,accident_travail,promotion,service,niveau_salaire)


if(prediction == 0):
    image_depart = Image.open('./images/reste.jpg')

    st.subheader("Ca reste en place :)")
    st.image(image_depart)
else:
    image_depart = Image.open('./images/ciao.jpg')

    st.subheader("C'est ciao l'employé !")
    st.image(image_depart)