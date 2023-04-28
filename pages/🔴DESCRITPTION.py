import streamlit as st
from PIL import Image
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# shapash pour l'interprétabilité des résultats

# ----------------------------------------------------------------------- FONCTIONS
def resumetable(df):
    st.write("format de la base : {}".format(df.shape))
    summary = pd.DataFrame(df.dtypes,columns=["dtypes"]).reset_index() #init
    summary["Name"]=summary["index"]
    summary=summary[["Name","dtypes"]]
    summary["Missing"]=df.isnull().sum().values #nb val manquantes
    summary["Miss_percent"]=round((summary["Missing"]*100)/df.shape[0],0)
    summary["Uniques"]=df.nunique().values #Nb moda unique
    summary["First Value"]=df.iloc[0].values
    summary["Second Value"]= df.iloc[1].values
    summary["Third Value"]= df.iloc[2].values
    return summary
















# Configuration de la page en étendue
st.set_page_config(layout='wide')



# Lecture des données
data = pd.read_csv(r'./depart_employes.csv', sep=";")

# ----------------------------------------------------------------------- DESCRIPTION DES DONNEES
st.title('Description des données')
st.info('Objectif : comprendre le jeu de données.')



st.markdown('##### A travers notre fichier “depart_employes.csv” nous allons essayer d’expliquer et de prédire la variable “départ”.')
st.dataframe(resumetable(data))
st.write("On a donc 9 variables explicatives et 1 variable à expliquer : depart. Cette variable prendre la valeur 0 si "
         "l'employé a quitté l'entreprise, 0 si ce n'est pas le cas")



st.markdown("##### Etat des lieux")
st.markdown("Nous pouvons constater que :")
st.markdown("* Le nombre d'entrées est limité (14999 lignes)")
st.markdown("* Les variables sont des nombres décimaux, des entiers et des catégories")
st.markdown("* La description de data montre aussi qu'il n'y a à priori pas d'outliers")
st.markdown("A priori il n'y a donc pas de modifications importantes à réaliser avant de commencer à travailler.")


st.markdown('***')

st.write("Nous aurions aussi pu afficher le graphique suivant pour visualiser notre jeu de données en entier, et visualiser "
         "les valeurs manquantes. Elles seraient affichées par une ligne blanche.")
fig = plt.figure(figsize=(10, 4))
sns.heatmap(data.isna(), cbar=False)
st.pyplot(fig)


st.markdown('***')
st.markdown("##### Quelle est la proportion de départ de maintient dans nos données ?")
proportion = data['depart'].value_counts(normalize=True)
prop1 , prop2 = st.columns(2)
prop1.metric('Employés qui restent', f"{round(proportion[0],2)}%")
prop2.metric('Employés qui partent', f"{round(proportion[1],2)}%")


