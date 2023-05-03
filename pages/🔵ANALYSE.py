import streamlit as st
from PIL import Image
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
# shapash pour l'interprétabilité des résultats


st.set_page_config(layout='wide')


data = pd.read_csv(r'./depart_employes.csv', sep=";")

# ----------------------------------------------------------------------- ANALYSE DES DONNEES
st.title('Analyse des données')
st.info('Objectif : analyser le jeu de données, la distribution, leurs relations.')



# La variable cible Y doit être de type binaire pour la Régréssion Logistique
data["target"] = data["depart"].astype('category')
# data.drop("depart", axis=1,inplace=True)
positive_data= data[data['target']==1]
negative_data= data[data['target']==0]
col=list(data.columns)[:-1]
num_cols=list(data.columns)[:-3]
cat_cols=list(data.columns)[-3:-1]


st.markdown("##### Observons nos données numériques : ")
filtre_numerique = st.selectbox("Sélectionnons une variable numérique.", ['Satisfaction','derniere_evaluation','Nombre_de_projets','Nombre_heures_mensuelles_moyenne','Temps_passe_dans_entreprise'])

# --------------------------------------------- Répartition des variables numériques
graph_num1, graph_num2 = st.columns(2)

fig_graphNum1 = plt.figure()
sns.distplot(positive_data[filtre_numerique], label='positive')
sns.distplot(negative_data[filtre_numerique], label='negative')
plt.legend()
graph_num1.pyplot(fig_graphNum1)


fig_graphNum2 = plt.figure()
sns.boxplot(data=data,y=filtre_numerique, x='target')
plt.legend()
graph_num2.pyplot(fig_graphNum2)

st.markdown('***')

st.markdown("##### Observons nos données catégorielles : ")
col1, col2 = st.columns(2)

# Sélectionner les variables catégorielles
cat_cols = ['Accident_du travail', 'promotion_5_dernieres_annees', 'Service', 'niveau_salaire']

filtre_categorique = col1.selectbox("Sélectionnez une variable catégorielle.", cat_cols)

fig_cat = plt.figure()
ax = sns.countplot(x=filtre_categorique, hue="depart", data=data, palette=["orange", "blue"])
ax.legend(title='depart', labels=['negative', 'positive'])
col1.pyplot(fig_cat)


col2.markdown("##### Nos données sont-elles corrélées ?")
# --------------------------------------------- Matrice de corrélation
fig_graphCorr = plt.figure()
sns.heatmap(data.corr(), annot=True, cmap="Blues")
plt.legend()
col2.pyplot(fig_graphCorr)

st.markdown('***')

st.markdown("##### Faites votre analyse")
choix1, choix2 = st.columns(2)

var1 = choix1.selectbox("Variable 1",data.columns)
var2 = choix2.selectbox("Variable 2",data.columns)


df_mean = data.groupby(var2)[var1].mean().reset_index(name='moyenne')
df_count = data.groupby(var2)[var1].count().reset_index(name='nombre')

fig1 = go.Figure()

# Ajouter la première ligne pour la moyenne
fig1.add_trace(
    go.Scatter(x=df_mean[var2], y=df_mean['moyenne'], name='Moyenne')
)

# Ajouter la deuxième ligne pour le nombre avec un deuxième axe y
fig1.add_trace(
    go.Scatter(x=df_count[var2], y=df_count['nombre'], name='Nombre', yaxis='y2')
)

# Configurer les axes et ajouter des légendes
fig1.update_layout(
    title=f"Moyenne et effectif de {var1} en fonction de {var2}",
    xaxis_title=var2,
    yaxis=dict(title=f"Moyenne de {var1}"),
    yaxis2=dict(title=f"Effectif de {var1}", overlaying='y', side='right'),
    legend=dict(x=0, y=1, traceorder="normal")
)

st.plotly_chart(fig1, theme="streamlit", use_container_width=True)



