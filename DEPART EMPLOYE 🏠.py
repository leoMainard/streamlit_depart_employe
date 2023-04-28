import streamlit as st
from PIL import Image

st.set_page_config(layout='centered')

st.title('Tu pars ?')

st.info('Objectif : Identifier les employés à risque de partir.')

image_depart = Image.open('./images/_depart.jpg')

st.image(image_depart)