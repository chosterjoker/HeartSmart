import streamlit as st
from multiapp import MultiApp
from apps import home, model, model2 # import your app modules here

app = MultiApp()

st.markdown(unsafe_allow_html=True, body="<span style='color:black; font-size: 60px'><strong><h4>HeartSmart:heart:</h4></strong></span>")

hide_menu_style = """
    <style>
    footer {visibility: hidden;}
    </style>"""

st.markdown(hide_menu_style, unsafe_allow_html=True)
# Add all your application here
app.add_app("Home", home.app)
app.add_app("Heart Disease Prediction", model.app)
app.add_app("Arrhythmia Prediction", model2.app)
# The main app
app.run()
