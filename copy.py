import streamlit as st
from multiapp import MultiApp
from apps import home, data, model, model2 # import your app modules here

app = MultiApp()

original_title = '<p style=" font-size: 60px; "><b>DIOC</b></p>'
st.markdown(original_title, unsafe_allow_html=True)

# Add all your application here
app.add_app("Home", home.app)
app.add_app("CVD Prediction", model.app)
app.add_app("Arrhythmia Prediction", model2.app)
app.add_app("Data", data.app)

# The main app
app.run()
