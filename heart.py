import streamlit as st
import pickle
import pandas as pd
import numpy as np

def app():
    pickle_in = open('model1.pkl', 'rb')
    classifier = pickle.load(pickle_in)
    st.title('Heart Disease Prediction Model')
    st.write('This Heart Disease Prediction Model uses patient medical information to diagnose Heart Disease. Fill out the form below: ')

    name = st.text_input("Name:")
    age = st.slider('Age: ', 1, 100)
    sex = st.selectbox("Sex: ", ['Male', 'Female'])
    sex = 0 if sex == 'Male' else 1;
    cp = st.selectbox("Chest Pain Type (1 = Typical Angina, 2 = Atypical Angina, 3 = Non—anginal Pain, 4 = Asymptotic) : ",["1","2","3","4"])
    trestbps = st.slider('Resting Blood Pressure (In mm/Hg unit)', 0, 200, 110)
    chol = st.slider('Serum Cholestoral (In mg/dl unit)', 0, 600, 115)
    fbs = st.selectbox("Fasting Blod Sugar", ["1","0"])
    restecg = st.selectbox("Resting Electrocardiographic results", ["1","0"])
    thalach = st.slider('Max Heart Rate', 0, 220, 115)
    exang = st.selectbox("Excercise Induced Angina (1=Yes, 0=No)",["1","0"])
    oldpeak = float(st.slider('Oldpeak (ST depression induced by excercise relative to rest)', 0.0, 10.0, 2.0))
    slope = st.selectbox("The Slope of the Peak Excercise ST Segment (Select 0, 1 or 2)",["0","1","2"])
    ca = st.selectbox("(Select 0, 1, 2 or 3)",["0","1","2","3"])
    thal = st.slider('3 = normal; 6 = fixed defect; 7 = reversable defect', 0, 10, 3)

    features = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]

    prediction_proba = classifier.predict_proba(features)


    submit = st.button('Predict')
    if submit:
            prediction = classifier.predict(features)
            if prediction == 0:
                st.success('Congratulations, you have a lower risk of Heart Disease')
                st.info('CAUTION: This is a prediction. Consult a doctor for further information if symptoms persist')
            else:
                st.error('Unfortunatly you are at risk of Heart Disease, consult your Doctor immediately' )
                st.info('CAUTION: This is a prediction. Consult a doctor for further information if symptoms persist')
