import streamlit as st
import pickle
import pandas as pd
import numpy as np

def app():
    st.sidebar.error('Caution: This is a prediction. Consult a professional for more information!')
    pickle_in = open('model_real.pkl', 'rb')
    classifier = pickle.load(pickle_in)
    st.markdown(unsafe_allow_html=True, body="<span style='color:black; font-size: 42px'><strong><h4>Heart Disease Prediction Model :heartbeat:</h4></strong></span>")
    st.write('This Heart Disease Prediction Model uses patient medical information to diagnose Heart Disease. Refer to `Learn More` for more information')

    with st.expander('Learn More'):
        st.write('This model uses a AdaBoost Classifier to classify specific medical information as more or less likely to have Heart Disease. For data and more information about specific features go to https://archive.ics.uci.edu/ml/datasets/Heart+Disease. Current Accuracy =  90%.')
        st.write('If self-diagnosing, ask a Doctor for your medical information from recent checkups. Try to fill as many fields as possible for an accurate classification. Consult a professional for more information.')

    st.subheader('Fill out the form below: ')

    name = st.text_input("Name:")
    age = st.slider('Age: ', 1, 150)
    sex = st.selectbox("Sex: ", ['Male', 'Female'])
    sex = 0 if sex == 'Male' else 1;
    cp = st.selectbox("Chest Pain Type (0 = Typical Angina, 1 = Atypical Angina, 2 = Non—anginal Pain, 3 = Asymptotic) : ",["0","1","2","3"])
    trestbps = st.slider('Resting Blood Pressure (mm/Hg)', 0, 200, 110)
    chol = st.slider('Serum Cholestoral (mg/dl)', 0, 600, 115)
    fbs = st.selectbox("Fasting Blood Sugar (If above 120 mg/dl choose 1, if less choose 0)", ["1","0"])
    restecg = st.selectbox("Resting Electrocardiographic results (1 = Pass, 0 = Fail)", ["1","0"])
    thalach = st.slider('Max Heart Rate achievied', 0, 220, 115)
    exang = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)",["1","0"])
    oldpeak = float(st.slider('ST depression induced by excercise relative to rest', 0.0, 3.0))
    slope = st.selectbox("The Slope of the Peak Excercise ST Segment (Select 0, 1 or 2)",["0","1","2"])
    ca = st.selectbox("# of major vessels colored by Flourosopy (Select 0, 1, 2,  3, or 4)",["0","1","2","3","4"])
    thal = st.slider('Thalassemia (1 = normal; 2 = fixed defect; 3 = reversable defect)', 1, 3)

    features = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]

    prediction_proba = classifier.predict_proba(features)

    submit = st.button('Predict')
    if submit:
            prediction = classifier.predict(features)
            if prediction == 0:
                st.success('You have a low risk of Heart Disease!')

            else:
                st.error('Unfortunatly you are at risk of Heart Disease, consult your Doctor asap!' )
