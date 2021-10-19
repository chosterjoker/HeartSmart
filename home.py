import streamlit as st

def app():
    st.title('Heart Disease Information')

    st.write('Heart Disease refers to the different heart conditions that disturb the flow and the pumping of blood in the heart, usually caused by fatty buildup in blood vessels. Below are the six main types of Heart Disease: ')

    st.markdown('**Congential Heart Disease (CHD)**')
    st.markdown('**Coronary Artery Disease (CAD)**')
    st.markdown('**Heart Arrhythmia**')
    st.markdown('**Dilated Cardiomyopathy**')
    st.markdown('**Heart Failure**')
    st.markdown('**Valve Disease**')

    st.title('Symptoms')
    st.write('- Heart Attack: Chest & neck pain, Vomiting or nausea, Fatigue, Shortness of breath, Dizziness')
    st.write('- Arrhythmia: Palpitations, Irregular heartbeating, Fluttering feelings in the chest')
    st.write('- Heart Failure: Fatigue, Shortness of breath, Swelling of feet, ankles, legs, abdomen, or neck')

    st.title('Statistics')
    st.write('- Ranked as the #1 cause of death')
    st.write('- Average of 200 deaths per 100,000 population')
    st.write('- Global Annual Deaths: 17.9 million')
    st.write('- 75% of deaths are in low and middle income countries')
    st.write('- Every 40 seconds, an American will experince a heart attack')

    st.title('Risk Factors')
    st.write('- High Blood Pressure')
    st.write('- Diabetes')
    st.write('- Obesity')
    st.write('- Smoking')
    st.write('- Genetics')
    st.write('- Physical Inactivity')

    st.title('Map')
    map = 'hd_all.jpg'
    st.image(map)

    st.title('Sources + More Information')
    st.write('CDC: https://www.cdc.gov/heartdisease/index.htm')
    st.write('Mayo Clinic: https://www.mayoclinic.org/diseases-conditions/heart-disease/symptoms-causes/syc-20353118 ')
    st.write('Million Hearts: https://millionhearts.hhs.gov/learn-prevent/risks.html')
    st.write('WebMD: https://www.webmd.com/heart-disease/default.htm')
