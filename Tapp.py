import streamlit as st
import joblib
import numpy as np

model = joblib.load('titanic_survival_model.pkl')

st.title('Titanic Survival Predictor 🚢')
st.write('Enter passenger details to predict survival chances')

pclass = st.selectbox('Passenger Class', [1, 2, 3])
sex = st.selectbox('Sex', ['Male', 'Female'])
age = st.number_input('Age', min_value=0, max_value=100, value=25)
sibsp = st.number_input('Siblings/Spouses aboard', min_value=0, max_value=10, value=0)
parch = st.number_input('Parents/Children aboard', min_value=0, max_value=10, value=0)
fare = st.number_input('Fare paid ($)', min_value=0.0, value=30.0)
embarked = st.selectbox('Port Embarked', ['Southampton', 'Cherbourg', 'Queenstown'])
title = st.selectbox('Title', ['Mr', 'Mrs', 'Miss', 'Master', 'Rare'])
has_cabin = st.selectbox('Had a recorded cabin?', ['No', 'Yes'])

sex_encoded = 0 if sex == 'Male' else 1
embarked_map = {'Southampton': 0, 'Cherbourg': 1, 'Queenstown': 2}
embarked_encoded = embarked_map[embarked]
title_map = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
title_encoded = title_map[title]
cabin_encoded = 1 if has_cabin == 'Yes' else 0

if st.button('Predict'):
    input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, 
                             embarked_encoded, title_encoded, cabin_encoded]])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.success(f'✅ This passenger would likely SURVIVE')
        st.write(f'Confidence: {probability[1]:.2%}')
    else:
        st.error(f'❌ This passenger would likely NOT survive')
        st.write(f'Confidence: {probability[0]:.2%}')
