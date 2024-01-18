import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('loanclass.pickle', 'rb') as file:
        model = pickle.load(file)
    return model

reg = load_model()

def show_predict_page():
    st.title("Loan Approval Prediction")

    property_areas = ["Rural", "Semiurban", "Urban"]
    property_area = st.selectbox("Select Property Area", property_areas)

    genders = ["Male", "Female"]
    gender = st.selectbox("Select Gender", genders)

    married_status = st.checkbox("Married")

    dependents = st.slider("Number of Dependents", 0, 3, 1)

    educations = ["Graduate", "Not Graduate"]
    education = st.selectbox("Education", educations)

    self_employed = st.checkbox("Self Employed")

    applicant_income = st.slider("Applicant Income", 150, 10000, 4000)

    coapplicant_income = st.slider("Coapplicant Income", 0, 6000, 1300)

    loan_amount = st.slider("Loan Amount", 9, 218, 100)

    loan_term = st.slider("Loan Term (Months)", 180, 480, 200)

    credit_history = st.checkbox("Credit History")

    ok = st.button("Predict Loan Approval")
    if ok:
        # Assuming Property_Area is the first feature and Credit_History is the last feature
        loc_index = property_areas.index(property_area)

        x = np.zeros(13)  # Assuming you have 13 features
        x[0] = 1 if gender == "Male" else 0
        x[1] = 1 if married_status else 0
        x[2] = dependents
        x[3] = 1 if education == "Graduate" else 0
        x[4] = 1 if self_employed else 0
        x[5] = applicant_income
        x[6] = coapplicant_income
        x[7] = loan_amount
        x[8] = loan_term
        x[9] = 1 if credit_history else 0

        if loc_index >= 0:
            x[loc_index + 9] = 1

        # Convert x to a 2D array for prediction
        x = x.reshape(1, -1)

        approval = reg.predict(x)[0]

        if approval == 1:
            st.subheader("Loan Approved!")
        else:
            st.subheader("Loan Not Approved.")

show_predict_page()
