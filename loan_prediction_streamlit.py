import streamlit as st
import numpy as np
import pickle

model = pickle.load(open('Linear_Model.pkl','rb'))
# Define the title of your Streamlit app
st.title('Loan Prediction System')

def main():
    # Collect input from the user

    #Gender
    gen_display = ('female', 'male')
    gender_options=list(range(len(gen_display)))
    gender = st.selectbox("Enter gender", gender_options,format_func=lambda x:gen_display[x])

    #married status
    married_display = ('Yes', 'No')
    married_options=list(range(len(gen_display)))
    married = st.selectbox("Enter marital status", married_options,format_func=lambda x:married_display[x])

    #number of dependents
    dependent_display = ('None', 'One', 'Two', 'More than Two')
    dep_option=list(range(len(dependent_display)))
    dependent = st.selectbox("Enter number of dependents",dep_option, format_func=lambda x:dependent_display[x])

    #education status
    education_display = ('Not Graduate', 'Graduate')
    education_option=list(range(len(education_display)))
    education = st.selectbox("Enter education", education_option,format_func=lambda x:education_display[x])

    self_employed_display = ('Yes', 'No')
    self_employed_option=list(range(len(self_employed_display)))
    self_employed = st.selectbox("Enter self-employment status", self_employed_option,format_func=lambda x:self_employed_display[x])

    property_area_display = ('Rural', 'Semiurban', 'Urban')
    property_area_option=list(range(len(property_area_display)))
    property_area = st.selectbox("Enter property area", property_area_option,format_func=lambda x:property_area_display[x])


    applicant_income = st.number_input("Enter applicant's income")

    coapplicant_income = st.number_input("Enter co-applicant's income")

    loan_amount = st.number_input("Enter loan amount")

   #loan duration
    dur_display=['2 months', '6 months', '8 months', '12 months','16 months']
    dur_option=range(len(dur_display))
    dur = st.selectbox("loan duartion",dur_option,format_func=lambda x:dur_display[x])

    credit_history = st.number_input("Enter credit history (0.0-1.0)")


    if st.button("Predict"):
         duration=0
         if dur==0:
            duration=60
         if dur==1:
            duration=180
         if dur==2:
            duration=240
         if dur==3:
            duration=360
         if dur==4:
            duration=480
         features=[[gender,married,dependent,education,self_employed,applicant_income,coapplicant_income,loan_amount,dur,credit_history,property_area ]]
         print(features)
         prediction=model.predict(features)
         lc=[str(i) for i in prediction]
         ans=int("".join(lc))
         if ans==0:
             st.error("you will not get loan from bank")
         else:
             st.success("you will get loan from bank")

main()
