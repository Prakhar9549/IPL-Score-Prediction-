import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Streamlit app
st.title("IPL Score Prediction")

# Form for input
with st.form("prediction_form"):
    st.header("Enter details")
    
    runs = st.number_input("Current Runs", min_value=0, value=0)
    wickets = st.number_input("Current Wickets", min_value=0, max_value=10, value=0)
    overs = st.number_input("Current Overs", min_value=0, value=0)
    runs_last_5 = st.number_input("Runs in last 5 Overs", min_value=0, value=0)
    wickets_last_5 = st.number_input("Wickets in last 5 Overs", min_value=0, max_value=10, value=0)
    bat_team = st.selectbox("Batting Team", options=["Select Team", "Kolkata Knight Riders", 'Chennai Super Kings', 'Rajasthan Royals',
       'Mumbai Indians', 'Royal Challengers Bangalore', 'Kings XI Punjab','Sunrisers Hyderabad'])
    bowl_team = st.selectbox("Bowling Team", options=["bowl_team", "Kolkata Knight Riders", 'Chennai Super Kings', 'Rajasthan Royals',
       'Mumbai Indians', 'Royal Challengers Bangalore', 'Kings XI Punjab','Sunrisers Hyderabad'])
   
    submit_button = st.form_submit_button("Predict")

if submit_button:
    # Check if any placeholder options are still selected
    if (bat_team == "Select Team" or bowl_team == "bowl_team"):
        st.error("Please fill in all the fields.")
    elif (runs<runs_last_5):
        st.error("Please fill correct last 5overs Runs.")
    elif (wickets<wickets_last_5):
        st.error("Please fill correct last 5overs Wickets.")    
    else:
        data = CustomData(
            runs=runs,
            wickets=wickets,
            overs=overs,
            runs_last_5=runs_last_5,
            wickets_last_5=wickets_last_5,
            bat_team=bat_team,
            bowl_team=bowl_team
        )
        pred_df = data.get_data_as_data_frame()

        st.write("Input DataFrame:")
        st.write(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        st.write("Prediction Results:")
        st.write(int(results[0]))
