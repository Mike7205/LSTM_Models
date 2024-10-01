# To jest wersja 9 z Arima
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, timedelta, date
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import appdirs as ad

st.set_page_config(layout="wide")

# start definicji strony
st.title('LSTM Prediction Models')

st.html(
    """
<style>
[data-testid="stSidebarContent"] {color: black; background-color: #91BFCF} #90EE90 #ADD8E6 #9CC2CF
</style>
""")

st.sidebar.title('Models list')
#st.sidebar.write('You selected:', comm)
today = date.today()

# Own LSTM EUR/PLN D+5 prediction model
st.subheader('Own LSTM EUR/PLN D+5 prediction model', divider='red')  
val_D5E = pd.read_pickle('D5_eur_tabel.pkl')
val_D5EP = val_D5E[['Date','Day + 5 Prediction']][-100:]
val_D5EU = pd.read_pickle('D1_EUR_a.pkl')
val_D5EUR = val_D5EU[['Date','EUR/PLN']][-100:]
day_es = val_D5EUR.shape[0]

st.write(f'Predictions for the last {day_es} days')  
fig_D5E = px.line(val_D5EP, x='Date', y=['Day + 5 Prediction'],color_discrete_map={'Day + 5 Prediction':'red'}, width=1000, height=500)
fig_D5E.add_trace(go.Scatter(x=val_D5EUR['Date'], y=val_D5EUR['EUR/PLN'], mode='lines', name='EUR/PLN', line=dict(color='blue')))
fig_D5E.update_layout(plot_bgcolor='white',showlegend=True,xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='Lightgrey'),
                      yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='Lightgrey'))
fig_D5E.add_vline(x = today,line_width=1, line_dash="dash", line_color="black")
st.plotly_chart(fig_D5E)

# Own EUR/PLN LSTM prediction model (D+1)    
st.subheader('EUR/PLN exchange rate (D+1) predictions', divider='red')
val = pd.read_pickle('D1_EUR_a.pkl')
val_1 = val[['Date','EUR/PLN','Day + 1 Prediction']][-100:]      #.iloc[:-1]
day_s = val_1.shape[0]
st.write(f'Predictions for the last {day_s} days')
fig_val = px.line(val_1, x='Date', y=['EUR/PLN','Day + 1 Prediction'],color_discrete_map={
                 'EUR/PLN':'blue','Day + 1 Prediction':'red'}, width=1000, height=500 ) 
fig_val.update_layout(plot_bgcolor='white',showlegend=True,xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='Lightgrey'),
                      yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='Lightgrey'))   
st.plotly_chart(fig_val, use_container_width=True)

# Own USD/PLN LSTM prediction model (D+1)     
st.subheader('USD/PLN exchange rate (D+1) predictions', divider='red')
val_s = pd.read_pickle('D1_USD_a.pkl')
val_s1 = val_s[['Date','USD/PLN','Day + 1 Prediction']][-100:]      #.iloc[:-1]
day_s1 = val_s1.shape[0]
st.write(f'Predictions for the last {day_s1} days')
fig_vals = px.line(val_s1, x='Date', y=['USD/PLN','Day + 1 Prediction'],color_discrete_map={
                 'USD/PLN':'#03B303','Day + 1 Prediction':'#D9017A'}, width=1000, height=500 ) # #89CFF0
fig_vals.update_layout(plot_bgcolor='white',showlegend=True,xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='Lightgrey'),
                      yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='Lightgrey'))   
st.plotly_chart(fig_vals, use_container_width=True)

# LLM model - stock news     
#st.subheader('AI about the markets', divider='red')
#from transformers import pipeline

# Inicjalizacja modelu
#generator = pipeline('text-generation', model='gpt2', framework='pt')
# Pole tekstowe do wprowadzenia promptu
#prompt = st.text_input("How about:", "Latest insides")

#if st.button("Check"):
    # Generowanie tekstu
 #   result = generator(prompt, max_length=100, num_return_sequences=1)
 #   st.write(result[0]['generated_text'])

